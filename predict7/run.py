#!/usr/bin/env python3
"""predict7: Extreme-Value & Tail-Risk Contrarian.

Emits calibrated interval/threshold claims for year X+1 on tail metrics
(annual/seasonal maxima and minima, extreme daily precipitation and wind,
threshold-exceedance day counts) using only `--data-dir` (dated <= X-12-31).

Models (stdlib only):
  * Block maxima/minima (temp_max_c, temp_min_c): GEV fitted by L-moments to
    yearly block extremes after a shrunken Theil-Sen trend shift to year X+1.
  * Daily extremes (precip_max_day_mm, wind_max_kmh): average of the block-max
    GEV quantile and a peaks-over-threshold model (declustered GPD
    exceedances, Poisson arrivals) converted to the block-maximum law.
  * Day counts (hot/frost/rainy days): trend-shifted empirical quantiles with
    small-sample widening and a binomial-noise floor.
  * Window precipitation totals: skewed empirical quantiles (gamma-like tail).
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
from collections import defaultdict

FIELDS = ("temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max")

# Nominal central coverage of interval claims (calibrated by backtest).
Q_LO, Q_HI = 0.09, 0.91
# Extra width multipliers on (quantile - median), (lower side, upper side) in
# physical units: short-record fits under-disperse, upper tails especially.
# Tuned by a 2010-2023 hindcast (20 locations, ERA5 via Open-Meteo): raw
# coverage was 0.76 with 2:1 upper-side misses; these give ~0.86-0.90.
WIDEN = {
    "default": (1.2, 1.2),
    "temp_max_c": (1.15, 1.6),
    "temp_min_c": (1.35, 1.35),
    "precip_max_day_mm": (1.1, 1.6),
    "wind_max_kmh": (1.15, 1.8),
    "hot_days_ge_30c": (1.1, 1.8),
    "frost_days_lt_0c": (1.4, 1.2),
    "rainy_days_ge_1mm": (1.1, 1.6),
    "precip_sum_mm": (1.2, 1.2),
}
CONF = 0.85  # hindcast hit rate of the widened intervals


def widen(var, med, lo, hi):
    a, b = WIDEN.get(var, WIDEN["default"])
    return med + a * (lo - med), med + b * (hi - med)
MAX_CLAIMS = 150

# Prior warming rate of temperature extremes (degC / yr) used for shrinkage.
TEMP_TREND_PRIOR = 0.025


# --------------------------------------------------------------------------- io
def load(data_dir):
    rows = defaultdict(dict)
    meta = {}
    for name in ("daily_history.csv", "daily_year_x.csv"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                lid = r["location_id"]
                meta[lid] = (float(r["lat"]), float(r["lon"]))
                vals = []
                for k in FIELDS:
                    v = r.get(k, "")
                    try:
                        vals.append(float(v) if v not in ("", "nan", "NaN", "None") else None)
                    except ValueError:
                        vals.append(None)
                rows[lid][r["date"][:10]] = vals
    return rows, meta


# ------------------------------------------------------------------ statistics
def quantile(xs, p):
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def median(xs):
    return quantile(xs, 0.5)


def theil_sen(ts, ys):
    slopes = [(ys[j] - ys[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts)) if ts[j] != ts[i]]
    return median(slopes) if slopes else 0.0


def lmoments(xs):
    s = sorted(xs)
    n = len(s)
    b0 = sum(s) / n
    b1 = sum(i * s[i] for i in range(n)) / (n * (n - 1))
    b2 = sum(i * (i - 1) * s[i] for i in range(n)) / (n * (n - 1) * (n - 2))
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return l1, l2, l3


def gev_fit(xs):
    """Hosking (1985) L-moment GEV; returns (loc, scale, k) with k = -shape."""
    l1, l2, l3 = lmoments(xs)
    if l2 <= 1e-9:
        return l1, 0.0, 0.0
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - math.log(2) / math.log(3)
    k = 7.8590 * c + 2.9554 * c * c
    k = max(-0.4, min(0.45, k))  # 16 blocks: keep the shape physically sane
    if abs(k) < 1e-6:
        a = l2 / math.log(2)
        return l1 - 0.5772156649 * a, a, 0.0
    a = l2 * k / ((1 - 2 ** (-k)) * math.gamma(1 + k))
    loc = l1 - a * (1 - math.gamma(1 + k)) / k
    return loc, a, k


def gev_q(par, p):
    loc, a, k = par
    y = -math.log(p)
    if abs(k) < 1e-6:
        return loc - a * math.log(y)
    return loc + a / k * (1 - y ** k)


def gpd_fit(exc):
    """L-moment GPD with known lower bound 0 (Hosking & Wallis 1987)."""
    n = len(exc)
    m = sum(exc) / n
    s = sorted(exc)
    b1 = sum(i * s[i] for i in range(n)) / (n * (n - 1))
    l2 = 2 * b1 - m
    if l2 <= 1e-9:
        return m, 0.0
    k = m / l2 - 2
    k = max(-0.4, min(0.4, k))
    return (1 + k) * m, k


def pot_block_q(daily_by_year, p, per_year=5.0, run=1):
    """Block-max quantile from declustered POT exceedances (Poisson-GPD)."""
    years = [y for y in daily_by_year if daily_by_year[y]]
    nyr = len(years)
    allv = sorted(v for y in years for v in daily_by_year[y])
    m = int(round(per_year * nyr))
    if nyr < 5 or len(allv) <= m + 2:
        return None
    u = allv[-m - 1]
    peaks = []
    for y in years:
        seq = daily_by_year[y]
        i = 0
        while i < len(seq):
            if seq[i] > u:
                j, best, gap = i, seq[i], 0
                while j + 1 < len(seq) and (seq[j + 1] > u or gap < run):
                    j += 1
                    if seq[j] > u:
                        best = max(best, seq[j])
                        gap = 0
                    else:
                        gap += 1
                peaks.append(best - u)
                i = j + 1
            else:
                i += 1
    if len(peaks) < 10:
        return None
    lam = len(peaks) / nyr
    a, k = gpd_fit(peaks)
    S = -math.log(p) / lam  # conditional exceedance probability
    if S >= 1 or a <= 0:
        return None
    if abs(k) < 1e-6:
        return u - a * math.log(S)
    return u + a / k * (1 - S ** k)


# ------------------------------------------------------------------- windows
def windows(year):
    """(tag, start, end, priority) windows inside the target year."""
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return [
        ("FY", f"{year}-01-01", f"{year}-12-31", 3.0),
        ("JJA", f"{year}-06-01", f"{year}-08-31", 2.0),
        ("DJF1", f"{year}-01-01", f"{year}-02-{29 if leap else 28}", 1.0),
    ]


def window_series(days, start_md, end_md, years):
    """Per historical year, list of daily value rows inside the same MM-DD span."""
    out = {}
    for y in years:
        d0 = dt.date(y, int(start_md[:2]), int(start_md[3:]) if start_md != "02-29" else 28)
        e_day = int(end_md[3:])
        if end_md[:2] == "02" and e_day >= 28:
            e_day = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
        d1 = dt.date(y, int(end_md[:2]), e_day)
        n_exp = (d1 - d0).days + 1
        seq = []
        d = d0
        while d <= d1:
            r = days.get(d.isoformat())
            if r is not None:
                seq.append(r)
            d += dt.timedelta(days=1)
        if len(seq) >= 0.97 * n_exp:
            out[y] = (seq, n_exp)
    return out


# --------------------------------------------------------------------- models
VARS = {
    # name: (field index, kind, priority weight)
    "temp_max_c": (1, "blockmax", 1.0),
    "temp_min_c": (2, "blockmin", 1.0),
    "precip_max_day_mm": (3, "dailymax", 1.0),
    "wind_max_kmh": (4, "dailymax", 0.9),
    "hot_days_ge_30c": (1, "count_ge30", 0.95),
    "frost_days_lt_0c": (2, "count_lt0", 0.95),
    "rainy_days_ge_1mm": (3, "count_ge1", 0.9),
    "precip_sum_mm": (3, "total", 0.6),
}


def shrunk_trend(ts, ys, prior, weight=0.5, lo=-1e9, hi=1e9):
    if len(ts) < 6:
        return prior
    return max(lo, min(hi, weight * theil_sen(ts, ys) + (1 - weight) * prior))


def model(var, ser, target_year):
    idx, kind, _ = VARS[var]
    years = sorted(ser)
    if len(years) < 8:
        return None
    vals = {y: [r[idx] for r in ser[y][0] if r[idx] is not None] for y in years}
    n_days = {y: ser[y][1] for y in years}
    if any(len(vals[y]) < 0.97 * n_days[y] for y in years):
        years = [y for y in years if len(vals[y]) >= 0.97 * n_days[y]]
        if len(years) < 8:
            return None
    n_target = max(n_days.values())

    if kind in ("blockmax", "blockmin"):
        sign = 1.0 if kind == "blockmax" else -1.0
        ys = [sign * (max(vals[y]) if sign > 0 else min(vals[y])) for y in years]
        # Warm extremes rise (+); cold extremes rise too (-> negated series falls).
        b = shrunk_trend(years, ys, sign * TEMP_TREND_PRIOR, lo=-0.03 if sign > 0 else -0.10,
                         hi=0.10 if sign > 0 else 0.03)
        shifted = [v + b * (target_year - y) for v, y in zip(ys, years)]
        par = gev_fit(shifted)
        n = len(shifted)
        infl = math.sqrt(1 + 1.0 / n)
        med = gev_q(par, 0.5)
        qlo = med + infl * (gev_q(par, Q_LO) - med)
        qhi = med + infl * (gev_q(par, Q_HI) - med)
        if sign < 0:
            qlo, qhi, med = -qhi, -qlo, -med
        qlo, qhi = widen(var, med, qlo, qhi)
        return {"kind": "interval", "low": round(qlo, 1), "high": round(qhi, 1),
                "conf": CONF, "degenerate": False,
                "why": f"GEV(L-mom, k={par[2]:.2f}) on {n} trend-shifted ({b * sign:+.3f}/yr) block "
                       f"{'maxima' if sign > 0 else 'minima'}"}

    if kind == "dailymax":
        ys = [max(vals[y]) for y in years]
        par = gev_fit(ys)
        daily = {y: vals[y] for y in years}
        per_year = 5.0 * n_target / 365.0 if n_target < 300 else 5.0
        per_year = max(2.0, per_year)
        run = 1 if var == "wind_max_kmh" else 0
        out = []
        for p in (Q_LO, 0.5, Q_HI):
            g = gev_q(par, p)
            pq = pot_block_q(daily, p, per_year=per_year, run=run)
            out.append(g if pq is None else 0.5 * (g + pq))
        med = out[1]
        infl = math.sqrt(1 + 1.0 / len(ys))
        lo = max(0.0, med + infl * (out[0] - med))
        hi = med + infl * (out[2] - med)
        lo, hi = widen(var, med, lo, hi)
        lo = max(0.0, lo)
        return {"kind": "interval", "low": round(lo, 1), "high": round(hi, 1),
                "conf": CONF, "degenerate": hi - lo < 0.2,
                "why": f"blend of block-max GEV (k={par[2]:.2f}) and declustered POT-GPD "
                       f"annual-max law over {len(ys)} yrs"}

    if kind.startswith("count"):
        if kind == "count_ge30":
            f = lambda v: v >= 30.0
            prior = 0.0
        elif kind == "count_lt0":
            f = lambda v: v < 0.0
            prior = 0.0
        else:
            f = lambda v: v >= 1.0
            prior = 0.0
        ys = [float(sum(1 for v in vals[y] if f(v))) for y in years]
        b = shrunk_trend(years, ys, prior)
        shifted = [max(0.0, v + b * (target_year - y)) for v, y in zip(ys, years)]
        n = len(shifted)
        med = median(shifted)
        infl = math.sqrt(1 + 1.0 / n) * 1.05
        lo = med + infl * (quantile(shifted, Q_LO) - med)
        hi = med + infl * (quantile(shifted, Q_HI) - med)
        lo, hi = widen(var, med, lo, hi)
        # Binomial sampling floor on the half-width (days are noisy events).
        pday = min(0.999, max(1e-6, med / n_target))
        floor = 1.34 * math.sqrt(n_target * pday * (1 - pday))
        lo = min(lo, med - floor)
        hi = max(hi, med + floor)
        lo = max(0, int(math.floor(lo)))
        hi = min(n_target, int(math.ceil(hi)))
        if max(ys) == 0:
            # Never observed in the record: one-sided "le 0"-style claim.
            return {"kind": "threshold", "op": "le", "value": 0, "conf": 0.95, "degenerate": True,
                    "why": f"zero exceedances in all {n} historical windows"}
        if min(ys) >= n_target:
            return {"kind": "threshold", "op": "ge", "value": n_target, "conf": 0.95,
                    "degenerate": True, "why": f"every day exceeded in all {n} historical windows"}
        return {"kind": "interval", "low": lo, "high": hi, "conf": CONF,
                "degenerate": False,
                "why": f"trend-shifted ({b:+.2f}/yr) empirical count quantiles, {n} yrs, "
                       f"binomial floor"}

    if kind == "total":
        ys = [sum(vals[y]) for y in years]
        # Window totals are right-skewed: fit quantiles in log space.
        lg = [math.log(max(v, 0.1)) for v in ys]
        n = len(lg)
        med = median(lg)
        infl = math.sqrt(1 + 1.0 / n) * 1.05
        lo = math.exp(med + infl * (quantile(lg, Q_LO) - med))
        hi = math.exp(med + infl * (quantile(lg, Q_HI) - med))
        lo, hi = widen(var, math.exp(med), lo, hi)
        return {"kind": "interval", "low": round(lo, 0), "high": round(hi, 0),
                "conf": CONF, "degenerate": hi - lo < 1,
                "why": f"log-space (skew-aware) empirical quantiles of {n} window totals"}
    return None


# ----------------------------------------------------------------------- main
def build_claims(year_x, data_dir):
    rows, meta = load(data_dir)
    target = year_x + 1
    hist_years = list(range(max(1980, year_x - 15), year_x + 1))
    cands = []
    for lid in sorted(rows):
        days = {d: v for d, v in rows[lid].items() if d <= f"{year_x}-12-31"}
        lat, lon = meta[lid]
        for tag, ws, we, wprio in windows(target):
            ser = window_series(days, ws[5:], we[5:], hist_years)
            for var, (_, _, vprio) in VARS.items():
                if tag != "FY" and var == "precip_sum_mm":
                    continue
                m = model(var, ser, target)
                if m is None:
                    continue
                prio = wprio * vprio * (0.15 if m["degenerate"] else 1.0)
                if m["kind"] == "interval":
                    assertion = {"kind": "interval", "low": m["low"], "high": m["high"]}
                else:
                    assertion = {"kind": "threshold", "op": m["op"], "value": m["value"]}
                cands.append((prio, {
                    "target": {"location_id": lid, "lat": lat, "lon": lon,
                               "window_start": ws, "window_end": we, "variable": var},
                    "assertion": assertion,
                    "confidence": round(m["conf"], 2),
                    "rationale": f"EVT/tail model: {m['why']}; window {tag}.",
                }))
    cands.sort(key=lambda c: -c[0])
    claims = []
    seen = set()
    for _, c in cands:
        t = c["target"]
        key = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if key in seen:
            continue
        seen.add(key)
        claims.append(dict(claim_id=f"p7_{len(claims) + 1:03d}", **c))
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    claims = build_claims(a.year_x, a.data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)
    print(f"predict7: wrote {len(claims)} claims for year {a.year_x + 1} -> {a.out}")


if __name__ == "__main__":
    main()
