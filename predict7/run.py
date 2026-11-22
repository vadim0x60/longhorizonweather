#!/usr/bin/env python3
"""predict7: Extreme-Value & Tail-Risk Contrarian (round 2).

Builds a full predictive distribution (a quantile table) for every
(location, window, variable) target in year X+1 from data dated <= X-12-31,
then emits sharp one-sided threshold claims and booleans on the targets rivals
use most, priced so each claim holds with a calibrated probability.

Distributions (stdlib only):
  * Block maxima/minima (temp_max_c, temp_min_c): GEV fitted by L-moments to
    yearly block extremes after a shrunken Theil-Sen trend shift to year X+1.
  * Daily extremes (precip_max_day_mm, wind_max_kmh): average of block-max GEV
    and declustered peaks-over-threshold (Poisson-GPD) quantiles.
  * Window means (temp_mean_c, temp_mean_max_c, temp_mean_min_c): Gaussian
    kernel mixture over trend-shifted yearly values.
  * Skewed, non-negative totals and day counts (precip_sum_mm, hot/frost/rainy
    days): the same kernel mixture in sqrt space, so the right tail is heavy and
    the median sits below the mean, as it does in the data.
All spreads are widened by factors set in a 2010-2023 hindcast.

Claims: "gt"/"lt" thresholds at the quantile where the claim holds with
probability P_SIDE, pointing to the side of climatology our forecast leans to,
and warmer/wetter-than-year-X booleans when P >= P_BOOL. When another
predictor's claim falls on the far side of our threshold, our claim holds with
at least P_SIDE probability and theirs with at most 1 - P_SIDE.
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
MAX_CLAIMS = 150
TEMP_TREND_PRIOR = 0.025  # degC / yr

CFG = {
    "P_SIDE": 0.62,      # probability our one-sided threshold claims hold
    "P_BOOL": 0.70,      # minimum probability for a boolean claim
    "KERNEL": 0.6,       # kernel bandwidth as a fraction of Silverman's rule
    # Spread multipliers (lower side, upper side) around the median.
    "WIDEN": {
        "temp_max_c": (1.1, 1.5), "temp_min_c": (1.35, 1.25),
        "precip_max_day_mm": (0.95, 1.6), "wind_max_kmh": (0.95, 1.75),
        "hot_days_ge_30c": (1.0, 1.55), "frost_days_lt_0c": (1.2, 1.2),
        "rainy_days_ge_1mm": (1.05, 1.35), "precip_sum_mm": (1.0, 1.5),
        "temp_mean_c": (1.3, 1.25), "temp_mean_max_c": (1.1, 1.5),
        "temp_mean_min_c": (1.45, 1.1),
    },
}

# Probability grid of the quantile tables.
GRID = [0.0025 + i * (0.995 / 200) for i in range(201)]


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


def stdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def theil_sen(ts, ys):
    slopes = [(ys[j] - ys[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts)) if ts[j] != ts[i]]
    return median(slopes) if slopes else 0.0


def shrunk_trend(ts, ys, prior, weight=0.5, lo=-1e9, hi=1e9):
    if len(ts) < 6:
        return prior
    return max(lo, min(hi, weight * theil_sen(ts, ys) + (1 - weight) * prior))


def lmoments(xs):
    s = sorted(xs)
    n = len(s)
    b0 = sum(s) / n
    b1 = sum(i * s[i] for i in range(n)) / (n * (n - 1))
    b2 = sum(i * (i - 1) * s[i] for i in range(n)) / (n * (n - 1) * (n - 2))
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def gev_fit(xs):
    """Hosking (1985) L-moment GEV; returns (loc, scale, k) with k = -shape."""
    l1, l2, l3 = lmoments(xs)
    if l2 <= 1e-9:
        return l1, 0.0, 0.0
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - math.log(2) / math.log(3)
    k = max(-0.4, min(0.45, 7.8590 * c + 2.9554 * c * c))
    if abs(k) < 1e-6:
        a = l2 / math.log(2)
        return l1 - 0.5772156649 * a, a, 0.0
    a = l2 * k / ((1 - 2 ** (-k)) * math.gamma(1 + k))
    return l1 - a * (1 - math.gamma(1 + k)) / k, a, k


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
    l2 = 2 * sum(i * s[i] for i in range(n)) / (n * (n - 1)) - m
    if l2 <= 1e-9:
        return m, 0.0
    k = max(-0.4, min(0.4, m / l2 - 2))
    return (1 + k) * m, k


def pot_model(daily_by_year, per_year=5.0, run=1):
    """Declustered POT exceedances -> (u, lam, scale, k) or None."""
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
    a, k = gpd_fit(peaks)
    return (u, len(peaks) / nyr, a, k) if a > 0 else None


def pot_q(pm, p):
    u, lam, a, k = pm
    S = -math.log(p) / lam
    if S >= 1:
        return None
    if abs(k) < 1e-6:
        return u - a * math.log(S)
    return u + a / k * (1 - S ** k)


def norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def kernel_table(points, h):
    """Quantile table of an equal-weight Gaussian mixture (CDF grid inversion)."""
    lo_b = min(points) - 6 * h
    hi_b = max(points) + 6 * h
    m = 400
    xs = [lo_b + (hi_b - lo_b) * i / m for i in range(m + 1)]
    n = len(points)
    fs = [sum(norm_cdf((x - c) / h) for c in points) / n for x in xs]
    out, j = [], 0
    for p in GRID:
        while j < m - 1 and fs[j + 1] < p:
            j += 1
        span = fs[j + 1] - fs[j]
        f = 0.0 if span <= 0 else min(1.0, max(0.0, (p - fs[j]) / span))
        out.append(xs[j] + f * (xs[j + 1] - xs[j]))
    return out


class Dist:
    """Predictive distribution as a monotone quantile table on GRID."""

    def __init__(self, qs, why, lower=None, upper=None):
        self.qs = qs
        self.why = why
        self.lower, self.upper = lower, upper

    def q(self, p):
        p = min(GRID[-1], max(GRID[0], p))
        x = (p - GRID[0]) / (GRID[1] - GRID[0])
        i = min(len(GRID) - 2, int(x))
        v = self.qs[i] + (x - i) * (self.qs[i + 1] - self.qs[i])
        if self.lower is not None:
            v = max(self.lower, v)
        if self.upper is not None:
            v = min(self.upper, v)
        return v

    def cdf(self, v):
        """P(Y <= v)."""
        qs = self.qs
        if v < qs[0]:
            return GRID[0] / 2
        if v >= qs[-1]:
            return 1 - (1 - GRID[-1]) / 2
        lo, hi = 0, len(qs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if qs[mid] <= v:
                lo = mid
            else:
                hi = mid
        span = qs[hi] - qs[lo]
        f = 0.0 if span <= 0 else (v - qs[lo]) / span
        return GRID[lo] + f * (GRID[hi] - GRID[lo])


def widen_table(var, qs):
    a, b = CFG["WIDEN"].get(var, (1.2, 1.2))
    med = qs[len(qs) // 2]
    return [med + (a if v < med else b) * (v - med) for v in qs]


# ------------------------------------------------------------------- windows
# Windows rivals target most (template popularity, not outcomes), with weight.
WINDOWS = [
    ("FY", "01-01", "12-31", 3.0),
    ("JJA", "06-01", "08-31", 2.5),
    ("MAM", "03-01", "05-31", 1.6),
    ("SON", "09-01", "11-30", 1.6),
    ("JF", "01-01", "02-28", 1.3),  # ends 02-29 in leap target years
    ("Q1", "01-01", "03-31", 1.0),
    ("Q4", "10-01", "12-31", 1.0),
    ("JUL", "07-01", "07-31", 0.9),
    ("JAN", "01-01", "01-31", 0.9),
]

# variable: (field index, family, weight)
VARS = {
    "temp_mean_c": (0, "mean", 1.0),
    "warmer_than_year_x": (0, "bool", 1.0),
    "precip_sum_mm": (3, "total", 1.0),
    "wetter_than_year_x": (3, "bool", 0.9),
    "temp_mean_max_c": (1, "mean", 0.8),
    "temp_mean_min_c": (2, "mean", 0.8),
    "hot_days_ge_30c": (1, "count", 0.9),
    "frost_days_lt_0c": (2, "count", 0.9),
    "rainy_days_ge_1mm": (3, "count", 0.8),
    "temp_max_c": (1, "blockmax", 0.9),
    "temp_min_c": (2, "blockmin", 0.6),
    "precip_max_day_mm": (3, "dailymax", 0.6),
    "wind_max_kmh": (4, "dailymax", 0.4),
}
COUNT_TEST = {"hot_days_ge_30c": lambda v: v >= 30.0,
              "frost_days_lt_0c": lambda v: v < 0.0,
              "rainy_days_ge_1mm": lambda v: v >= 1.0}


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def window_series(days, start_md, end_md, years):
    """Per historical year, (values rows, expected day count) inside the MM-DD span."""
    out = {}
    for y in years:
        d0 = dt.date(y, int(start_md[:2]), int(start_md[3:]))
        e_day = int(end_md[3:])
        if end_md[:2] == "02" and e_day >= 28:
            e_day = 29 if is_leap(y) else 28
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


def stat(var, vals):
    if var in ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c"):
        return sum(vals) / len(vals)
    if var == "precip_sum_mm":
        return sum(vals)
    if var in COUNT_TEST:
        f = COUNT_TEST[var]
        return float(sum(1 for v in vals if f(v)))
    if var in ("temp_max_c", "precip_max_day_mm", "wind_max_kmh"):
        return max(vals)
    if var == "temp_min_c":
        return min(vals)
    raise ValueError(var)


# --------------------------------------------------------------------- models
def yearly(var, ser):
    idx = VARS[var][0]
    out = {}
    for y, (seq, n_exp) in ser.items():
        vals = [r[idx] for r in seq if r[idx] is not None]
        if len(vals) >= 0.97 * n_exp:
            out[y] = (vals, n_exp)
    return out


def distribution(var, ser, target_year):
    """Predictive Dist for var over the window in target_year, or None."""
    fam = VARS[var][1]
    yr = yearly(var, ser)
    years = sorted(yr)
    if len(years) < 8:
        return None
    n = len(years)
    n_target = max(n_exp for _, n_exp in yr.values())
    ys = [stat(var, yr[y][0]) for y in years]

    if fam in ("blockmax", "blockmin"):
        sign = 1.0 if fam == "blockmax" else -1.0
        zs = [sign * v for v in ys]
        b = shrunk_trend(years, zs, sign * TEMP_TREND_PRIOR,
                         lo=-0.03 if sign > 0 else -0.10, hi=0.10 if sign > 0 else 0.03)
        par = gev_fit([v + b * (target_year - y) for v, y in zip(zs, years)])
        infl = math.sqrt(1 + 1.0 / n)
        med = gev_q(par, 0.5)
        qs = [med + infl * (gev_q(par, p) - med) for p in GRID]
        if sign < 0:
            qs = [-v for v in reversed(qs)]
        return Dist(widen_table(var, qs), f"GEV (L-moments, k={par[2]:.2f}) on {n} "
                    f"trend-shifted block {'maxima' if sign > 0 else 'minima'}")

    if fam == "dailymax":
        par = gev_fit(ys)
        daily = {y: yr[y][0] for y in years}
        per_year = max(2.0, 5.0 * n_target / 365.0) if n_target < 300 else 5.0
        pm = pot_model(daily, per_year=per_year, run=1 if var == "wind_max_kmh" else 0)
        raw = []
        for p in GRID:
            g = gev_q(par, p)
            pq = pot_q(pm, p) if pm else None
            raw.append(g if pq is None else 0.5 * (g + pq))
        for i in range(1, len(raw)):
            raw[i] = max(raw[i], raw[i - 1])
        med = raw[len(raw) // 2]
        infl = math.sqrt(1 + 1.0 / n)
        qs = widen_table(var, [med + infl * (v - med) for v in raw])
        return Dist(qs, f"block-max GEV (k={par[2]:.2f}) blended with declustered POT-GPD",
                    lower=0.0)

    # Kernel-mixture families.
    if fam == "mean":
        b = shrunk_trend(years, ys, TEMP_TREND_PRIOR, lo=-0.03, hi=0.08)
        pts = [v + b * (target_year - y) for v, y in zip(ys, years)]
        fwd, back = (lambda v: v), (lambda v: v)
        why = f"kernel mixture of {n} trend-shifted ({b:+.3f}/yr) window means"
        lower = upper = None
    else:  # total / count: sqrt space, trend shrunk toward zero
        b = shrunk_trend(years, ys, 0.0)
        pts_raw = [max(0.0, v + b * (target_year - y)) for v, y in zip(ys, years)]
        if max(pts_raw) <= 0:
            return Dist([0.0] * len(GRID), f"zero in all {n} historical windows", 0.0, 0.0)
        pts = [math.sqrt(v) for v in pts_raw]
        fwd, back = math.sqrt, (lambda v: max(0.0, v) ** 2)
        why = f"sqrt-space (skew-aware) kernel mixture of {n} trend-shifted ({b:+.2f}/yr) values"
        lower, upper = 0.0, (float(n_target) if fam == "count" else None)
    sd = stdev(pts)
    if sd <= 1e-9:
        sd = 0.05 * (abs(pts[0]) + 1)
    h = CFG["KERNEL"] * 1.06 * sd * n ** -0.2
    infl = math.sqrt(1 + 1.0 / n)
    tq = kernel_table(pts, h)
    med = tq[len(tq) // 2]
    tq = [med + infl * (v - med) for v in tq]
    qs = widen_table(var, [back(v) for v in tq])
    return Dist(qs, why, lower, upper)


# --------------------------------------------------------------------- claims
def fmt(v, var):
    return int(v) if VARS[var][1] == "count" else round(v, 1)


def numeric_claim(var, dist, clim_med):
    """One-sided threshold on the side of climatology our median leans to."""
    p = CFG["P_SIDE"]
    med = dist.q(0.5)
    if VARS[var][1] == "count":
        if med >= clim_med:
            v = math.floor(dist.q(1 - p) - 1e-9)  # gt v  <=>  >= v+1
            prob = 1 - dist.cdf(v + 0.5)
            return {"kind": "threshold", "op": "gt", "value": int(v)}, prob
        v = math.ceil(dist.q(p) + 1e-9)  # lt v  <=>  <= v-1
        prob = dist.cdf(v - 0.5)
        return {"kind": "threshold", "op": "lt", "value": int(v)}, prob
    if med >= clim_med:
        v = math.floor(dist.q(1 - p) * 10) / 10
        return {"kind": "threshold", "op": "gt", "value": v}, 1 - dist.cdf(v)
    v = math.ceil(dist.q(p) * 10) / 10
    return {"kind": "threshold", "op": "lt", "value": v}, dist.cdf(v)


def build_claims(year_x, data_dir):
    rows, meta = load(data_dir)
    target = year_x + 1
    hist_years = list(range(max(1980, year_x - 15), year_x + 1))
    cands = []
    for lid in sorted(rows):
        days = {d: v for d, v in rows[lid].items() if d <= f"{year_x}-12-31"}
        lat, lon = meta[lid]
        for tag, s_md, e_md, wprio in WINDOWS:
            if e_md == "02-28" and is_leap(target):
                e_md = "02-29"
            ws = f"{target}-{s_md}"
            we = f"{target}-{e_md}"
            ser = window_series(days, s_md, e_md, hist_years)
            for var, (_, fam, vprio) in VARS.items():
                if fam == "bool":
                    base = "temp_mean_c" if var == "warmer_than_year_x" else "precip_sum_mm"
                    dist = distribution(base, ser, target)
                    yr = yearly(base, ser)
                    if dist is None or year_x not in yr:
                        continue
                    ref = stat(base, yr[year_x][0])
                    p_true = 1 - dist.cdf(ref)
                    val = p_true >= 0.5
                    prob = max(p_true, 1 - p_true)
                    if prob < CFG["P_BOOL"]:
                        continue
                    assertion = {"kind": "boolean", "value": val}
                    why = f"P({base} X+1 > X={ref:.1f}) = {p_true:.2f} from {dist.why}"
                    edge = prob - 0.5
                else:
                    dist = distribution(var, ser, target)
                    if dist is None or dist.q(0.05) == dist.q(0.95):
                        continue
                    yr = yearly(var, ser)
                    clim = median([stat(var, yr[y][0]) for y in yr])
                    assertion, prob = numeric_claim(var, dist, clim)
                    if prob > CFG["P_SIDE"] + 0.2 or assertion["value"] < 0:
                        continue  # degenerate (e.g. "gt -1" hot days): nobody disagrees
                    lean = (dist.q(0.5) - clim) / max(1e-6, dist.q(0.75) - dist.q(0.25))
                    why = f"{dist.why}; median {dist.q(0.5):.1f} vs climatology {clim:.1f}"
                    edge = prob - 0.5 + 0.3 * min(1.0, abs(lean))
                cands.append((wprio * vprio * (0.5 + edge), {
                    "target": {"location_id": lid, "lat": lat, "lon": lon,
                               "window_start": ws, "window_end": we, "variable": var},
                    "assertion": assertion,
                    "confidence": round(prob, 3),
                    "rationale": f"EVT/tail-risk: {why}; window {tag}.",
                }))
    cands.sort(key=lambda c: -c[0])
    claims, seen = [], set()
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
