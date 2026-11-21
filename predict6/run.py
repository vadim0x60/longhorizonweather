#!/usr/bin/env python3
"""predict6 — The Spectral & Oscillatory Cycle Theorist.

Models each yearly target series y_t (one value per trailing year <= X) as

    y_t = level + trend * t + cycle(t) + AR(1) anomaly + noise

* trend: Theil-Sen slope, shrunk toward a physical prior (warming for temps).
* AR(1): lag-1 autocorrelation of detrended residuals (damped persistence),
  so an extreme year X is mostly expected to mean-revert in X+1.
* cycle: strongest periodogram harmonic with period 2..8 years, added only
  when it passes a Fisher g-test, with amplitude shrunk.
* noise: residual spread inflated for parameter uncertainty and calibrated
  by rolling-origin backtests (see INFLATE).

Claims are 80%-ish intervals around the forecast, booleans for
warmer/wetter-than-year-X when the forecast is far from the year-X value
(mean-reversion bets), and categorical hottest/wettest month when the
climatological mode is dominant. Stdlib only; reads only --data-dir.
"""
import argparse
import calendar
import csv
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

ND = NormalDist()
MAX_CLAIMS = 150
INFLATE = 1.15  # residual-sigma inflation, calibrated by rolling backtests
COVER = 0.80    # target interval coverage
CYCLE_P = 0.02      # Fisher g-test gate for the spectral cycle
CYCLE_SHRINK = 0.25  # amplitude shrinkage of the extrapolated cycle
PHI_SHRINK = 30.0   # AR(1) shrinkage pseudo-count

TEMP_VARS = {"temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "temp_max_c", "temp_min_c"}


# ---------------------------------------------------------------- data
def load(data_dir, year_x):
    rows = defaultdict(dict)  # loc -> date -> record
    meta = {}
    path = os.path.join(data_dir, "daily_history.csv")
    if not os.path.exists(path):
        path = os.path.join(data_dir, "daily_year_x.csv")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            if d > f"{year_x}-12-31":
                continue  # causality guard
            def num(k):
                v = r.get(k, "")
                try:
                    v = float(v)
                    return v if math.isfinite(v) else None
                except (TypeError, ValueError):
                    return None
            rows[r["location_id"]][d] = (num("temperature_2m_mean"), num("temperature_2m_max"),
                                          num("temperature_2m_min"), num("precipitation_sum"),
                                          num("wind_speed_10m_max"))
            meta[r["location_id"]] = (float(r["lat"]), float(r["lon"]))
    return rows, meta


def window_days(year, ms, ds, me, de):
    import datetime as dt
    a = dt.date(year, ms, ds)
    b = dt.date(year, me, min(de, calendar.monthrange(year, me)[1]))
    out = []
    while a <= b:
        out.append(a.isoformat())
        a += dt.timedelta(days=1)
    return out


def aggregate(recs, var):
    """recs: list of record tuples (None for missing days)."""
    idx = {"temp_mean_c": 0, "temp_mean_max_c": 1, "temp_mean_min_c": 2, "temp_max_c": 1,
           "temp_min_c": 2, "precip_sum_mm": 3, "precip_max_day_mm": 3, "wind_max_kmh": 4,
           "hot_days_ge_30c": 1, "frost_days_lt_0c": 2, "rainy_days_ge_1mm": 3}[var]
    vals = [r[idx] for r in recs if r is not None and r[idx] is not None]
    n_exp = len(recs)
    if n_exp == 0 or len(vals) < 0.9 * n_exp:
        return None
    if var in ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c"):
        return sum(vals) / len(vals)
    if var in ("temp_max_c", "precip_max_day_mm", "wind_max_kmh"):
        return max(vals)
    if var == "temp_min_c":
        return min(vals)
    scale = n_exp / len(vals)
    if var == "precip_sum_mm":
        return sum(vals) * scale
    if var == "hot_days_ge_30c":
        return sum(v >= 30.0 for v in vals) * scale
    if var == "frost_days_lt_0c":
        return sum(v < 0.0 for v in vals) * scale
    if var == "rainy_days_ge_1mm":
        return sum(v >= 1.0 for v in vals) * scale
    raise ValueError(var)


def yearly_series(loc_rows, years, win, var):
    ys, vs = [], []
    for y in years:
        days = window_days(y, *win)
        recs = [loc_rows.get(d) for d in days]
        v = aggregate(recs, var)
        if v is not None:
            ys.append(y)
            vs.append(v)
    return ys, vs


# ---------------------------------------------------------------- model
def theil_sen(t, y):
    s = sorted((y[j] - y[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)))
    return s[len(s) // 2] if s else 0.0


def median(v):
    s = sorted(v)
    n = len(s)
    return 0.5 * (s[(n - 1) // 2] + s[n // 2])


def spectral_cycle(t, r, t_next):
    """Strongest harmonic (period 2..8 y) of residuals r; Fisher g-test gated."""
    n = len(r)
    if n < 10:
        return 0.0, None
    var = sum(x * x for x in r)
    if var <= 0:
        return 0.0, None
    best = None
    nf = 0
    for k in range(1, n // 2 + 1):
        period = n / k
        if not 2.0 <= period <= 8.0:
            continue
        nf += 1
        w = 2 * math.pi * k / n
        c = sum(x * math.cos(w * ti) for x, ti in zip(r, t))
        s = sum(x * math.sin(w * ti) for x, ti in zip(r, t))
        p = (c * c + s * s) * 2 / n
        if best is None or p > best[0]:
            best = (p, w, c, s)
    if best is None or nf == 0:
        return 0.0, None
    p, w, c, s = best
    g = p / var  # fraction of variance in peak (approx.)
    # Fisher test p-value approximation: nf * (1-g)^(nf-1)
    pval = min(1.0, nf * (1 - min(g, 0.999)) ** (nf - 1))
    if pval > CYCLE_P:
        return 0.0, None
    a, b = 2 * c / n, 2 * s / n
    shrink = CYCLE_SHRINK
    return shrink * (a * math.cos(w * t_next) + b * math.sin(w * t_next)), 2 * math.pi / w


def forecast(years, vals, var, year_next):
    n = len(vals)
    t = [y - years[-1] for y in years]  # t=0 is year X
    tn = year_next - years[-1]
    b = theil_sen(t, vals)
    prior = 0.03 if var in TEMP_VARS else 0.0
    wb = n / (n + 15.0)
    b = wb * b + (1 - wb) * prior
    a = median([v - b * ti for v, ti in zip(vals, t)])
    r = [v - (a + b * ti) for v, ti in zip(vals, t)]
    mr = sum(r) / n
    rc = [x - mr for x in r]
    den = sum(x * x for x in rc)
    phi = sum(rc[i] * rc[i - 1] for i in range(1, n)) / den if den > 0 else 0.0
    phi = max(-0.4, min(0.5, phi * n / (n + PHI_SHRINK)))
    cyc, period = spectral_cycle(t, rc, tn)
    mu = a + b * tn + mr + phi * rc[-1] + cyc
    sd = math.sqrt(den / max(n - 2, 1)) if den > 0 else 0.0
    sigma = sd * math.sqrt(1 + 1.0 / n + (tn - sum(t) / n) ** 2 / max(sum((ti - sum(t) / n) ** 2 for ti in t), 1e-9)) * INFLATE
    return {"mu": mu, "sigma": sigma, "trend": b, "phi": phi, "period": period, "cyc": cyc,
            "last": vals[-1], "clim": sum(vals) / n, "n": n, "resid": rc}


# ---------------------------------------------------------------- claims
WINDOWS = {
    "year": (1, 1, 12, 31),
    "jja": (6, 1, 8, 31),
    "djf_jf": (1, 1, 2, 31),
    "mam": (3, 1, 5, 31),
    "son": (9, 1, 11, 30),
    "jan": (1, 1, 1, 31),
    "jul": (7, 1, 7, 31),
}

COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
NONNEG = COUNT_VARS | {"precip_sum_mm", "precip_max_day_mm", "wind_max_kmh"}


def rnd(v, var):
    return round(v) if var in COUNT_VARS else round(v, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X = args.year_x
    Y1 = X + 1
    rows, meta = load(args.data_dir, X)
    cands = []  # (priority, claim)

    def target(loc, win, var):
        lat, lon = meta[loc]
        ms, ds, me, de = WINDOWS[win]
        de = min(de, calendar.monthrange(Y1, me)[1])
        return {"location_id": loc, "lat": lat, "lon": lon,
                "window_start": f"{Y1}-{ms:02d}-{ds:02d}", "window_end": f"{Y1}-{me:02d}-{de:02d}",
                "variable": var}

    for loc in sorted(rows):
        lr = rows[loc]
        years = sorted({int(d[:4]) for d in lr})
        years = [y for y in years if y <= X]

        def fc(win, var):
            ys, vs = yearly_series(lr, years, WINDOWS[win], var)
            if len(vs) < 8 or ys[-1] != X:
                return None
            return forecast(ys, vs, var, Y1)

        # --- interval claims on continuous targets
        specs = [("year", "temp_mean_c", 10), ("jja", "temp_mean_c", 9), ("year", "precip_sum_mm", 8),
                 ("djf_jf", "temp_mean_c", 6), ("jul", "temp_mean_c", 6), ("jan", "temp_mean_c", 6),
                 ("year", "temp_mean_max_c", 4), ("year", "temp_mean_min_c", 4),
                 ("year", "rainy_days_ge_1mm", 3), ("year", "hot_days_ge_30c", 5),
                 ("year", "frost_days_lt_0c", 5), ("year", "temp_max_c", 4)]
        for win, var, pri in specs:
            f = fc(win, var)
            if f is None or f["sigma"] <= 0:
                continue
            if var in COUNT_VARS and f["clim"] < 3:
                continue  # rare-event counts are not Gaussian; skip
            z = ND.inv_cdf(0.5 + COVER / 2)
            lo, hi = f["mu"] - z * f["sigma"], f["mu"] + z * f["sigma"]
            if var in NONNEG:
                lo = max(0.0, lo)
            lo, hi = rnd(lo, var), rnd(hi, var)
            if var in COUNT_VARS:
                lo, hi = math.floor(f["mu"] - z * f["sigma"]), math.ceil(f["mu"] + z * f["sigma"])
                lo = max(0, lo)
            anom = (f["mu"] - f["last"]) / f["sigma"]
            cyc = f" cycle P={f['period']:.1f}y adds {f['cyc']:+.2f}." if f["period"] else ""
            cands.append((pri, {
                "target": target(loc, win, var),
                "assertion": {"kind": "interval", "low": lo, "high": hi},
                "confidence": COVER,
                "rationale": (f"Damped AR(1)+trend: X={f['last']:.2f}, clim={f['clim']:.2f}, "
                              f"trend={f['trend']:+.3f}/y, phi={f['phi']:.2f}, fcst={f['mu']:.2f}"
                              f"±{f['sigma']:.2f}.{cyc} Forecast minus year X = {anom:+.1f} sd"
                              f"{' (mean reversion of extreme year X)' if abs(anom) >= 1 else ''}."),
            }))

        # --- warmer/wetter than year X: mean-reversion bets
        for win, var, bvar, pri in [("year", "temp_mean_c", "warmer_than_year_x", 9),
                                    ("jja", "temp_mean_c", "warmer_than_year_x", 7),
                                    ("djf_jf", "temp_mean_c", "warmer_than_year_x", 6),
                                    ("year", "precip_sum_mm", "wetter_than_year_x", 8),
                                    ("jja", "precip_sum_mm", "wetter_than_year_x", 5)]:
            f = fc(win, var)
            if f is None or f["sigma"] <= 0 or f["clim"] < 5 and var == "precip_sum_mm":
                continue
            p = 1 - ND.cdf((f["last"] - f["mu"]) / f["sigma"])
            if 0.30 < p < 0.70:
                continue
            val = p >= 0.5
            conf = max(p, 1 - p)
            cands.append((pri + 2 * (conf - 0.7), {
                "target": target(loc, win, bvar),
                "assertion": {"kind": "boolean", "value": val},
                "confidence": round(min(conf, 0.85), 3),
                "rationale": (f"Year X {var}={f['last']:.2f} vs damped-cycle forecast {f['mu']:.2f}±{f['sigma']:.2f}: "
                              f"P(X+1 > X)={p:.2f}; extreme year X swings back."),
            }))

        # --- hottest / wettest month (climatological mode, full year)
        for cvar, base, pri in [("hottest_month", "temp_mean_c", 7), ("wettest_month", "precip_sum_mm", 3)]:
            counts = defaultdict(int)
            ny = 0
            for y in years:
                best = None
                ok = True
                for m in range(1, 13):
                    v = aggregate([lr.get(d) for d in window_days(y, m, 1, m, 31)], base)
                    if v is None:
                        ok = False
                        break
                    if best is None or v > best[0]:
                        best = (v, m)
                if ok:
                    counts[best[1]] += 1
                    ny += 1
            if ny < 8:
                continue
            m, c = max(counts.items(), key=lambda kv: kv[1])
            p = (c + 0.5) / (ny + 1.5)
            if p < 0.80:
                continue
            cands.append((pri + 2 * (p - 0.7), {
                "target": target(loc, "year", cvar),
                "assertion": {"kind": "categorical", "value": f"{m:02d}"},
                "confidence": round(min(p, 0.88), 3),
                "rationale": f"Seasonal phase locked: month {m:02d} was the {cvar.split('_')[0]} month in {c}/{ny} trailing years.",
            }))

    cands.sort(key=lambda pc: -pc[0])
    out, seen = [], set()
    for _, c in cands:
        t = c["target"]
        key = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if key in seen:
            continue
        seen.add(key)
        c = {"claim_id": f"p6_{len(out) + 1:03d}", **c}
        out.append(c)
        if len(out) >= MAX_CLAIMS:
            break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"predict6: wrote {len(out)} claims for X+1={Y1} -> {args.out}")


if __name__ == "__main__":
    main()
