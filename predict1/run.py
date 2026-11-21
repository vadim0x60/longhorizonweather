#!/usr/bin/env python3
"""predict1 — "The Neural Sequence Forecaster".

Treats each (location, calendar window, variable) as a yearly sequence and
forecasts its year-X+1 value as a predictive distribution:

  1. Build monthly aggregates from `daily_history.csv` (years <= X) plus the
     bundled ERA5 monthly archive for 1964..1979 (`data/era5_monthly_pre1980.csv`,
     strictly older than any allowed X >= 1980).
  2. Fit, at runtime and only on years <= X, a convex ensemble of sequence
     forecasters (all-years mean, last-10 mean, OLS trend) by walk-forward
     backtesting, and pool its standardized residuals into empirical quantiles.
  3. Optionally blend in a Tinker-hosted LoRA sequence model (see tinker_model.py,
     trained by tinker_train.py on GHCN station data with target years <= 1969)
     that reads anonymized anomaly sequences only (no years, no place names).
  4. Emit up to 150 claims: thresholds placed at the naive 15-year climatology
     whenever our distribution disagrees with it, booleans vs year X, and modal
     hottest/wettest months.

Never reads anything dated after X.
"""
import argparse
import bisect
import calendar
import csv
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PRE1980 = os.path.join(HERE, "data", "era5_monthly_pre1980.csv")
MAX_CLAIMS = 150

AGG_FIELDS = ["n_t", "st", "stx", "stn", "txmax", "tnmin", "n_p", "psum", "pmax",
              "n_w", "wmax", "hot", "frost", "rainy"]

NUM_VARS = ["temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "temp_max_c",
            "temp_min_c", "precip_sum_mm", "precip_max_day_mm", "wind_max_kmh",
            "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"]
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
VAR_GROUP = {
    "temp_mean_c": "temp", "temp_mean_max_c": "temp", "temp_mean_min_c": "temp",
    "temp_max_c": "temp_ext", "temp_min_c": "temp_ext",
    "precip_sum_mm": "precip", "precip_max_day_mm": "precip_ext", "rainy_days_ge_1mm": "precip",
    "wind_max_kmh": "wind", "hot_days_ge_30c": "count", "frost_days_lt_0c": "count",
}
# How likely other predictors are to target this variable / window (collision prior).
VAR_W = {"temp_mean_c": 1.0, "precip_sum_mm": 0.9, "temp_max_c": 0.6, "temp_min_c": 0.6,
         "temp_mean_max_c": 0.55, "temp_mean_min_c": 0.55, "hot_days_ge_30c": 0.6,
         "frost_days_lt_0c": 0.6, "rainy_days_ge_1mm": 0.6, "precip_max_day_mm": 0.45,
         "wind_max_kmh": 0.45, "warmer_than_year_x": 0.7, "wetter_than_year_x": 0.6,
         "hottest_month": 0.7, "wettest_month": 0.7}


def windows():
    w = [("year", 1, 12, 1.0), ("mam", 3, 5, 0.55), ("jja", 6, 8, 0.8), ("son", 9, 11, 0.55),
         ("jf", 1, 2, 0.3)]
    for m in range(1, 13):
        w.append((f"m{m:02d}", m, m, 0.5 if m in (1, 7) else 0.35))
    return w


def fnum(s):
    if s is None or s == "" or s.lower() in ("nan", "null", "none"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def new_agg():
    return {"n_t": 0, "st": 0.0, "stx": 0.0, "stn": 0.0, "txmax": -1e9, "tnmin": 1e9,
            "n_p": 0, "psum": 0.0, "pmax": -1e9, "n_w": 0, "wmax": -1e9,
            "hot": 0, "frost": 0, "rainy": 0}


def aggregate_daily(path, max_year, aggs, locs, year_x_daily=None, year_x=None):
    """Fold a daily CSV into monthly aggregates keyed (loc, year, month)."""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            y, m = int(d[:4]), int(d[5:7])
            if y > max_year:
                continue  # hard causality guard
            loc = r["location_id"]
            if loc not in locs:
                locs[loc] = (float(r["lat"]), float(r["lon"]))
            a = aggs.setdefault((loc, y, m), new_agg())
            tm, tx, tn = fnum(r.get("temperature_2m_mean")), fnum(r.get("temperature_2m_max")), fnum(r.get("temperature_2m_min"))
            p, w = fnum(r.get("precipitation_sum")), fnum(r.get("wind_speed_10m_max"))
            if tm is not None and tx is not None and tn is not None:
                a["n_t"] += 1; a["st"] += tm; a["stx"] += tx; a["stn"] += tn
                a["txmax"] = max(a["txmax"], tx); a["tnmin"] = min(a["tnmin"], tn)
                a["hot"] += tx >= 30.0; a["frost"] += tn < 0.0
            if p is not None:
                a["n_p"] += 1; a["psum"] += p; a["pmax"] = max(a["pmax"], p); a["rainy"] += p >= 1.0
            if w is not None:
                a["n_w"] += 1; a["wmax"] = max(a["wmax"], w)
            if year_x_daily is not None and y == year_x:
                year_x_daily[(loc, d[5:])] = (tm, p)


def load_pre1980(aggs, locs):
    if not os.path.exists(PRE1980):
        return
    with open(PRE1980, newline="") as f:
        for r in csv.DictReader(f):
            y = int(r["year"])
            if y >= 1980:
                continue
            loc = r["location_id"]
            locs.setdefault(loc, (float(r["lat"]), float(r["lon"])))
            a = {k: float(r[k]) for k in AGG_FIELDS}
            aggs.setdefault((loc, y, int(r["month"])), a)


def window_value(var, months_aggs):
    """Value of `var` over a window made of full months, or None if incomplete."""
    nt = sum(a["n_t"] for _, a, _ in months_aggs)
    np_ = sum(a["n_p"] for _, a, _ in months_aggs)
    ndays = sum(nd for _, _, nd in months_aggs)
    tok = nt >= 0.9 * ndays
    pok = np_ >= 0.9 * ndays
    if var == "temp_mean_c":
        return sum(a["st"] for _, a, _ in months_aggs) / nt if tok else None
    if var == "temp_mean_max_c":
        return sum(a["stx"] for _, a, _ in months_aggs) / nt if tok else None
    if var == "temp_mean_min_c":
        return sum(a["stn"] for _, a, _ in months_aggs) / nt if tok else None
    if var == "temp_max_c":
        return max(a["txmax"] for _, a, _ in months_aggs) if tok else None
    if var == "temp_min_c":
        return min(a["tnmin"] for _, a, _ in months_aggs) if tok else None
    if var == "hot_days_ge_30c":
        return sum(a["hot"] for _, a, _ in months_aggs) * ndays / nt if tok else None
    if var == "frost_days_lt_0c":
        return sum(a["frost"] for _, a, _ in months_aggs) * ndays / nt if tok else None
    if var == "precip_sum_mm":
        return sum(a["psum"] for _, a, _ in months_aggs) * ndays / np_ if pok else None
    if var == "precip_max_day_mm":
        return max(a["pmax"] for _, a, _ in months_aggs) if pok else None
    if var == "rainy_days_ge_1mm":
        return sum(a["rainy"] for _, a, _ in months_aggs) * ndays / np_ if pok else None
    if var == "wind_max_kmh":
        nw = sum(a["n_w"] for _, a, _ in months_aggs)
        return max(a["wmax"] for _, a, _ in months_aggs) if nw >= 0.9 * ndays else None
    raise KeyError(var)


def build_series(aggs, loc, m1, m2, years):
    """{var: {year: value}} plus monthly temp/precip per year for argmax targets."""
    out = defaultdict(dict)
    monthly = {}
    for y in years:
        ma = []
        for m in range(m1, m2 + 1):
            a = aggs.get((loc, y, m))
            if a is None:
                break
            ma.append((m, a, calendar.monthrange(y, m)[1]))
        else:
            for v in NUM_VARS:
                val = window_value(v, ma)
                if val is not None:
                    out[v][y] = val
            mt = {m: (a["st"] / a["n_t"] if a["n_t"] >= 0.9 * nd else None) for m, a, nd in ma}
            mp = {m: (a["psum"] * nd / a["n_p"] if a["n_p"] >= 0.9 * nd else None) for m, a, nd in ma}
            monthly[y] = (mt, mp)
    return out, monthly


# ---------------------------------------------------------------- forecasters
def ols_predict(ys, vs, t):
    n = len(ys)
    mx, my = sum(ys) / n, sum(vs) / n
    sxx = sum((y - mx) ** 2 for y in ys)
    if sxx <= 0:
        return my
    b = sum((y - mx) * (v - my) for y, v in zip(ys, vs)) / sxx
    return my + b * (t - mx)


def base_forecasts(series, t):
    """Component forecasts for year t from strictly earlier years of `series`."""
    ys = sorted(y for y in series if y < t)
    if len(ys) < 1:
        return None
    vs = [series[y] for y in ys]
    clim = sum(vs) / len(vs)
    last = vs[-10:]
    last10 = sum(last) / len(last)
    trend = ols_predict(ys, vs, t) if len(ys) >= 8 else last10
    return (clim, last10, trend)


def scale_of(series):
    vs = list(series.values())
    if len(vs) < 2:
        return None
    mu = sum(vs) / len(vs)
    return math.sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1))


WEIGHT_GRID = [(a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)]


def fit_ensemble(all_series, year_x):
    """Walk-forward backtest on years <= X: per variable group choose convex
    weights over (clim, last10, trend) and collect standardized residuals."""
    rows = defaultdict(list)  # group -> [(comps, target, scale, wkind)]
    for (loc, wname, var), series in all_series.items():
        s = scale_of(series)
        if not s or s < 1e-6:
            continue
        for t in sorted(series):
            if t > year_x:
                continue
            if sum(1 for y in series if y < t) < 8:
                continue
            comps = base_forecasts(series, t)
            rows[VAR_GROUP[var]].append((comps, series[t], s, var, wname))
    model = {}
    for g, rs in rows.items():
        best, bw = None, (1.0, 0.0, 0.0)
        for w in WEIGHT_GRID:
            err = 0.0
            for comps, tgt, s, _, _ in rs:
                f = w[0] * comps[0] + w[1] * comps[1] + w[2] * comps[2]
                err += ((tgt - f) / s) ** 2
            if best is None or err < best:
                best, bw = err, w
        model[g] = {"w": bw, "n": len(rs)}
    resid = defaultdict(list)  # (var, wkind) -> standardized residuals
    for g, rs in rows.items():
        w = model[g]["w"]
        for comps, tgt, s, var, wname in rs:
            f = w[0] * comps[0] + w[1] * comps[1] + w[2] * comps[2]
            e = (tgt - f) / s
            wk = wname if wname in ("year",) else ("season" if wname in ("mam", "jja", "son", "jf") else "month")
            resid[(var, wk)].append(e)
            resid[(var, "*")].append(e)
    for k in resid:
        resid[k].sort()
    return model, resid


class Predictive:
    """Empirical location-scale distribution m + s * e."""

    def __init__(self, m, s, eps, lo=None):
        self.m, self.s, self.eps, self.lo = m, s, eps, lo

    def cdf(self, v):
        z = (v - self.m) / self.s
        k = bisect.bisect_right(self.eps, z)
        return (k + 0.5) / (len(self.eps) + 1)  # smoothed, never 0 or 1

    def q(self, p):
        e = self.eps
        i = min(len(e) - 1, max(0, int(round(p * (len(e) - 1)))))
        v = self.m + self.s * e[i]
        return max(v, self.lo) if self.lo is not None else v


# --------------------------------------------------------------------- claims
def month_window(y, m1, m2):
    return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{calendar.monthrange(y, m2)[1]:02d}"


def fmt(v):
    return round(float(v), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tinker", action="store_true", help="skip the Tinker LoRA component")
    args = ap.parse_args()
    X, Y = args.year_x, args.year_x + 1

    aggs, locs, xdaily = {}, {}, {}
    load_pre1980(aggs, locs)
    hist = os.path.join(args.data_dir, "daily_history.csv")
    yx = os.path.join(args.data_dir, "daily_year_x.csv")
    data_locs = {}
    for p in (hist, yx):
        if os.path.exists(p):
            aggregate_daily(p, X, aggs, data_locs, xdaily, X)
    if not data_locs:
        sys.exit("no daily data found in --data-dir")
    years = sorted({y for (_, y, _) in aggs if y <= X})
    recent = [y for y in years if y >= max(1980, X - 14)]  # what a naive 15-y climatology sees

    all_series, monthly_by = {}, {}
    for loc in data_locs:
        for wname, m1, m2, _ in windows():
            ser, monthly = build_series(aggs, loc, m1, m2, years)
            for v, s in ser.items():
                all_series[(loc, wname, v)] = s
            monthly_by[(loc, wname)] = monthly
    model, resid = fit_ensemble(all_series, X)

    tinker_pred, blend = {}, 0.0
    if not args.no_tinker and os.environ.get("TINKER_API_KEY"):
        try:
            import tinker_model  # noqa: local module
            blend = json.load(open(tinker_model.META)).get("blend_weight", 0.0)
            if blend > 0:
                tinker_pred = tinker_model.predictions(all_series, X, data_locs)
        except Exception as e:  # never let the remote component break a run
            print(f"[predict1] tinker component skipped: {e!r}", file=sys.stderr)

    def blended(loc, wname, var, m):
        p = tinker_pred.get((loc, wname, var))
        return m if p is None else (1 - blend) * m + blend * p

    cands = []
    for loc, (lat, lon) in sorted(data_locs.items()):
        for wname, m1, m2, ww in windows():
            ws, we = month_window(Y, m1, m2)
            base_t = {"location_id": loc, "lat": lat, "lon": lon, "window_start": ws, "window_end": we}
            for var in NUM_VARS:
                series = all_series.get((loc, wname, var))
                if not series or X not in series or len(series) < 3:
                    continue
                comps = base_forecasts(series, Y)
                w = model.get(VAR_GROUP[var], {"w": (1, 0, 0)})["w"]
                m = sum(wi * ci for wi, ci in zip(w, comps))
                s = scale_of(series)
                wk = "year" if wname == "year" else ("season" if wname in ("mam", "jja", "son", "jf") else "month")
                eps = resid.get((var, wk)) or resid.get((var, "*"))
                lo = 0.0 if (var in COUNT_VARS or var.startswith("precip")) else None
                if s is None or s < 1e-6 or not eps or len(eps) < 30:
                    vals = set(series.values())
                    if var in COUNT_VARS and vals == {0.0} and len(series) >= 10:
                        cands.append((0.15 * ww * VAR_W[var], dict(base_t, variable=var),
                                      {"kind": "threshold", "op": "lt", "value": 0.5}, 0.95,
                                      f"zero in all {len(series)} years on record"))
                    continue
                m = blended(loc, wname, var, m)  # LoRA sequence-model blend
                dist = Predictive(m, s, eps, lo)
                rv = [series[y] for y in recent if y in series]
                naive = sum(rv) / len(rv)
                p_up = 1 - dist.cdf(naive)
                why = (f"ensemble w={w} mean {m:.2f}, sigma {s:.2f}; naive {len(rv)}-yr clim {naive:.2f}")
                if p_up >= 0.6:
                    a, p = {"kind": "threshold", "op": "gt", "value": fmt(naive)}, p_up
                elif p_up <= 0.4:
                    a, p = {"kind": "threshold", "op": "lt", "value": fmt(naive)}, 1 - p_up
                else:
                    lo_q, hi_q = dist.q(0.12), dist.q(0.88)
                    if hi_q - lo_q < 1e-6:
                        continue
                    a = {"kind": "interval", "low": fmt(lo_q), "high": fmt(hi_q)}
                    p = dist.cdf(hi_q) - dist.cdf(lo_q)
                    cands.append((0.35 * ww * VAR_W[var] * (2 * p - 1), dict(base_t, variable=var), a, p,
                                  "central interval; " + why))
                    continue
                cands.append((ww * VAR_W[var] * (2 * p - 1) * 1.6, dict(base_t, variable=var), a, p, why))

            # booleans vs. year X (same window), using the known year-X value
            for var, bvar in (("temp_mean_c", "warmer_than_year_x"), ("precip_sum_mm", "wetter_than_year_x")):
                series = all_series.get((loc, wname, var))
                if not series or X not in series or len(series) < 3:
                    continue
                comps = base_forecasts(series, Y)
                w = model.get(VAR_GROUP[var], {"w": (1, 0, 0)})["w"]
                s = scale_of(series)
                wk = "year" if wname == "year" else ("season" if wname in ("mam", "jja", "son", "jf") else "month")
                eps = resid.get((var, wk)) or resid.get((var, "*"))
                if not s or s < 1e-6 or not eps or len(eps) < 30:
                    continue
                m = sum(wi * ci for wi, ci in zip(w, comps))
                m = blended(loc, wname, var, m)
                p_true = 1 - Predictive(m, s, eps).cdf(series[X])
                val, p = (True, p_true) if p_true >= 0.5 else (False, 1 - p_true)
                if p < 0.6:
                    continue
                cands.append((ww * VAR_W[bvar] * (2 * p - 1) * 1.6, dict(base_t, variable=bvar),
                              {"kind": "boolean", "value": val}, p,
                              f"year-X value {series[X]:.2f} vs forecast {m:.2f} (sigma {s:.2f})"))

            # modal hottest / wettest month (multi-month windows only)
            if m2 > m1:
                monthly = monthly_by.get((loc, wname), {})
                for idx, cvar in ((0, "hottest_month"), (1, "wettest_month")):
                    counts, tot = defaultdict(float), 0.0
                    for y, mm in monthly.items():
                        vals = {k: v for k, v in mm[idx].items() if v is not None}
                        if len(vals) != m2 - m1 + 1:
                            continue
                        wy = 1.0 + 1.0 * (y >= X - 14)  # weight recent decades more
                        counts[max(vals, key=vals.get)] += wy
                        tot += wy
                    if tot < 5:
                        continue
                    nmon = m2 - m1 + 1
                    best = max(counts, key=counts.get)
                    p = (counts[best] + 0.5) / (tot + 0.5 * nmon)
                    if p < 0.45:
                        continue
                    cands.append((ww * VAR_W[cvar] * (2 * p - 1) * 1.6, dict(base_t, variable=cvar),
                                  {"kind": "categorical", "value": f"{best:02d}"}, p,
                                  f"modal {cvar.split('_')[0]} month in {int(round(tot))} weighted yrs"))

    cands.sort(key=lambda c: -c[0])
    seen, claims = set(), []
    for _, tgt, a, p, why in cands:
        key = (tgt["location_id"], tgt["window_start"], tgt["window_end"], tgt["variable"])
        if key in seen:
            continue
        seen.add(key)
        claims.append({"claim_id": f"p1_{len(claims) + 1:03d}", "target": tgt, "assertion": a,
                       "confidence": round(min(0.99, max(0.5, p)), 3), "rationale": why[:300]})
        if len(claims) >= MAX_CLAIMS:
            break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(claims, f, indent=1)
    print(f"[predict1] X={X}: {len(claims)} claims from {len(cands)} candidates; "
          f"ensemble={ {g: v['w'] for g, v in model.items()} }; tinker_pred={len(tinker_pred)} blend={blend}", file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
