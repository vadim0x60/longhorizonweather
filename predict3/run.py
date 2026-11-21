#!/usr/bin/env python3
"""predict3: "Synoptic Data Engineer & Teleconnection Realist".

Offline, stdlib-only statistical forecaster for year X+1:

  * For every (location, window, variable) it builds the yearly series of the
    window statistic from daily_history.csv (max(1980, X-15)..X).
  * Mean model: shrunk linear trend (warming prior for temperatures, none for
    precipitation/wind) + pooled persistence of the location's detrended
    Sep-Dec temperature anomaly of year X + (temperature only) a pooled ENSO
    teleconnection term on Oct-Dec Nino3.4 of year X.
  * Skewed, non-negative variables (precipitation, day counts) are modelled in
    sqrt space.
  * Predictive spread = residual spread with parameter-uncertainty inflation.
  * Claims: central intervals (~65% coverage) for continuous targets, booleans
    for warmer/wetter-than-year-X when the model is confident, categorical
    hottest/wettest month when the empirical frequency is dominant.

Calibration (offline backtests over X=2006..2023, 150 claims/year, scored on X+1):
mean stated confidence 0.70, realised hit rate 0.70 (0.72 for 2006-14, 0.68 for 2015-23).
Ablations: persistence clearly helps; per-location ENSO fits hurt; pooled ENSO is
neutral-to-slightly-positive for temperature.

Usage: python3 predict3/run.py --year-x <X> --data-dir <data_dir> --out <claims_json>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
from collections import defaultdict

ND = statistics.NormalDist()

MAX_CLAIMS = 150
MAX_PER_LOCATION = 9
COVERAGE = 0.65          # nominal central-interval coverage
SIGMA_INFLATE = 1.00     # calibrated so out-of-sample z-scores have ~unit spread (backtests X=2006..2023)
PHI_MAX = 0.7            # cap on pooled persistence coefficient
USE_ENSO = True          # pooled ENSO term, temperature variables only (backtest: small gain 2015-2023, neutral 2006-2014)
ENSO_VARS = {"temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c"}
OTHER_TREND_SHRINK = 0.5  # extra shrink of EB-weighted trends for non-temperature variables
BOOL_MIN_CONF = 0.64
CAT_MIN_CONF = 0.65

# window key -> (start MM-DD, end MM-DD, popularity weight)
WINDOWS = {
    "ANN": ("01-01", "12-31", 1.0),
    "JJA": ("06-01", "08-31", 0.8),
    "MAM": ("03-01", "05-31", 0.5),
    "SON": ("09-01", "11-30", 0.5),
    "JF": ("01-01", "02-29", 0.45),
    "JFM": ("01-01", "03-31", 0.4),
    "JAN": ("01-01", "01-31", 0.4),
    "JUL": ("07-01", "07-31", 0.4),
}

# variable -> (daily column, aggregation, transform, popularity weight, rounding step)
CONT_VARS = {
    "temp_mean_c": ("temperature_2m_mean", "mean", "id", 1.0, 0.05),
    "precip_sum_mm": ("precipitation_sum", "sum", "sqrt", 0.85, 1.0),
    "temp_max_c": ("temperature_2m_max", "max", "id", 0.5, 0.1),
    "temp_min_c": ("temperature_2m_min", "min", "id", 0.5, 0.1),
    "temp_mean_max_c": ("temperature_2m_max", "mean", "id", 0.35, 0.05),
    "temp_mean_min_c": ("temperature_2m_min", "mean", "id", 0.35, 0.05),
    "precip_max_day_mm": ("precipitation_sum", "max", "sqrt", 0.3, 0.5),
    "wind_max_kmh": ("wind_speed_10m_max", "max", "id", 0.3, 0.5),
    "hot_days_ge_30c": ("temperature_2m_max", "count_ge30", "sqrt", 0.6, 1),
    "frost_days_lt_0c": ("temperature_2m_min", "count_lt0", "sqrt", 0.6, 1),
    "rainy_days_ge_1mm": ("precipitation_sum", "count_ge1", "id", 0.4, 1),
}
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
TEMP_VARS = {"temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c", "hot_days_ge_30c", "frost_days_lt_0c"}
# Which (window, variable) pairs to consider; window popularity * variable popularity ranks them.
PAIRS = (
    [("ANN", v) for v in CONT_VARS]
    + [(w, "temp_mean_c") for w in ("JJA", "MAM", "SON", "JF", "JFM", "JAN", "JUL")]
    + [(w, "precip_sum_mm") for w in ("JJA", "MAM", "SON", "JF")]
    + [("JJA", "temp_max_c"), ("JJA", "hot_days_ge_30c"), ("JF", "frost_days_lt_0c"), ("JF", "temp_min_c")]
)
# Trend priors (per year) in transformed units: warming for temperature, none otherwise.
TREND_PRIOR = {"temp": (0.03, 0.015), "other": (0.0, None)}


def transform(kind, x):
    return math.sqrt(max(x, 0.0)) if kind == "sqrt" else x


def inverse(kind, z):
    return max(z, 0.0) ** 2 if kind == "sqrt" else z


# ---------------------------------------------------------------- data


def load(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    X = manifest["year_x"]
    cutoff = f"{X}-12-31"
    daily = defaultdict(dict)  # loc -> date -> row
    meta = {}
    with open(os.path.join(data_dir, "daily_history.csv")) as f:
        for r in csv.DictReader(f):
            if r["date"] > cutoff:
                raise AssertionError("data beyond year X in data dir")
            loc = r["location_id"]
            meta[loc] = (float(r["lat"]), float(r["lon"]))
            daily[loc][r["date"]] = {k: (float(r[k]) if r[k] != "" else None) for k in r if k not in ("location_id", "lat", "lon", "date")}
    nino = {}
    p = os.path.join(data_dir, "indices_monthly.json")
    if os.path.exists(p):
        with open(p) as f:
            series = json.load(f).get("series", {})
        for row in series.get("nino34_anom", []):
            if row["date"] <= f"{X}-12":
                nino[row["date"]] = row["value"]
    return X, daily, meta, nino


def window_dates(year, start_md, end_md):
    s = dt.date.fromisoformat(f"{year}-{start_md}")
    em, ed = map(int, end_md.split("-"))
    if em == 2 and ed == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        ed = 28
    return s, dt.date(year, em, ed)


def window_values(rows, year, start_md, end_md, col):
    s, e = window_dates(year, start_md, end_md)
    out = []
    d = s
    while d <= e:
        r = rows.get(d.isoformat())
        if r is None or r[col] is None:
            return None
        out.append(r[col])
        d += dt.timedelta(days=1)
    return out


def aggregate(vals, agg):
    if agg == "mean":
        return sum(vals) / len(vals)
    if agg == "sum":
        return sum(vals)
    if agg == "max":
        return max(vals)
    if agg == "min":
        return min(vals)
    if agg == "count_ge30":
        return float(sum(v >= 30.0 for v in vals))
    if agg == "count_lt0":
        return float(sum(v < 0.0 for v in vals))
    if agg == "count_ge1":
        return float(sum(v >= 1.0 for v in vals))
    raise ValueError(agg)


# ---------------------------------------------------------------- statistics


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / max(n - 2, 1)
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    return a, b, se, mx, sxx, res


def fit_trend(years, zs, is_temp):
    """Linear trend with the slope shrunk toward a physical prior."""
    a, b, se, my, syy, _ = ols(years, zs)
    mz = sum(zs) / len(zs)
    prior_mean, prior_sd = TREND_PRIOR["temp" if is_temp else "other"]
    if prior_sd is None:
        # No trend prior: shrink strongly toward zero (precip / wind trends are mostly noise in 16 years).
        w = 1.0 / (1.0 + (se / max(abs(b), 1e-9)) ** 2) if se > 0 else 1.0
        b_post = b * w * OTHER_TREND_SHRINK
    else:
        # Scale prior to the variable's units (temperature-like variables only).
        prec_d = 1.0 / se ** 2 if se > 0 else 1e12
        prec_p = 1.0 / prior_sd ** 2
        b_post = (b * prec_d + prior_mean * prec_p) / (prec_d + prec_p)
    a_post = mz - b_post * my
    res = [z - (a_post + b_post * y) for y, z in zip(years, zs)]
    return a_post, b_post, res, my, syy


# ---------------------------------------------------------------- model


def build(X, daily, nino):
    locs = sorted(daily)
    years_all = sorted({int(d[:4]) for loc in locs for d in daily[loc]})
    # Predictors known at the end of year y.
    nino_ond = {}
    for y in years_all:
        v = [nino.get(f"{y}-{m:02d}") for m in (10, 11, 12)]
        if all(x is not None for x in v):
            nino_ond[y] = sum(v) / 3
    sond = {}  # loc -> {year: Sep-Dec mean temp}
    for loc in locs:
        sond[loc] = {}
        for y in years_all:
            vals = window_values(daily[loc], y, "09-01", "12-31", "temperature_2m_mean")
            if vals:
                sond[loc][y] = sum(vals) / len(vals)
    sond_anom = {}
    for loc in locs:
        ys = sorted(sond[loc])
        _, _, res, _, _ = fit_trend(ys, [sond[loc][y] for y in ys], True)
        sond_anom[loc] = dict(zip(ys, res))

    models = {}
    for wkey, var in PAIRS:
        start_md, end_md, _ = WINDOWS[wkey]
        col, agg, tf, _, _ = CONT_VARS[var]
        per_loc = {}
        for loc in locs:
            ser = {}
            for y in years_all:
                vals = window_values(daily[loc], y, start_md, end_md, col)
                if vals:
                    ser[y] = aggregate(vals, agg)
            ys = sorted(ser)
            if len(ys) < 6 or X not in ser:
                continue
            raw = [ser[y] for y in ys]
            if max(raw) - min(raw) < 1e-9:
                continue  # degenerate (e.g. never any frost)
            zs = [transform(tf, v) for v in raw]
            a, b, res, my, syy = fit_trend(ys, zs, var in TEMP_VARS and tf == "id")
            per_loc[loc] = {"years": ys, "raw": ser, "z": dict(zip(ys, zs)), "a": a, "b": b,
                            "res": dict(zip(ys, res)), "my": my, "syy": syy}
        if not per_loc:
            continue
        # Pooled persistence of the Sep-Dec temperature anomaly (temperature-like variables only).
        phi = 0.0
        if var in TEMP_VARS and tf == "id":
            num = den = 0.0
            for loc, m in per_loc.items():
                for y in m["years"]:
                    p = sond_anom[loc].get(y - 1)
                    if p is not None:
                        num += p * m["res"][y]
                        den += p * p
            phi = min(max(num / den if den > 0 else 0.0, 0.0), PHI_MAX)
        # ENSO teleconnection per location, EB-shrunk across locations.
        est = {}
        for loc, m in per_loc.items():
            xs, rs = [], []
            for y in m["years"]:
                n_prev = nino_ond.get(y - 1)
                p = sond_anom[loc].get(y - 1, 0.0)
                if n_prev is not None and (y - 1) in m["res"]:
                    xs.append(n_prev)
                    rs.append(m["res"][y] - phi * p)
            if len(xs) >= 6 and max(xs) - min(xs) > 0.3:
                _, g, se, _, _, _ = ols(xs, rs)
                est[loc] = (g, se)
        # One inverse-variance pooled coefficient: per-location fits (<= 15 points) had no skill in backtests.
        gammas = {}
        if est and USE_ENSO and var in ENSO_VARS:
            w = {k: 1 / se ** 2 for k, (g, se) in est.items() if math.isfinite(se) and se > 0}
            gp = sum(est[k][0] * w[k] for k in w) / sum(w.values()) if w else 0.0
            gammas = {k: gp for k in est}
        for loc, m in per_loc.items():
            g = gammas.get(loc, 0.0)
            fitted_res = []
            for y in m["years"]:
                pred = phi * sond_anom[loc].get(y - 1, 0.0) + g * nino_ond.get(y - 1, 0.0) if (y - 1) in m["res"] else 0.0
                fitted_res.append(m["res"][y] - pred)
            n = len(fitted_res)
            k = 2  # per-location trend; phi and pooled gamma are shared across 20 locations
            s = math.sqrt(sum(r * r for r in fitted_res) / max(n - k, 2))
            t1 = X + 1
            lev = 1 + 1 / n + (t1 - m["my"]) ** 2 / m["syy"] if m["syy"] > 0 else 1 + 1 / n
            # Small-sample t inflation (approx) + misspecification factor.
            t_infl = 1 + 1.0 / max(n - k, 2)
            sigma = s * math.sqrt(lev) * t_infl * SIGMA_INFLATE
            mu = m["a"] + m["b"] * t1 + phi * sond_anom[loc].get(X, 0.0) + g * nino_ond.get(X, 0.0)
            models[(loc, wkey, var)] = {
                "mu": mu, "sigma": sigma, "tf": tf, "phi": phi, "gamma": g, "trend": m["b"],
                "z_x": m["z"][X], "raw": m["raw"], "nino_x": nino_ond.get(X),
            }
    return models


# ---------------------------------------------------------------- claims


def round_out(lo, hi, step, is_count):
    if is_count:
        return float(math.floor(lo)), float(math.ceil(hi))
    lo = math.floor(lo / step + 1e-9) * step
    hi = math.ceil(hi / step - 1e-9) * step
    nd = 0 if step >= 1 else 2
    return round(lo, nd), round(hi, nd)


def prob_interval(m, lo, hi):
    tf = m["tf"]
    dist = statistics.NormalDist(m["mu"], m["sigma"])
    # In sqrt space, mass below zero maps to a value of 0, which lies inside [0, hi].
    plo = 0.0 if tf == "sqrt" and lo <= 0 else dist.cdf(transform(tf, lo))
    return dist.cdf(transform(tf, hi)) - plo


def edge(p):
    """Ranking weight: expected margin over a disjoint rival (2p-1), saturating so that
    near-certain but trivial claims do not crowd out contested, popular targets."""
    return min(1.0, (2 * p - 1) / 0.3)


def month_frequency(daily, loc, years, col, agg):
    counts = defaultdict(int)
    for y in years:
        vals = []
        for mth in range(1, 13):
            last = (dt.date(y + (mth == 12), mth % 12 + 1, 1) - dt.timedelta(days=1)).day
            v = window_values(daily[loc], y, f"{mth:02d}-01", f"{mth:02d}-{last:02d}", col)
            if not v:
                vals = None
                break
            vals.append(aggregate(v, agg))
        if vals:
            counts[f"{vals.index(max(vals)) + 1:02d}"] += 1
    return counts


def make_claims(X, daily, meta, models):
    T = X + 1
    cands = []

    def target(loc, wkey, var):
        s, e = window_dates(T, *WINDOWS[wkey][:2])
        lat, lon = meta[loc]
        return {"location_id": loc, "lat": lat, "lon": lon, "window_start": s.isoformat(),
                "window_end": e.isoformat(), "variable": var}

    for (loc, wkey, var), m in models.items():
        col, agg, tf, vpop, step = CONT_VARS[var]
        is_count = var in COUNT_VARS
        z = ND.inv_cdf(0.5 + COVERAGE / 2)
        lo = inverse(tf, m["mu"] - z * m["sigma"])
        hi = inverse(tf, m["mu"] + z * m["sigma"])
        if tf == "sqrt" and m["mu"] - z * m["sigma"] <= 0:
            lo = 0.0
        lo, hi = round_out(lo, hi, step, is_count)
        if is_count:
            lo = max(lo, 0.0)
        p = prob_interval(m, lo - 0.5, hi + 0.5) if is_count else prob_interval(m, lo, hi)
        if is_count and hi <= 0:
            continue  # trivial "zero days" claim
        pop = WINDOWS[wkey][2] * vpop
        # Counts with tiny climatological values are rarely claimed by others.
        if is_count and max(m["raw"].values()) < 3:
            continue
        ann = f"trend {m['trend']:+.3f}/yr" + (f", persistence phi={m['phi']:.2f}" if m["phi"] else "") + (
            f", ENSO gamma={m['gamma']:+.2f} (OND Nino3.4 {m['nino_x']:+.2f})" if m["gamma"] and m["nino_x"] is not None else "")
        cands.append({
            "priority": pop * edge(p), "loc": loc,
            "claim": {"target": target(loc, wkey, var),
                      "assertion": {"kind": "interval", "low": lo, "high": hi},
                      "confidence": round(min(max(p, 0.5), 0.95), 3),
                      "rationale": f"{len(m['raw'])}-yr ERA5 window stats, {ann}; central {COVERAGE:.0%} predictive interval"},
        })
        # Year-over-year booleans.
        if var in ("temp_mean_c", "precip_sum_mm"):
            bvar = "warmer_than_year_x" if var == "temp_mean_c" else "wetter_than_year_x"
            pt = 1 - statistics.NormalDist(m["mu"], m["sigma"]).cdf(m["z_x"])
            conf = max(pt, 1 - pt)
            if conf >= BOOL_MIN_CONF:
                bpop = WINDOWS[wkey][2] * (0.7 if var == "temp_mean_c" else 0.5)
                cands.append({
                    "priority": bpop * edge(conf), "loc": loc,
                    "claim": {"target": target(loc, wkey, bvar),
                              "assertion": {"kind": "boolean", "value": pt > 0.5},
                              "confidence": round(min(conf, 0.95), 3),
                              "rationale": f"Year X value {inverse(tf, m['z_x']):.1f} vs forecast {inverse(tf, m['mu']):.1f} ({ann}): regression to trend"},
                })

    # Categorical hottest / wettest month over the full year.
    for loc in sorted(daily):
        years = sorted({int(d[:4]) for d in daily[loc]})
        for var, col, agg, pop in (("hottest_month", "temperature_2m_mean", "mean", 0.5),
                                   ("wettest_month", "precipitation_sum", "sum", 0.4)):
            counts = month_frequency(daily, loc, years, col, agg)
            n = sum(counts.values())
            if n < 6:
                continue
            best, c = max(sorted(counts.items()), key=lambda kv: kv[1])
            p = (c + 0.5) / (n + 3.0)
            if p >= CAT_MIN_CONF:
                cands.append({
                    "priority": pop * edge(p), "loc": loc,
                    "claim": {"target": target(loc, "ANN", var),
                              "assertion": {"kind": "categorical", "value": best},
                              "confidence": round(min(p, 0.95), 3),
                              "rationale": f"Month {best} was the {var.split('_')[0]} month in {c}/{n} years of record"},
                })

    cands.sort(key=lambda c: (-c["priority"], c["loc"], c["claim"]["target"]["variable"], c["claim"]["target"]["window_start"]))
    per_loc = defaultdict(int)
    seen = set()
    out = []
    for c in cands:
        t = c["claim"]["target"]
        key = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if key in seen or per_loc[c["loc"]] >= MAX_PER_LOCATION or c["priority"] <= 0:
            continue
        seen.add(key)
        per_loc[c["loc"]] += 1
        out.append(c["claim"])
        if len(out) >= MAX_CLAIMS:
            break
    for i, c in enumerate(out, 1):
        c["claim_id"] = f"p3_{i:03d}"
    return [{"claim_id": c["claim_id"], **{k: c[k] for k in ("target", "assertion", "confidence", "rationale")}} for c in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, daily, meta, nino = load(a.data_dir)
    if X != a.year_x:
        raise SystemExit(f"manifest year_x {X} != --year-x {a.year_x}")
    models = build(X, daily, nino)
    claims = make_claims(X, daily, meta, models)
    for c in claims:
        assert c["target"]["window_start"].startswith(f"{X + 1}-") and c["target"]["window_end"].startswith(f"{X + 1}-")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)
    print(f"[predict3] wrote {len(claims)} claims for {X + 1} to {a.out}")


if __name__ == "__main__":
    main()
