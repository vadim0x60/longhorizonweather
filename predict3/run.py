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
  * Mean model also carries a pooled lag-1 persistence of each variable's own
    detrended anomaly (helps precipitation and day counts).
  * Claims (contract: >= 60 points, <= 12 per location, <= 150 total):
      - 100 `point` claims for ANN/JJA temp_mean_c and ANN precip_sum_mm,
        temp_mean_max_c, temp_mean_min_c at all 20 locations. Point = clim +
        lambda*(median - clim), lambda = 1 with >= 16 years of record, else 0.67*n/16:
        short records (X near 1980) give noisy trends that lost Round 4 (X=1987);
      - directional thresholds halfway between the trailing climatology and our
        median, only when P >= 0.65;
      - warmer/wetter-than-year-X booleans when P >= 0.65; hottest/wettest month.
    Non-point claims are ranked by how many rival claims a target attracts (POP,
    measured offline on past years) times the expected net margin.

Dev-time evaluation (rivals' Round 4 code run offline with Tinker off, scored by
verify/run.py on locally collected truth), split by era to avoid fitting one decade:
X=1985..1995 +58, X=1998..2023 +695 (odd years +485, even +268). The simulation
reproduces live Round 4 (X=1987: -34 simulated vs -37 live with the previous settings;
+8 with these). Trend prior sd 0.015 beat 0.01 and 0.03 across both eras.
Model ablations: Sep-Dec persistence clearly helps; per-location ENSO fits hurt;
pooled ENSO is neutral-to-slightly-positive for temperature.

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
import sys
from collections import defaultdict

ND = statistics.NormalDist()

MAX_CLAIMS = 150
MAX_PER_LOCATION = 12   # Round 3 contract cap per location_id
LAMBDA = 0.5             # threshold position between climatology (0) and our median (1)
THRESHOLD_WEIGHT = 0.3   # base ranking weight of threshold claims relative to booleans
SIGMA_INFLATE = 1.00     # calibrated so out-of-sample z-scores have ~unit spread (backtests X=2006..2023)
PHI_MAX = 0.7            # cap on pooled persistence coefficient
OWN_LAG = True           # pooled lag-1 persistence of each variable's own detrended anomaly
RHO_MAX = 0.6
USE_ENSO = True          # pooled ENSO term, temperature variables only (backtest: small gain 2015-2023, neutral 2006-2014)
ENSO_VARS = {"temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c"}
OTHER_TREND_SHRINK = 0.5  # extra shrink of EB-weighted trends for non-temperature variables
BOOL_MIN_CONF = 0.65
# Round 3: point duels (closer to truth wins once two points differ by min_gap). Point claims go on the
# most contested numeric targets for every location; >= 60 are required by the contract.
POINT_PAIRS = [("ANN", "temp_mean_c"), ("JJA", "temp_mean_c"), ("ANN", "precip_sum_mm"), ("ANN", "temp_mean_max_c"), ("ANN", "temp_mean_min_c")]
MIN_POINTS = 60
POINT_LAMBDA = {"temp": (0.67, 16), "other": (0.67, 16)}  # (short-record scale, full-record years): lambda = 1 at >= 16 yrs, else scale*n/16
POINT_PRIORITY = 100.0   # points are placed before any other claim
POINT_CONF = 0.59        # backtest share of point duels won vs naive climatology/trend/persistence points
THR_MIN_CONF = 0.65
WET_BOOL_MIN_CONF = 0.65
CAT_MIN_CONF = 0.5

# window key -> (start MM-DD, end MM-DD). A Feb end is always 02-28, as rivals use.
WINDOWS = {
    "ANN": ("01-01", "12-31"), "JJA": ("06-01", "08-31"), "MAM": ("03-01", "05-31"),
    "SON": ("09-01", "11-30"), "JF": ("01-01", "02-28"), "JFM": ("01-01", "03-31"),
    "AMJ": ("04-01", "06-30"), "JAS": ("07-01", "09-30"), "OND": ("10-01", "12-31"),
}
for _m in range(1, 13):
    WINDOWS[f"M{_m:02d}"] = (f"{_m:02d}-01", f"{_m:02d}-{[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][_m - 1]:02d}")

# variable -> (daily column, aggregation, transform, rounding step)
CONT_VARS = {
    "temp_mean_c": ("temperature_2m_mean", "mean", "id", 0.05),
    "precip_sum_mm": ("precipitation_sum", "sum", "sqrt", 1.0),
    "temp_max_c": ("temperature_2m_max", "max", "id", 0.1),
    "temp_min_c": ("temperature_2m_min", "min", "id", 0.1),
    "temp_mean_max_c": ("temperature_2m_max", "mean", "id", 0.05),
    "temp_mean_min_c": ("temperature_2m_min", "mean", "id", 0.05),
    "precip_max_day_mm": ("precipitation_sum", "max", "sqrt", 0.5),
    "wind_max_kmh": ("wind_speed_10m_max", "max", "id", 0.5),
    "hot_days_ge_30c": ("temperature_2m_max", "count_ge30", "sqrt", 1),
    "frost_days_lt_0c": ("temperature_2m_min", "count_lt0", "sqrt", 1),
    "rainy_days_ge_1mm": ("precipitation_sum", "count_ge1", "id", 1),
}
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
TEMP_VARS = {"temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c", "hot_days_ge_30c", "frost_days_lt_0c"}
BOOL_BASE = {"warmer_than_year_x": "temp_mean_c", "wetter_than_year_x": "precip_sum_mm"}

# Expected number of rival claims per location on each (window, variable), measured by running
# the other predictors' published code offline on past years. Contradictions only happen on
# shared targets, so claims are ranked by this popularity times their expected margin.
POP = {
    ("ANN", "temp_mean_c"): 3.8, ("ANN", "warmer_than_year_x"): 3.7, ("JJA", "warmer_than_year_x"): 3.3,
    ("ANN", "wetter_than_year_x"): 3.2, ("JJA", "temp_mean_c"): 3.1, ("ANN", "precip_sum_mm"): 2.7,
    ("JJA", "wetter_than_year_x"): 2.5, ("MAM", "warmer_than_year_x"): 1.7, ("SON", "warmer_than_year_x"): 1.6,
    ("ANN", "temp_mean_max_c"): 1.6, ("ANN", "rainy_days_ge_1mm"): 1.6, ("ANN", "hot_days_ge_30c"): 1.5,
    ("JF", "temp_mean_c"): 1.35, ("MAM", "wetter_than_year_x"): 1.3, ("ANN", "temp_mean_min_c"): 1.3,
    ("SON", "temp_mean_c"): 1.25, ("MAM", "temp_mean_c"): 1.2, ("SON", "wetter_than_year_x"): 1.2,
    ("ANN", "wind_max_kmh"): 1.05, ("ANN", "temp_max_c"): 1.05, ("ANN", "temp_min_c"): 1.0,
    ("ANN", "precip_max_day_mm"): 1.0, ("MAM", "hottest_month"): 0.9, ("ANN", "frost_days_lt_0c"): 0.85,
    ("SON", "hottest_month"): 0.85, ("M07", "temp_mean_c"): 0.8, ("M01", "temp_mean_c"): 0.8,
    ("JF", "warmer_than_year_x"): 0.75, ("JFM", "warmer_than_year_x"): 0.7, ("ANN", "hottest_month"): 0.65,
    ("OND", "warmer_than_year_x"): 0.65, ("JJA", "precip_sum_mm"): 0.65, ("M07", "warmer_than_year_x"): 0.55,
    ("M01", "warmer_than_year_x"): 0.55, ("OND", "wetter_than_year_x"): 0.55, ("JFM", "wetter_than_year_x"): 0.5,
    ("AMJ", "warmer_than_year_x"): 0.5, ("JAS", "warmer_than_year_x"): 0.45, ("M04", "warmer_than_year_x"): 0.4,
    ("JJA", "temp_max_c"): 0.4, ("JJA", "temp_min_c"): 0.4, ("M06", "warmer_than_year_x"): 0.35,
    ("M05", "warmer_than_year_x"): 0.35, ("M08", "warmer_than_year_x"): 0.35, ("JAS", "wetter_than_year_x"): 0.35,
    ("ANN", "wettest_month"): 0.3, ("JJA", "hottest_month"): 0.3,
}
# Continuous models needed: every popular continuous target plus the bases of the booleans.
PAIRS = sorted({(w, v) for w, v in POP if v in CONT_VARS} | {(w, BOOL_BASE[v]) for w, v in POP if v in BOOL_BASE})

# Trend priors (per year) in transformed units: warming for temperature, none otherwise.
TREND_PRIOR = {"temp": (0.03, 0.015), "other": (0.0, None)}  # sd 0.015: best over both eras X=1985..95 and 1998..2023 (0.03 lost the early era)


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
        start_md, end_md = WINDOWS[wkey]
        col, agg, tf, _ = CONT_VARS[var]
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
        # Pooled own-variable lag-1 persistence of the detrended residual (all variables).
        rho = 0.0
        if OWN_LAG:
            num = den = 0.0
            for loc, m in per_loc.items():
                for y in m["years"]:
                    if (y - 1) in m["res"]:
                        e = m["res"][y] - phi * sond_anom[loc].get(y - 1, 0.0)
                        num += m["res"][y - 1] * e
                        den += m["res"][y - 1] ** 2
            rho = min(max(num / den if den > 0 else 0.0, 0.0), RHO_MAX)
        # ENSO teleconnection per location, pooled across locations.
        est = {}
        for loc, m in per_loc.items():
            xs, rs = [], []
            for y in m["years"]:
                n_prev = nino_ond.get(y - 1)
                p = sond_anom[loc].get(y - 1, 0.0)
                if n_prev is not None and (y - 1) in m["res"]:
                    xs.append(n_prev)
                    rs.append(m["res"][y] - phi * p - rho * m["res"][y - 1])
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
                pred = (phi * sond_anom[loc].get(y - 1, 0.0) + rho * m["res"][y - 1] + g * nino_ond.get(y - 1, 0.0)) if (y - 1) in m["res"] else 0.0
                fitted_res.append(m["res"][y] - pred)
            n = len(fitted_res)
            k = 2  # per-location trend; phi and pooled gamma are shared across 20 locations
            s = math.sqrt(sum(r * r for r in fitted_res) / max(n - k, 2))
            t1 = X + 1
            lev = 1 + 1 / n + (t1 - m["my"]) ** 2 / m["syy"] if m["syy"] > 0 else 1 + 1 / n
            # Small-sample t inflation (approx) + misspecification factor.
            t_infl = 1 + 1.0 / max(n - k, 2)
            sigma = s * math.sqrt(lev) * t_infl * SIGMA_INFLATE
            mu = m["a"] + m["b"] * t1 + phi * sond_anom[loc].get(X, 0.0) + rho * m["res"][X] + g * nino_ond.get(X, 0.0)
            models[(loc, wkey, var)] = {
                "mu": mu, "sigma": sigma, "tf": tf, "phi": phi, "gamma": g, "trend": m["b"],
                "z_x": m["z"][X], "raw": m["raw"], "nino_x": nino_ond.get(X),
            }
    return models


# ---------------------------------------------------------------- claims


def contest_weight(p):
    """Expected net score per rival claim on a target: a disjoint rival claim costs us when we
    are wrong (margin 2p-1), and well-informed rivals disagree with us less often as p -> 1."""
    return max(0.0, (2 * p - 1) * 2 * (1 - p))


def window_month_winner(daily, loc, years, wkey, col, agg):
    """{year: "MM"} of the month with the highest monthly aggregate inside the window."""
    s_md, e_md = WINDOWS[wkey]
    months = range(int(s_md[:2]), int(e_md[:2]) + 1)
    out = {}
    for y in years:
        vals = {}
        for mth in months:
            last = (dt.date(y + (mth == 12), mth % 12 + 1, 1) - dt.timedelta(days=1)).day
            v = window_values(daily[loc], y, f"{mth:02d}-01", f"{mth:02d}-{last:02d}", col)
            if not v:
                break
            vals[f"{mth:02d}"] = aggregate(v, agg)
        else:
            out[y] = max(sorted(vals), key=lambda k: vals[k])
    return out


def threshold_claim(m, var):
    """Directional threshold between climatology and our forecast median.

    Rivals anchor on the trailing climatology (or on year X); a threshold on our side of the
    climatology, a fraction LAMBDA of the way to our median, is disjoint from every rival claim
    that sits on the other side of it, while still holding with probability > 0.5 for us."""
    _, _, tf, step = CONT_VARS[var]
    is_count = var in COUNT_VARS
    clim = sum(m["raw"].values()) / len(m["raw"])
    med = inverse(tf, m["mu"])
    thr = clim + LAMBDA * (med - clim)
    dist = statistics.NormalDist(m["mu"], m["sigma"])
    if med >= clim:
        op = "gt"
        thr = math.floor(thr) if is_count else math.floor(thr / step + 1e-9) * step
        p = 1 - dist.cdf(transform(tf, thr + (0.5 if is_count else 0.0)))
    else:
        op = "lt"
        thr = math.ceil(thr) if is_count else math.ceil(thr / step - 1e-9) * step
        if thr <= 0:
            return None
        p = dist.cdf(transform(tf, thr - (0.5 if is_count else 0.0)))
    thr = int(thr) if is_count or step >= 1 else round(thr, 2)
    return {"kind": "threshold", "op": op, "value": thr}, p, clim, med


def make_claims(X, daily, meta, models):
    T = X + 1
    cands = []

    def target(loc, wkey, var):
        s, e = window_dates(T, *WINDOWS[wkey])
        lat, lon = meta[loc]
        return {"location_id": loc, "lat": lat, "lon": lon, "window_start": s.isoformat(),
                "window_end": e.isoformat(), "variable": var}

    def add(priority, loc, wkey, var, assertion, p, why):
        cands.append({"priority": priority, "loc": loc, "claim": {
            "target": target(loc, wkey, var), "assertion": assertion,
            "confidence": round(min(max(p, 0.5), 0.97), 3), "rationale": why}})

    for (loc, wkey, var), m in sorted(models.items()):
        ann = f"trend {m['trend']:+.3f}/yr" + (f", Sep-Dec persistence phi={m['phi']:.2f}" if m["phi"] else "") + (
            f", ENSO gamma={m['gamma']:+.2f} x OND Nino3.4 {m['nino_x']:+.2f}" if m["gamma"] and m["nino_x"] is not None else "")
        pop = POP.get((wkey, var), 0.0)
        if (wkey, var) in POINT_PAIRS:
            # Predictive median: minimises expected absolute error, which decides point duels.
            med = inverse(m["tf"], m["mu"])
            clim = sum(m["raw"].values()) / len(m["raw"])
            short_scale, n_full = POINT_LAMBDA["temp" if var in TEMP_VARS else "other"]
            n = len(m["raw"])
            lam = 1.0 if n >= n_full else short_scale * n / n_full  # short records: trust trends less
            pv = clim + lam * (med - clim)
            add(POINT_PRIORITY + pop, loc, wkey, var, {"kind": "point", "value": round(pv, 2)}, POINT_CONF,
                f"point = climatology {clim:.2f} + {lam:.2f} x (median {med:.2f} - climatology) ({ann})")
        elif pop:
            r = threshold_claim(m, var)
            if r and r[1] >= THR_MIN_CONF and not (var in COUNT_VARS and max(m["raw"].values()) < 3):
                assertion, p, clim, med = r
                add(pop * (THRESHOLD_WEIGHT + contest_weight(p)), loc, wkey, var, assertion, p,
                    f"forecast median {med:.2f} vs {len(m['raw'])}-yr climatology {clim:.2f} ({ann})")
        for bvar, base in BOOL_BASE.items():
            bpop = POP.get((wkey, bvar), 0.0)
            if base != var or not bpop:
                continue
            pt = 1 - statistics.NormalDist(m["mu"], m["sigma"]).cdf(m["z_x"])
            conf = max(pt, 1 - pt)
            if conf >= (BOOL_MIN_CONF if bvar == "warmer_than_year_x" else WET_BOOL_MIN_CONF):
                add(bpop * contest_weight(conf), loc, wkey, bvar, {"kind": "boolean", "value": pt > 0.5}, conf,
                    f"year X {inverse(m['tf'], m['z_x']):.1f} vs forecast {inverse(m['tf'], m['mu']):.1f} ({ann})")

    # Categorical hottest / wettest month inside popular windows (empirical frequency, smoothed).
    for loc in sorted(daily):
        years = sorted({int(d[:4]) for d in daily[loc]})
        for (wkey, var), pop in POP.items():
            if var not in ("hottest_month", "wettest_month"):
                continue
            col, agg = ("temperature_2m_mean", "mean") if var == "hottest_month" else ("precipitation_sum", "sum")
            wins = window_month_winner(daily, loc, years, wkey, col, agg)
            if len(wins) < 6:
                continue
            counts = defaultdict(int)
            for y, mm in wins.items():
                counts[mm] += 1 + (y >= X - 4)  # recent years count double (warming shifts the seasonal peak)
            n = sum(counts.values())
            best = max(sorted(counts), key=lambda k: counts[k])
            p = (counts[best] + 0.5) / (n + 3.0)
            if p >= CAT_MIN_CONF:
                add(pop * contest_weight(p), loc, wkey, var, {"kind": "categorical", "value": best}, p,
                    f"month {best} won {counts[best]}/{n} (recency-weighted) years in this window")

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
    n_points = sum(c["assertion"]["kind"] == "point" for c in out)
    if n_points < MIN_POINTS:  # only possible with incomplete data; still emit what we have
        print(f"[predict3] WARNING: only {n_points} point claims (contract asks for {MIN_POINTS})", file=sys.stderr)
    return [{"claim_id": f"p3_{i:03d}", **c} for i, c in enumerate(out, 1)]


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
