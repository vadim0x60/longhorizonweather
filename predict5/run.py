#!/usr/bin/env python3
"""predict5 — The Secular Warming & Trend Extrapolator.

Bets that year X+1 continues the multi-decadal warming signal. For each target
a per-location OLS trend over the trailing history (shrunk toward a global
warming prior) gives a predictive Normal distribution, and from it the
probability that a warm-side claim holds.

Scoring only rewards claims that contradict another predictor, so the claims
are aimed at targets rivals commonly use (calendar year, seasons, quarters,
months) and at assertions rivals commonly make on the cold side of the trend:
`warmer_than_year_x = false` after a warm year X, `le`/`lt` thresholds at the
15-year baseline, and climatological intervals whose top sits below the trend.
FAMILY_WEIGHT ranks the claim families by backtested contradiction value.

Offline, stdlib only, reads only --data-dir (dates <= X-12-31).
"""
import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import date, timedelta

MAX_CLAIMS = 150
HERE = os.path.dirname(os.path.abspath(__file__))
PRE1980_PATH = os.path.join(HERE, "data", "pre1980_daily.csv.gz")  # Open-Meteo archive, 1964..1979
# Round 7 backtest (corrected data, round-6 field, X = 1980..2023): without the archive -278
# (X <= 1995 -302), with it +56 (X <= 1995 +32, X = 1988 -26 -> +4); 1996+ unchanged at +24.
USE_PRE1980 = True
HISTORY_START = 1964 if USE_PRE1980 else 1980
PER_LOCATION = 12   # CONTRACTS.md round 3: at most 12 claims per location_id
MIN_POINTS = 60     # CONTRACTS.md round 3: at least 60 point claims
MIN_YEARS = 1       # shortest usable history (X = 1980 has only 1980 itself)
WF_MIN_YEARS = 5    # shortest history used inside the walk-forward and Sep-Dec fits
PRIOR_SLOPE_C_PER_YR = 0.025  # global-mean land warming since 1980, roughly
PRIOR_WEIGHT_YEARS = 16.0      # prior counts like this many years of slope evidence (temperatures)
COUNT_PRIOR_WEIGHT_YEARS = 8.0  # same, for hot/frost day counts
# ENSO lag helped point skill for daily-mean and daily-min temperatures only (backtest).
ENSO_VARS = ("temp_mean_c", "temp_mean_min_c")
ENSO_RIDGE_YEARS = 6.0         # ridge on the ENSO lag coefficient, in years of evidence; None disables

_MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# (name, start_mmdd, end_mmdd); windows that do not cross a year boundary.
WINDOWS = [
    ("year", "01-01", "12-31"),
    ("JF", "01-01", "02-28"),
    ("MAM", "03-01", "05-31"),
    ("JJA", "06-01", "08-31"),
    ("SON", "09-01", "11-30"),
    ("Q1", "01-01", "03-31"),
    ("Q2", "04-01", "06-30"),
    ("Q3", "07-01", "09-30"),
    ("Q4", "10-01", "12-31"),
] + [(f"M{m:02d}", f"{m:02d}-01", f"{m:02d}-{_MDAYS[m - 1]:02d}") for m in range(1, 13)]

FIELDS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max"]
TEMP_VARS = ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c")
COUNT_VARS = ("hot_days_ge_30c", "frost_days_lt_0c")

# family -> (weight, min_p); a candidate scores weight * (2p - 1). Against any disjoint
# rival claim a calibrated claim with p > 0.5 has expected score >= 2p - 1, so the weights
# only steer the 150 slots toward targets rivals actually claim (calendar year and seasons
# of temp_mean_c most often). Backtest vs the round-2 rival field (X = 1996..2023):
# temperature thresholds at the predictive 35th percentile (q65) hit 0.66 when claimed at
# p = 0.65; `gt` baseline + 1 sd on the annual mean won 7 of 8 contradictions (mostly vs
# `le` claims); warmer_than_year_x = false, and = true below p 0.75, lost to
# persistence-aware rivals, so those are not emitted. Unlisted families are never emitted.
FAMILY_WEIGHT = {
    "temp_mean_c|gt_clim_hi|year": (4.0, 0.55),
    "temp_mean_c|gt_q65|year": (3.0, 0.6),
    "temp_mean_c|gt_q65|season": (2.0, 0.6),
    "warmer|T|year": (2.0, 0.75),
    "warmer|T|season": (1.6, 0.75),
    "temp_mean_max_c|gt_q65|year": (1.5, 0.6),
    "temp_mean_min_c|gt_q65|year": (1.5, 0.6),
    "hot_days_ge_30c|gt_base|year": (1.0, 0.7),
    "frost_days_lt_0c|lt_base|year": (1.0, 0.7),
    "temp_mean_max_c|gt_q65|season": (0.8, 0.6),
    "temp_mean_min_c|gt_q65|season": (0.8, 0.6),
    "hot_days_ge_30c|gt_base|season": (0.8, 0.7),
    "frost_days_lt_0c|lt_base|season": (0.8, 0.7),
    "warmer|T|quarter": (0.6, 0.8),
    "temp_mean_c|gt_q65|quarter": (0.6, 0.6),
    "temp_mean_c|gt_q65|month": (0.3, 0.6),
}


# Point claims, in priority order per location. Backtest X = 1996..2023 of this forecaster vs a
# 15-year-climatology point forecaster (duels with gap >= min_gap): annual temp_mean_c 152-67,
# annual temp_mean_max_c 209-102, annual temp_mean_min_c 130-68, seasonal temp_mean_c 593-446,
# hot days ~even, frost days ~even. Precipitation is not claimed (no edge over climatology).
# Round 4: annual hot-day points lost duels to the master rival field in every variant (-8 to -27
# over X = 1985..2023), so hot days are only claimed as thresholds.
POINT_PLAN = [
    ("temp_mean_c", "year"), ("temp_mean_max_c", "year"), ("temp_mean_min_c", "year"),
    ("temp_mean_c", "JJA"), ("temp_mean_c", "JF"),
    ("temp_mean_c", "MAM"), ("temp_mean_c", "SON"), ("temp_mean_max_c", "JJA"),
]
# Point values keep a share lambda of the trend departure from the 15-year mean, chosen per year
# by walk-forward error (point_damping); at least half the secular signal is always kept.
# Round 4 backtest vs the master rival field, X = 1985..2023: fixed full trend +25 (early -22),
# lambda in {0..1} +28 (late -55), lambda in {0.5, 0.75, 1} +135 (early +86, late +49).
# Round 5 (round-4 rival field): {0.5..1} +266, adding 1.25 +330, adding 1.25 and 1.5 +373
# (X <= 1995 +6, 1996+ +367); a 0.25 floor (+219) and carried-over persistence (+402 but
# X <= 1995 -16) did not hold in both eras. lambda > 1 is only chosen when the shrunk trend has
# been under-forecasting the last WF_YEARS years.
# Round 8 (round-7 field, corrected data, X = 1980..2023, with Sep-Dec 0.2): floor 0.5 +178,
# floor 0.75 +271 (X <= 1995 +131, 1996+ +140); floor 1.0 lost (-69 without Sep-Dec change).
# At X = 2014 the 0.5 floor was chosen for all 120 points after the slow-warming 2005-2014.
DAMPINGS = (0.75, 1.0, 1.25, 1.5)
DEFAULT_DAMPING = 0.5
# lambda <= this * years of history; None = no cap. Round 7 backtest (corrected data, round-6
# field, X = 1980..2023): caps of 0.05 / 0.075 / 0.1 per year gave -409 / -313 / -269 vs -278
# uncapped without the pre-1980 archive, and change nothing once it is used, so it stays off.
LAMBDA_CAP_PER_YEAR = None
# Weight of year X's Sep-Dec departure from its own shrunk trend, added to temperature points
# (late-year ocean/ENSO anomalies persist). Round 6 backtest vs the round-5 rival field,
# X = 1985..2023: none +277, fixed 0.1 +606 (X <= 1995 +76, 1996+ +530), 0.2 +548, 0.3 +530;
# letting the walk-forward choose from {0, 0.2, 0.4} gave +322 (X <= 1995 -2).
# Round 8 re-test (round-7 field, X = 1980..2023, lambda floor 0.75): 0.1 +60, 0.2 +271.
# Adding predict4's 0.2 carry-over of year X's own residual: +154 (X <= 1995 -26), not used.
# Round 11 (round-10 field, X = 1980..2023): 0.2 +11 (X <= 1995 -14), 0.25 +95 (X <= 1995 +27,
# 1996+ +68), 0.3 +31. 0.3 beat 0.2 against the round-8 and round-9 fields but lost to it vs round 7.
SOND_CARRY = (0.25,)
PERSISTENCE = (0.0,)     # share of year X's departure from trend carried into X+1
POINT_ONLY_VARS = ()     # variables claimed only as points
WF_YEARS = 10
POINT_SLOTS = 120   # the remaining 30 slots go to the best threshold/boolean claims


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def load_daily(data_dir, max_date):
    """{location_id: {"lat","lon","days": {date: tuple}}} from history + year_x, deduped."""
    locs = {}
    for name in ("daily_history.csv", "daily_year_x.csv"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                d = r["date"]
                if d > max_date:
                    continue  # never look past X-12-31
                loc = locs.setdefault(r["location_id"], {"lat": float(r["lat"]), "lon": float(r["lon"]), "days": {}})
                vals = []
                for c in FIELDS:
                    v = r.get(c, "")
                    vals.append(float(v) if v not in ("", None, "nan", "NaN") else None)
                loc["days"][d] = tuple(vals)
    # Bundled 1964..1979 archive (predict5/data), merged only for locations in --data-dir and only
    # before the data dir's own first day, so it fills the years datacollect starts after.
    if USE_PRE1980 and os.path.exists(PRE1980_PATH) and locs:
        first = min(min(l["days"]) for l in locs.values())
        with gzip.open(PRE1980_PATH, "rt", newline="") as f:
            for r in csv.DictReader(f):
                d = r["date"]
                loc = locs.get(r["location_id"])
                if loc is None or d >= first or d > max_date:
                    continue
                loc["days"][d] = tuple(float(r[c]) if r[c] not in ("", "nan", "NaN") else None for c in FIELDS)
    return locs


def window_days(year, s, e):
    if e == "02-28" and is_leap(year):
        e = "02-29"  # verify moves Feb-end windows to the true month end
    d0 = date.fromisoformat(f"{year}-{s}")
    d1 = date.fromisoformat(f"{year}-{e}")
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def stat(days, ds, var):
    rows = [days.get(d) for d in ds]
    if sum(r is not None for r in rows) < 0.9 * len(ds):
        return None
    rows = [r for r in rows if r is not None]
    col = {"temp_mean_c": 0, "temp_mean_max_c": 1, "temp_mean_min_c": 2}.get(var)
    if col is not None:
        xs = [r[col] for r in rows if r[col] is not None]
        return sum(xs) / len(xs) if xs else None
    if var == "hot_days_ge_30c":
        return float(sum(1 for r in rows if r[1] is not None and r[1] >= 30.0))
    if var == "frost_days_lt_0c":
        return float(sum(1 for r in rows if r[2] is not None and r[2] < 0.0))
    if var == "precip_sum_mm":
        xs = [r[3] for r in rows if r[3] is not None]
        return sum(xs) * len(ds) / len(xs) if xs else None
    raise ValueError(var)


def load_enso(data_dir, x):
    """{year: Oct-Dec mean Nino-3.4 anomaly} for years <= x, or {} without indices."""
    path = os.path.join(data_dir, "indices_monthly.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    series = doc.get("series", doc).get("nino34_anom") or []
    acc = defaultdict(list)
    for r in series:
        y, m = int(r["date"][:4]), int(r["date"][5:7])
        if y <= x and m >= 10:
            acc[y].append(float(r["value"]))
    return {y: sum(v) / len(v) for y, v in acc.items() if len(v) == 3}


def enso_adjust(yrs, ys, pred_fn, enso, x):
    """Lagged ENSO term: residual(y) ~ beta * NinoOND(y-1). Returns (shift for x+1, beta)."""
    if ENSO_RIDGE_YEARS is None or x not in enso:
        return 0.0, 0.0
    pts = [(enso[t - 1], y - pred_fn(t)) for t, y in zip(yrs, ys) if (t - 1) in enso]
    if len(pts) < 8:
        return 0.0, 0.0
    nbar = sum(n for n, _ in pts) / len(pts)
    snn = sum((n - nbar) ** 2 for n, _ in pts)
    var_n = snn / len(pts)
    beta = sum((n - nbar) * r for n, r in pts) / (snn + ENSO_RIDGE_YEARS * var_n)
    return beta * (enso[x] - nbar), beta


def trend_forecast(years, ys, target_year, prior_slope, prior_weight=None):
    """OLS slope shrunk toward prior_slope; returns (pred, predictive_sd, slope)."""
    n = len(ys)
    ybar = sum(ys) / n
    tbar = sum(years) / n
    sxx = sum((t - tbar) ** 2 for t in years)
    b_ols = sum((t - tbar) * (y - ybar) for t, y in zip(years, ys)) / sxx if sxx else 0.0
    pw = PRIOR_WEIGHT_YEARS if prior_weight is None else prior_weight
    w_prior = pw ** 3 / 12.0  # sxx of a pw-year evenly spaced series
    b = (sxx * b_ols + w_prior * prior_slope) / (sxx + w_prior)
    resid = [y - (ybar + b * (t - tbar)) for t, y in zip(years, ys)]
    s = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    h = target_year - tbar
    sd = s * math.sqrt(1.0 + 1.0 / n + h * h / (sxx + w_prior))
    return ybar + b * h, max(sd, 1e-6), b, (lambda t: ybar + b * (t - tbar))


def _cand(lid, loc, wname, s, e, var, assertion, p, family, why):
    return dict(lid=lid, loc=loc, w=(s, e), wname=wname, var=var, assertion=assertion,
                p=p, family=family, why=why)


def window_kind(wname):
    if wname == "year":
        return "year"
    if wname in ("JF", "MAM", "JJA", "SON"):
        return "season"
    return "quarter" if wname[0] == "Q" else "month"


def sond_resid(sond, year, prior, pw):
    """Year's Sep-Dec departure from the shrunk trend of Sep-Dec over the 16 years up to it; 0 if unknown."""
    if not sond:
        return 0.0
    sub = [(t, v) for t, v in zip(*sond) if year - 15 <= t <= year]
    if len(sub) < WF_MIN_YEARS or sub[-1][0] != year:
        return 0.0
    sy, sv = [t for t, _ in sub], [v for _, v in sub]
    fit = trend_forecast(sy, sv, year + 1, prior, pw)[3]
    return sv[-1] - fit(year)


def point_damping(items, enso, x):
    """Per (variable, window kind), the share lambda of the trend + ENSO departure from the 15-year
    mean that minimized pooled absolute error in a walk-forward backtest over target years
    x-WF_YEARS+1 .. x, each forecast only from years before it. Uses data <= x only."""
    grid = [(lm, kp, kc) for lm in DAMPINGS for kp in PERSISTENCE for kc in SOND_CARRY]
    err = defaultdict(lambda: [0.0] * len(grid))
    for it in items:
        yrs, ys, var = it["yrs"], it["ys"], it["var"]
        for j, t in enumerate(yrs):
            if t <= x - WF_YEARS:
                continue
            sub = [(a, b) for a, b in zip(yrs[:j], ys[:j]) if a >= t - 16]
            if len(sub) < WF_MIN_YEARS:
                continue
            sy, sv = [a for a, _ in sub], [b for _, b in sub]
            base = sum(sv) / len(sv)
            pred, _, _, fit = trend_forecast(sy, sv, t, it["prior"], it["pw"])
            if var in ENSO_VARS:
                pred += enso_adjust(sy, sv, fit, {k: v for k, v in enso.items() if k < t}, t - 1)[0]
            acc = err[(var, it["wk"])]
            last_resid = sv[-1] - fit(sy[-1]) if sy[-1] == t - 1 else 0.0
            sr = sond_resid(it.get("sond"), t - 1, it["prior"], it["pw"])
            for i, (lm, kp, kc) in enumerate(grid):
                v = base + lm * (pred - base) + kp * last_resid + kc * sr
                if var not in TEMP_VARS:
                    v = max(v, 0.0)
                acc[i] += abs(v - ys[j])
    # ties: prefer more trend, then less persistence
    return {k: grid[min(range(len(grid)), key=lambda i: (v[i], -grid[i][0], grid[i][1], grid[i][2]))]
            for k, v in err.items() if any(v)}


def build_pool(locs, x, enso=None):
    """Every candidate claim, possibly several per target (one per family)."""
    enso = enso or {}
    target_year = x + 1
    hist_years = list(range(max(HISTORY_START, x - 15), x + 1))
    items = []
    for lid, loc in locs.items():
        days = loc["days"]
        for wname, s, e in WINDOWS:
            span = {yr: window_days(yr, s, e) for yr in hist_years}
            for var in TEMP_VARS + COUNT_VARS + POINT_ONLY_VARS:
                series = [(yr, stat(days, span[yr], var)) for yr in hist_years]
                series = [(t, v) for t, v in series if v is not None]
                if len(series) < MIN_YEARS or series[-1][0] != x:
                    continue
                yrs = [t for t, _ in series]
                ys = [v for _, v in series]
                base = sum(ys) / len(ys)
                if var in TEMP_VARS:
                    prior, pw = PRIOR_SLOPE_C_PER_YR * (1.3 if var == "temp_mean_min_c" else 1.0), None
                elif var in POINT_ONLY_VARS:
                    prior, pw = 0.0, PRIOR_WEIGHT_YEARS
                else:
                    if base < 2.0 or base > 0.95 * len(span[x]):
                        continue
                    prior = (0.02 if var == "hot_days_ge_30c" else -0.02) * base
                    pw = COUNT_PRIOR_WEIGHT_YEARS
                pred, sd, b, fit = trend_forecast(yrs, ys, target_year, prior, pw)
                sond = None
                if var in TEMP_VARS and SOND_CARRY != (0.0,):
                    sser = [(yr, stat(days, window_days(yr, "09-01", "12-31"), var)) for yr in hist_years]
                    sser = [(t, v) for t, v in sser if v is not None]
                    sond = ([t for t, _ in sser], [v for _, v in sser])
                items.append(dict(sond=sond, lid=lid, loc=loc, wname=wname, s=s, e=e, wk=window_kind(wname),
                                  var=var, yrs=yrs, ys=ys, base=base, pred=pred, sd=sd, b=b, fit=fit,
                                  prior=prior, pw=pw))
    lam = point_damping(items, enso, x)
    out = []
    for it in items:
        lid, loc, wname, s, e, wk, var = (it[k] for k in ("lid", "loc", "wname", "s", "e", "wk", "var"))
        yrs, ys, base, sd, b = it["yrs"], it["ys"], it["base"], it["sd"], it["b"]
        sd_clim = math.sqrt(sum((v - base) ** 2 for v in ys) / (len(ys) - 1)) if len(ys) > 1 else sd
        shift = enso_adjust(yrs, ys, it["fit"], enso, x)[0] if var in ENSO_VARS else 0.0
        pred = it["pred"] + shift
        if var not in TEMP_VARS:
            pred = max(pred, 0.0)
        # Point forecast: 15y mean plus the walk-forward-chosen share of the trend + ENSO departure.
        lm, kp, kc = lam.get((var, wk), (DEFAULT_DAMPING, 0.0, 0.0))
        if LAMBDA_CAP_PER_YEAR is not None:
            lm = min(lm, LAMBDA_CAP_PER_YEAR * len(yrs))  # short records: trust the trend less
        pv = (base + lm * (pred - base) + kp * (ys[-1] - it["fit"](yrs[-1]))
              + kc * sond_resid(it["sond"], x, it["prior"], it["pw"]))
        if var not in TEMP_VARS:
            pv = max(pv, 0.0)
        out.append(_cand(lid, loc, wname, s, e, var,
                         {"kind": "point", "value": round(pv, 2 if var in TEMP_VARS else 1)}, 1.0,
                         f"{var}|point|{wk}",
                         f"point forecast: 15y mean {base:.2f}, trend {b * 10:+.2f}/decade,"
                         f" ENSO lag {shift:+.2f}, damping {lm:.2f}, persistence {kp:.2f}, Sep-Dec carry {kc:.2f} (walk-forward) -> {pv:.2f}"))
        if var in POINT_ONLY_VARS:
            continue
        if var in TEMP_VARS:
            info = (f"15y mean {base:.2f}C, trend {b * 10:+.2f}C/decade, ENSO lag {shift:+.2f}C -> {pred:.2f}C (sd {sd:.2f})")
            # Thresholds at points where cold-leaning rivals put their upper bound.
            levels = {
                "base": base,
                "clim_q": base + 0.5 * sd_clim,
                "clim_hi": base + sd_clim,
                "q60": pred - 0.253 * sd,   # P(>t) = 0.60
                "q65": pred - 0.385 * sd,   # P(>t) = 0.65
                "q70": pred - 0.524 * sd,   # P(>t) = 0.70
                "last": ys[-1],
            }
            for lname, t in levels.items():
                t = round(t, 2)
                p = 1.0 - phi((t - pred) / sd)
                out.append(_cand(lid, loc, wname, s, e, var,
                                 {"kind": "threshold", "op": "gt", "value": t}, p,
                                 f"{var}|gt_{lname}|{wk}", f"{info}; P(>{t})={p:.2f}"))
            if var == "temp_mean_c":
                yx = ys[-1]
                pw = 1.0 - phi((yx - pred) / sd)
                val = pw >= 0.5
                out.append(_cand(lid, loc, wname, s, e, "warmer_than_year_x",
                                 {"kind": "boolean", "value": val}, max(pw, 1 - pw),
                                 f"warmer|{'T' if val else 'F'}|{wk}",
                                 f"{x} was {yx:.2f}C; {info}"))
        else:
            sd = max(sd, 1.0)
            info = f"15y mean {base:.1f} days, trend {b * 10:+.1f}/decade -> {pred:.1f} (sd {sd:.1f})"
            if var == "hot_days_ge_30c":
                for lname, t in {"base": math.floor(base), "clim_hi": math.floor(base + sd_clim)}.items():
                    p = 1.0 - phi((t + 0.5 - pred) / sd)
                    out.append(_cand(lid, loc, wname, s, e, var,
                                     {"kind": "threshold", "op": "gt", "value": t}, p,
                                     f"{var}|gt_{lname}|{wk}", f"{info}; P(>{t})={p:.2f}"))
            else:
                for lname, t in {"base": math.ceil(base), "clim_lo": math.ceil(base - sd_clim)}.items():
                    p = phi((t - 0.5 - pred) / sd)
                    out.append(_cand(lid, loc, wname, s, e, var,
                                     {"kind": "threshold", "op": "lt", "value": t}, p,
                                     f"{var}|lt_{lname}|{wk}", f"{info}; P(<{t})={p:.2f}"))
    return out


def select(pool, x, weights=None, point_plan=None):
    """Point claims from POINT_PLAN first, then the best other family per target,
    ranked by family weight x edge (2p-1), within the per-location and total caps."""
    weights = FAMILY_WEIGHT if weights is None else weights
    point_plan = POINT_PLAN if point_plan is None else point_plan
    rank = {k: i for i, k in enumerate(point_plan)}
    points = [c for c in pool if c["assertion"]["kind"] == "point" and (c["var"], c["wname"]) in rank]
    # Interleave locations so every location gets its top-priority points before anyone's later ones.
    points.sort(key=lambda c: (rank[(c["var"], c["wname"])], c["lid"]))
    scored = [(math.inf, c) for c in points[:POINT_SLOTS]]
    rest = []
    for c in pool:
        if c["assertion"]["kind"] == "point":
            continue
        w, min_p = weights.get(c["family"], (0.0, 1.0))
        if w <= 0 or c["p"] < min_p:
            continue
        rest.append((w * (2 * c["p"] - 1), c))
    rest.sort(key=lambda sc: (-sc[0], sc[1]["lid"], sc[1]["wname"], sc[1]["var"]))
    scored += rest
    seen, per_loc, claims = set(), defaultdict(int), []
    y1 = x + 1
    for _, c in scored:
        key = (c["lid"], c["w"], c["var"])
        if key in seen or per_loc[c["lid"]] >= PER_LOCATION:
            continue
        seen.add(key)
        per_loc[c["lid"]] += 1
        end = c["w"][1]
        if end == "02-28" and is_leap(y1):
            end = "02-29"
        claims.append({
            "claim_id": f"p5_{len(claims) + 1:03d}",
            "target": {
                "location_id": c["lid"], "lat": c["loc"]["lat"], "lon": c["loc"]["lon"],
                "window_start": f"{y1}-{c['w'][0]}", "window_end": f"{y1}-{end}",
                "variable": c["var"],
            },
            "assertion": c["assertion"],
            # Model p runs hot (interannual noise is fat-tailed); shrink toward 0.5.
            "confidence": (0.6 if c["assertion"]["kind"] == "point"
                           else round(0.5 + 0.7 * (min(c["p"], 0.99) - 0.5), 3)),
            "rationale": f"Secular warming extrapolation ({c['wname']}): {c['why']}",
        })
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    max_date = f"{a.year_x}-12-31"
    man_path = os.path.join(a.data_dir, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
        if man.get("max_date", max_date) < max_date:
            max_date = man["max_date"]
    locs = load_daily(a.data_dir, max_date)
    claims = select(build_pool(locs, a.year_x, load_enso(a.data_dir, a.year_x)), a.year_x)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)
    print(f"predict5: wrote {len(claims)} claims for {a.year_x + 1} to {a.out}")


if __name__ == "__main__":
    main()
