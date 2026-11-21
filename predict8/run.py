#!/usr/bin/env python3
"""predict8: spatial-graph + pooled ridge ensemble with time-series CV (stdlib only).

For year X, reads only <data_dir>/daily_history.csv (dates <= X-12-31) and emits
claims about year X+1. Per variable, one pooled model is fit across all
locations and windows on standardized anomalies z = (s_t - clim_<t) / sd_<t, with
causal features computed only from years < t:
  ptrend     network-pooled warming/drying rate (all cities, same variable and
             window group) extrapolated to t
  trend      this series' own OLS trend extrapolation (z units)
  lag1/lag3  persistence of last year / last 3 years
  net        cross-city mean of lag1 (network signal, leave-one-out)
  nbr        distance-weighted lag1 of the 3 nearest cities (spatial graph)
  late       this city's Sep-Dec(t-1) temperature anomaly
  enso       Lima Sep-Dec(t-1) temperature anomaly (coastal Nino-1+2 proxy),
             with a per-city coefficient
Models (climatology, trend, full, full+teleconnection) compete in an
expanding-window CV over years <= X, separately per variable and window group
(annual, season, month). A (variable, group) is only claimed when the best
model beats climatology with a year-blocked t-stat >= 2. Intervals come from
empirical CV residual quantiles, so their stated confidence is the CV coverage.
"""
import argparse
import calendar
import csv
import json
import math
import random
import sys
from collections import defaultdict

COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "wind_speed_10m_max"]
TM, TX, TN, PR, WS = range(5)

WINDOWS = [("ANN", 1, 12)] + [("JF", 1, 2), ("MAM", 3, 5), ("JJA", 6, 8), ("SON", 9, 11)] + \
          [("M%02d" % m, m, m) for m in range(1, 13)]

# variable -> (daily column, aggregator)
VARS = {
    "temp_mean_c": (TM, "mean"),
    "temp_mean_max_c": (TX, "mean"),
    "temp_mean_min_c": (TN, "mean"),
    "temp_max_c": (TX, "max"),
    "temp_min_c": (TN, "min"),
    "precip_sum_mm": (PR, "sum"),
    "precip_max_day_mm": (PR, "max"),
    "rainy_days_ge_1mm": (PR, "ge1"),
    "hot_days_ge_30c": (TX, "ge30"),
    "frost_days_lt_0c": (TN, "lt0"),
    "wind_max_kmh": (WS, "max"),
}
NONNEG = {"precip_sum_mm", "precip_max_day_mm", "rainy_days_ge_1mm", "hot_days_ge_30c",
          "frost_days_lt_0c", "wind_max_kmh"}
COUNTS = {"rainy_days_ge_1mm", "hot_days_ge_30c", "frost_days_lt_0c"}

INTERVAL_COVER = 0.70     # central CV-residual coverage of interval claims
MIN_TSTAT = 2.0           # year-blocked t-stat of CV MSE gain over climatology
MIN_PAST = 5              # years of past data needed to standardize a target year
MAX_CLAIMS = 150
WIN_W = {"ANN": 1.0, "JF": 0.8, "MAM": 0.8, "JJA": 0.85, "SON": 0.8}
VAR_W = {"temp_mean_c": 1.0, "precip_sum_mm": 0.9, "warmer_than_year_x": 0.85,
         "wetter_than_year_x": 0.8, "hottest_month": 0.9, "wettest_month": 0.8,
         "temp_mean_max_c": 0.75, "temp_mean_min_c": 0.75, "temp_max_c": 0.7,
         "temp_min_c": 0.7, "hot_days_ge_30c": 0.7, "frost_days_lt_0c": 0.7,
         "rainy_days_ge_1mm": 0.65, "precip_max_day_mm": 0.6, "wind_max_kmh": 0.5}


# ---------------------------------------------------------------- data

def load(data_dir, year_x):
    days = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # loc -> year -> month -> rows
    meta = {}
    with open(f"{data_dir}/daily_history.csv", newline="") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            y = int(d[:4])
            if y > year_x:  # causality guard
                continue
            vals = []
            for c in COLS:
                v = r.get(c, "")
                vals.append(float(v) if v not in ("", None, "nan", "NaN") else None)
            loc = r["location_id"]
            meta[loc] = (float(r["lat"]), float(r["lon"]))
            days[loc][y][int(d[5:7])].append(vals)
    return days, meta


def agg(rows, col, how):
    v = [r[col] for r in rows if r[col] is not None]
    if not v or len(v) < 0.9 * len(rows):
        return None
    if how == "mean":
        return sum(v) / len(v)
    if how == "max":
        return max(v)
    if how == "min":
        return min(v)
    if how == "sum":
        return sum(v) * len(rows) / len(v)
    if how == "ge1":
        return float(sum(1 for x in v if x >= 1.0))
    if how == "ge30":
        return float(sum(1 for x in v if x >= 30.0))
    if how == "lt0":
        return float(sum(1 for x in v if x < 0.0))
    raise ValueError(how)


def window_rows(ym, m0, m1):
    rows = []
    for m in range(m0, m1 + 1):
        mr = ym.get(m)
        if not mr or len(mr) < 27:
            return None
        rows.extend(mr)
    return rows


def build_series(days, years):
    """series[var][(loc, win)] = {year: value}"""
    series = {v: {} for v in VARS}
    for loc, by_year in days.items():
        for wname, m0, m1 in WINDOWS:
            for var, (col, how) in VARS.items():
                s = {}
                for y in years:
                    rows = window_rows(by_year.get(y, {}), m0, m1)
                    if rows is not None:
                        val = agg(rows, col, how)
                        if val is not None:
                            s[y] = val
                series[var][(loc, wname)] = s
    return series


# ---------------------------------------------------------------- stats helpers

def mean(a):
    return sum(a) / len(a)


def sd(a):
    m = mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / max(1, len(a) - 1))


def quantile(sorted_a, q):
    if not sorted_a:
        return 0.0
    p = q * (len(sorted_a) - 1)
    i = int(math.floor(p))
    j = min(i + 1, len(sorted_a) - 1)
    return sorted_a[i] + (sorted_a[j] - sorted_a[i]) * (p - i)


def std_stats(s, t):
    """climatology, spread and trend extrapolation from years < t only."""
    past = sorted(y for y in s if y < t)
    if len(past) < MIN_PAST:
        return None
    v = [s[y] for y in past]
    c = mean(v)
    sdev = sd(v)
    if sdev < 1e-6:
        return None
    ym = mean(past)
    sxx = sum((y - ym) ** 2 for y in past)
    slope = sum((y - ym) * (s[y] - c) for y in past) / sxx
    trend = c + slope * (t - ym)
    return c, sdev, past, trend


def zval(s, y, st):
    if y not in s:
        return None
    return (s[y] - st[0]) / st[1]


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, h)))


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                if f:
                    for k in range(c, n + 1):
                        m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge(X, y, lam):
    """ridge with unpenalized intercept in column 0."""
    k = len(X[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi in zip(X, y):
        for i in range(k):
            xv = xi[i]
            if xv == 0.0:
                continue
            b[i] += xv * yi
            ai = a[i]
            for j in range(k):
                ai[j] += xv * xi[j]
    n = len(X)
    for i in range(1, k):
        a[i][i] += lam * n
    return solve(a, b)


def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


# ---------------------------------------------------------------- features

class Featurizer:
    def __init__(self, series, meta, years):
        self.series = series
        self.meta = meta
        self.locs = sorted(meta)
        self.years = years
        self.nbrs = {}
        for l in self.locs:
            d = sorted((haversine(meta[l], meta[o]), o) for o in self.locs if o != l)[:3]
            self.nbrs[l] = [(1.0 / max(dist, 100.0), o) for dist, o in d]
        self.enso_loc = "lima_pe" if "lima_pe" in meta else None
        self._late = {}
        self._lag1 = {}
        self._pslope = {}

    def pooled_slope(self, var, grp, t):
        """network-pooled trend in z units per year, fit on years < t across all cities/windows."""
        key = (var, grp, t)
        if key not in self._pslope:
            sxy = sxx = 0.0
            for (loc, win), s in self.series[var].items():
                if wgroup(win) != grp:
                    continue
                st = std_stats(s, t)
                if st is None:
                    continue
                c, sdev, past, _ = st
                ym = mean(past)
                for y in past:
                    sxy += (y - ym) * (s[y] - c) / sdev
                    sxx += (y - ym) ** 2
            self._pslope[key] = sxy / sxx if sxx else 0.0
        return self._pslope[key]

    def late_anom(self, loc, t):
        """Sep-Dec(t-1) temp_mean anomaly vs earlier Sep-Dec, from years < t."""
        key = (loc, t)
        if key not in self._late:
            sep = self.series["temp_mean_c"].get((loc, "SON"), {})
            dec = self.series["temp_mean_c"].get((loc, "M12"), {})
            s = {y: 0.75 * sep[y] + 0.25 * dec[y] for y in sep if y in dec}
            st = std_stats(s, t - 1)
            z = zval(s, t - 1, st) if st else None
            self._late[key] = 0.0 if z is None else max(-3.0, min(3.0, z))
        return self._late[key]

    def lag1(self, var, loc, win, t):
        key = (var, loc, win, t)
        if key not in self._lag1:
            s = self.series[var].get((loc, win), {})
            st = std_stats(s, t)
            z = zval(s, t - 1, st) if st else None
            self._lag1[key] = None if z is None else max(-3.0, min(3.0, z))
        return self._lag1[key]

    def row(self, var, loc, win, t):
        """returns (features, clim, sd, z_target or None) for target year t, or None."""
        s = self.series[var].get((loc, win), {})
        st = std_stats(s, t)
        if st is None:
            return None
        c, sdev, past, trend = st
        if var in COUNTS:
            vals = [s[y] for y in past]
            if sum(1 for v in vals if v == 0) > 0.3 * len(vals):
                return None
        zs = [max(-3.0, min(3.0, (s[y] - c) / sdev)) for y in past]
        f_trend = max(-3.0, min(3.0, (trend - c) / sdev))
        f_lag1 = zs[-1] if past[-1] == t - 1 else 0.0
        f_lag3 = mean(zs[-3:])
        others = [self.lag1(var, o, win, t) for o in self.locs if o != loc]
        others = [o for o in others if o is not None]
        f_net = mean(others) if others else 0.0
        wsum = vsum = 0.0
        for w, o in self.nbrs[loc]:
            z = self.lag1(var, o, win, t)
            if z is not None:
                wsum += w
                vsum += w * z
        f_nbr = vsum / wsum if wsum else 0.0
        f_late = self.late_anom(loc, t)
        f_enso = self.late_anom(self.enso_loc, t) if self.enso_loc else 0.0
        f_ptrend = self.pooled_slope(var, wgroup(win), t) * (t - mean(past))
        feats = [f_ptrend, f_trend, f_lag1, f_lag3, f_net, f_nbr, f_late, f_enso]
        z_t = (s[t] - c) / sdev if t in s else None
        return feats, c, sdev, z_t


MODELS = ["clim", "trend", "full", "tele"]


def design(model, feats, loc, locs):
    if model == "trend":
        return [1.0, feats[0]]
    base = [1.0] + feats[:7]
    if model == "full":
        return base
    enso = [0.0] * len(locs)
    enso[locs.index(loc)] = feats[7]
    return base + enso


LAMBDA = {"trend": 0.05, "full": 0.2, "tele": 0.4}


# ---------------------------------------------------------------- per-variable modelling

def fit_variable(var, grp, fz, years, year_x):
    keys = [k for k in fz.series[var] if wgroup(k[1]) == grp]
    samples = []  # (t, loc, win, feats, z)
    first = years[0] + MIN_PAST
    for t in range(first, year_x + 1):
        for loc, win in keys:
            r = fz.row(var, loc, win, t)
            if r and r[3] is not None:
                samples.append((t, loc, win, r[0], max(-4.0, min(4.0, r[3]))))
    tyears = sorted({s[0] for s in samples})
    eval_years = [t for t in tyears if t - tyears[0] >= 3] if tyears else []
    if len(eval_years) < 4:
        return None
    res = {m: [] for m in MODELS}           # (t, loc, win, residual, zhat)
    for t in eval_years:
        train = [s for s in samples if s[0] < t]
        test = [s for s in samples if s[0] == t]
        for m in MODELS:
            if m == "clim":
                w = None
            else:
                w = ridge([design(m, s[3], s[1], fz.locs) for s in train], [s[4] for s in train], LAMBDA[m])
            for s in test:
                zhat = 0.0 if w is None else dot(w, design(m, s[3], s[1], fz.locs))
                res[m].append((t, s[1], s[2], s[4] - zhat, zhat))
    # year-blocked comparison against climatology
    def year_mse(m):
        d = defaultdict(list)
        for t, _, _, r, _ in res[m]:
            d[t].append(r * r)
        return {t: mean(v) for t, v in d.items()}
    clim_y = year_mse("clim")
    best = None
    for m in MODELS[1:]:
        my = year_mse(m)
        diffs = [clim_y[t] - my[t] for t in clim_y]
        md = mean(diffs)
        sdd = sd(diffs) if len(diffs) > 1 else 0.0
        tstat = md / (sdd / math.sqrt(len(diffs))) if sdd > 0 else (99.0 if md > 0 else 0.0)
        skill = md / mean(list(clim_y.values()))
        cand = (skill, tstat, m)
        if best is None or skill > best[0]:
            best = cand
    skill, tstat, m = best
    w = None if m == "clim" else ridge([design(m, s[3], s[1], fz.locs) for s in samples],
                                       [s[4] for s in samples], LAMBDA[m])
    resid = sorted(r[3] for r in res[m])
    return {"model": m, "w": w, "skill": skill, "tstat": tstat, "resid": resid,
            "cv": res[m], "n_eval_years": len(eval_years)}


def wgroup(win):
    return "ANN" if win == "ANN" else ("MON" if win.startswith("M") and win != "MAM" else "SEAS")


def predict(var, fit, fz, loc, win, t):
    r = fz.row(var, loc, win, t)
    if r is None:
        return None
    feats, c, sdev, _ = r
    zhat = 0.0 if fit["w"] is None else dot(fit["w"], design(fit["model"], feats, loc, fz.locs))
    return zhat, c, sdev


def boolean_cv_accuracy(var, fit, fz):
    """CV accuracy of P(s_t > s_{t-1}) majority calls, from CV residuals."""
    hits = n = 0
    for t, loc, win, r, zhat in fit["cv"]:
        resid = fit["resid"]
        s = fz.series[var][(loc, win)]
        st = std_stats(s, t)
        if st is None or (t - 1) not in s or t not in s:
            continue
        zprev = (s[t - 1] - st[0]) / st[1]
        p = prob_gt(resid, zprev - zhat)
        if abs(p - 0.5) < 0.15:
            continue
        n += 1
        hits += int((p > 0.5) == (s[t] > s[t - 1]))
    return hits, n


def prob_gt(sorted_resid, x):
    """P(resid > x) from the empirical residual distribution."""
    lo, hi = 0, len(sorted_resid)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_resid[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return (len(sorted_resid) - lo + 0.5) / (len(sorted_resid) + 1.0)


# ---------------------------------------------------------------- claims

def window_dates(t, m0, m1):
    return "%04d-%02d-01" % (t, m0), "%04d-%02d-%02d" % (t, m1, calendar.monthrange(t, m1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, T = a.year_x, a.year_x + 1
    days, meta = load(a.data_dir, X)
    years = sorted({y for l in days for y in days[l]})
    claims = []
    if len(years) >= MIN_PAST + 4:
        claims = make_claims(days, meta, years, X, T)
    claims = claims[:MAX_CLAIMS]
    for i, c in enumerate(claims, 1):
        c["claim_id"] = "p8_%03d" % i
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)


def make_claims(days, meta, years, X, T):
    series = build_series(days, years)
    fz = Featurizer(series, meta, years)
    rng = random.Random(X)
    cands = []  # (score, loc, claim)
    fits = {}
    for var in VARS:
        for grp in ("ANN", "SEAS", "MON"):
            f = fits[var, grp] = fit_variable(var, grp, fz, years, X)
            if f:
                print("%-20s %-4s model=%-5s skill=%+.3f t=%+.2f eval_years=%d" % (
                    var, grp, f["model"], f["skill"], f["tstat"], f["n_eval_years"]), file=sys.stderr)
    wins = {w[0]: w for w in WINDOWS}

    def target(loc, win, var):
        _, m0, m1 = wins[win]
        ws, we = window_dates(T, m0, m1)
        lat, lon = meta[loc]
        return {"location_id": loc, "lat": lat, "lon": lon, "window_start": ws, "window_end": we,
                "variable": var}

    # 1. interval claims on variables with solid CV edge over climatology
    for (var, grp), fit in fits.items():
        if not fit or grp == "MON" or fit["tstat"] < MIN_TSTAT or fit["skill"] <= 0.02:
            continue
        a_lo = quantile(fit["resid"], (1 - INTERVAL_COVER) / 2)
        a_hi = quantile(fit["resid"], 1 - (1 - INTERVAL_COVER) / 2)
        for loc in fz.locs:
            for win in WIN_W:
                if wgroup(win) != grp:
                    continue
                p = predict(var, fit, fz, loc, win, T)
                if p is None:
                    continue
                zhat, c, sdev = p
                lo, hi = c + sdev * (zhat + a_lo), c + sdev * (zhat + a_hi)
                if var in NONNEG:
                    lo = max(0.0, lo)
                if var in COUNTS:
                    lo, hi = float(math.floor(lo)), float(math.ceil(hi))
                    if hi <= 0:
                        continue
                else:
                    lo, hi = math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10
                score = WIN_W[win] * VAR_W[var] * INTERVAL_COVER * (1 + min(fit["skill"], 0.5))
                cands.append((score, loc, {
                    "target": target(loc, win, var),
                    "assertion": {"kind": "interval", "low": lo, "high": hi},
                    "confidence": INTERVAL_COVER,
                    "rationale": "%s ridge (CV skill %.0f%% vs climatology, t=%.1f over %d yrs); "
                                 "clim %.1f, forecast %.1f; central %d%% CV-residual band" % (
                                     fit["model"], 100 * fit["skill"], fit["tstat"], fit["n_eval_years"],
                                     c, c + sdev * zhat, 100 * INTERVAL_COVER)}))

    # 2. warmer/wetter_than_year_x booleans (regression to the mean + trend)
    for (bvar, var), grp in [(bv, g) for bv in (("warmer_than_year_x", "temp_mean_c"),
                                                 ("wetter_than_year_x", "precip_sum_mm")) for g in ("ANN", "SEAS")]:
        fit = fits.get((var, grp))
        if not fit:
            continue
        hits, n = boolean_cv_accuracy(var, fit, fz)
        # one-sided binomial z-test vs a 50% coin flip
        if n < 20 or (hits - 0.5 * n) / math.sqrt(0.25 * n) < MIN_TSTAT:
            continue
        acc = hits / n
        for loc in fz.locs:
            for win in WIN_W:
                if wgroup(win) != grp:
                    continue
                p = predict(var, fit, fz, loc, win, T)
                s = series[var].get((loc, win), {})
                if p is None or X not in s:
                    continue
                zhat, c, sdev = p
                pg = prob_gt(fit["resid"], (s[X] - c) / sdev - zhat)
                conf = max(pg, 1 - pg)
                if conf < 0.68:
                    continue
                conf = min(conf, acc)
                score = WIN_W[win] * VAR_W[bvar] * conf
                cands.append((score, loc, {
                    "target": target(loc, win, bvar),
                    "assertion": {"kind": "boolean", "value": pg > 0.5},
                    "confidence": round(conf, 3),
                    "rationale": "Year X %s was %.1f vs forecast %.1f (clim %.1f); CV accuracy of "
                                 "confident calls %d/%d" % (var, s[X], c + sdev * zhat, c, hits, n)}))

    # 3. hottest / wettest month (annual window) via Monte Carlo over CV residuals
    for cvar, var, minp in (("hottest_month", "temp_mean_c", 0.7), ("wettest_month", "precip_sum_mm", 0.7)):
        fit = fits.get((var, "MON"))
        if not fit:
            continue
        resid = fit["resid"]
        for loc in fz.locs:
            preds = {}
            for m in range(1, 13):
                p = predict(var, fit, fz, loc, "M%02d" % m, T)
                if p:
                    preds[m] = p
            if len(preds) < 12:
                continue
            # causal check: how often was the climatological argmax month right in history?
            s_by_m = {m: series[var][(loc, "M%02d" % m)] for m in range(1, 13)}
            hist_hits = hist_n = 0
            for y in years[MIN_PAST:]:
                if all(y in s_by_m[m] for m in s_by_m):
                    clim = {m: mean([s_by_m[m][yy] for yy in s_by_m[m] if yy < y]) for m in s_by_m}
                    hist_n += 1
                    hist_hits += int(max(clim, key=clim.get) == max(s_by_m, key=lambda m: s_by_m[m][y]))
            wins_ct = defaultdict(int)
            N = 2000
            for _ in range(N):
                best = max(preds, key=lambda m: preds[m][1] + preds[m][2] * (preds[m][0] + rng.choice(resid)))
                wins_ct[best] += 1
            mbest = max(wins_ct, key=wins_ct.get)
            pm = wins_ct[mbest] / N
            if hist_n:
                pm = min(pm, (hist_hits + 1) / (hist_n + 2))
            if pm < minp:
                continue
            score = VAR_W[cvar] * pm
            cands.append((score, loc, {
                "target": target(loc, "ANN", cvar),
                "assertion": {"kind": "categorical", "value": "%02d" % mbest},
                "confidence": round(pm, 3),
                "rationale": "Monte Carlo over monthly %s forecasts: P=%.2f; climatological argmax "
                             "right %d/%d past years" % (var, wins_ct[mbest] / N, hist_hits, hist_n)}))

    # round-robin across locations so every city gets its best claims first
    by_loc = defaultdict(list)
    for score, loc, cl in cands:
        by_loc[loc].append((score, cl))
    for l in by_loc:
        by_loc[l].sort(key=lambda x: -x[0])
    out, seen, i = [], set(), 0
    while len(out) < MAX_CLAIMS and any(i < len(v) for v in by_loc.values()):
        for l in sorted(by_loc, key=lambda l: -by_loc[l][i][0] if i < len(by_loc[l]) else 0):
            if i < len(by_loc[l]) and len(out) < MAX_CLAIMS:
                cl = by_loc[l][i][1]
                tg = cl["target"]
                key = (tg["location_id"], tg["window_start"], tg["window_end"], tg["variable"])
                if key not in seen:
                    seen.add(key)
                    out.append({"claim_id": "", "target": tg, "assertion": cl["assertion"],
                                "confidence": cl["confidence"], "rationale": cl["rationale"]})
        i += 1
    return out


if __name__ == "__main__":
    main()
