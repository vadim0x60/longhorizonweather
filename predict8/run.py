#!/usr/bin/env python3
"""predict8: spatial-graph + pooled ridge ensemble with time-series CV (stdlib only).

For year X, reads only <data_dir>/daily_history.csv and the optional
<data_dir>/indices_monthly.json (both strictly <= X-12-31) and emits claims about
year X+1.

Model. Per variable and window group (annual, season/quarter, month) one pooled
ridge model is fit across all locations and windows on standardized anomalies
z = (s_t - clim_<t) / sd_<t, with causal features from years < t only:
  ptrend   network-pooled trend of the variable (all cities) extrapolated to t
  trend    the series' own OLS trend extrapolation
  lag1/3   persistence of last year / mean of the last 3 years
  net      cross-city mean of lag1 (network signal, leave-one-out)
  nbr      distance-weighted lag1 of the 3 nearest cities (spatial graph)
  late     the city's own Sep-Dec(t-1) temperature anomaly
  (temperatures are modelled net of a fixed 0.025 C/yr warming-prior offset)
  per-city teleconnections: Oct-Dec(t-1) Nino3.4 (Lima Sep-Dec anomaly as a
  fallback when no index file), and in "tele2" also Oct-Dec PDO and Nov-Dec NAO.
Models (climatology, trend, full, tele, tele2) compete in expanding-window CV over
years <= X; the one with the lowest CV MSE is the forecast, and its CV residuals
(empirical, per window group) give the predictive distribution.

Claims (one per target, <= 150 total, <= 12 per location_id):
  * point forecasts (4 per city, >= 60 overall): ridge median blended with the
    series' shrunk own trend, on annual/JJA/JF temperature and annual precip/counts.
  * warmer/wetter_than_year_x booleans where P(side) >= BOOL_MIN, confidence
    capped by the CV hit rate of such calls.
  * thresholds anchored at the trailing-history baseline, only on sides that were
    calibrated in cross-year backtests (THRESH_SIDES).
  * hottest_month / wettest_month categoricals when the Monte Carlo argmax is
    dominant and the climatological argmax was historically reliable.
"""
import argparse
import calendar
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "wind_speed_10m_max"]
TM, TX, TN, PR, WS = range(5)

# name -> (start month, start day, end month, end day); end day 0 = last day of month
WINDOWS = {
    "ANN": (1, 1, 12, 0),
    "JF": (1, 1, 2, 0), "JF28": (1, 1, 2, 28), "MAM": (3, 1, 5, 0), "JJA": (6, 1, 8, 0), "SON": (9, 1, 11, 0),
    "Q1": (1, 1, 3, 0), "Q2": (4, 1, 6, 0), "Q3": (7, 1, 9, 0), "Q4": (10, 1, 12, 0),
}
for _m in range(1, 13):
    WINDOWS["M%02d" % _m] = (_m, 1, _m, 0)


def wgroup(win):
    return "ANN" if win == "ANN" else ("MON" if win[0] == "M" and win != "MAM" else "SEAS")


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
COUNTS = {"rainy_days_ge_1mm", "hot_days_ge_30c", "frost_days_lt_0c"}
TEMPS = {"temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "temp_max_c", "temp_min_c"}
WARMING_C_PER_YR = float(os.environ.get("P8_WARM", "0.025"))  # physical prior: global land warming rate since 1980 (offset, not fitted)
NONNEG = COUNTS | {"precip_sum_mm", "precip_max_day_mm", "wind_max_kmh"}

# popularity weights: how often other predictors claim this kind of target
VAR_POP = {"temp_mean_c": 1.0, "temp_mean_max_c": 0.8, "temp_mean_min_c": 0.8, "hot_days_ge_30c": 0.7,
           "frost_days_lt_0c": 0.6, "precip_sum_mm": 0.6, "rainy_days_ge_1mm": 0.4, "temp_max_c": 0.4,
           "temp_min_c": 0.3}
WIN_POP = {"ANN": 1.0, "JJA": 0.9, "JF": 0.7, "JF28": 0.7, "MAM": 0.6, "SON": 0.6,
           "Q1": 0.5, "Q2": 0.5, "Q3": 0.5, "Q4": 0.5}
WIN_POP.update({"M%02d" % m: 0.35 for m in range(1, 13)})


def anchor_targets(T):
    for var, vw in VAR_POP.items():
        for win, w in WIN_POP.items():
            if win == "JF28" and not calendar.isleap(T):
                continue  # identical to JF
            yield (var, win), vw * w


BOOL_WINDOWS = {"ANN": 1.0, "JJA": 0.9, "MAM": 0.8, "SON": 0.8, "JF": 0.8,
                "Q1": 0.7, "Q2": 0.7, "Q3": 0.7, "Q4": 0.7}
BOOL_WINDOWS.update({"M%02d" % m: 0.6 for m in range(1, 13)})

BOOL_MIN = 0.62           # minimum model probability for a boolean claim
# Anchored-threshold sides we claim, with the minimum model probability for each. Chosen from
# cross-year calibration (X=1996..2023, every candidate scored on X+1): warm-side temperature,
# hot-day and both precipitation sides were calibrated; cool-side temperature, frost-day and
# low rainy-day claims hit 5-15 points below their stated probability, so they are not made.
THRESH_SIDES = {
    "temp_mean_c": {"gt": 0.62}, "temp_mean_max_c": {"gt": 0.62}, "temp_max_c": {"gt": 0.62},
    "hot_days_ge_30c": {"gt": 0.6}, "precip_sum_mm": {"gt": 0.6, "le": 0.6},
    "rainy_days_ge_1mm": {"gt": 0.62},
}
if os.environ.get("P8_TMIN"):  # analysis hook: every side at this probability
    THRESH_SIDES = {v: {"gt": float(os.environ["P8_TMIN"]), "le": float(os.environ["P8_TMIN"]),
                        "ge": float(os.environ["P8_TMIN"]), "lt": float(os.environ["P8_TMIN"])}
                    for v in VARS}        # minimum model probability for an anchored threshold claim
MIN_PAST = 5              # years of past data needed to standardize a target year
MAX_CLAIMS = 150
MAX_PER_LOC = 12          # CONTRACTS.md round 3: at most 12 claims per location_id
# Point values blend the ridge median with the series' own trend shrunk toward the warming
# prior. In point duels against naive point forecasters (climatology, persistence, OLS trend,
# shrunk trend) over X=1996..2022, the blend beat the pure ridge median against every baseline;
# temperatures did best at 0.75, precipitation and day counts at 0.5.
PT_BLEND = {"temp": float(os.environ.get("P8_PT_BLEND", "0.75")), "other": float(os.environ.get("P8_PT_BLEND", "0.5"))}
POINTS_PER_LOC = 4        # point claims per city (>= 60 point claims required overall)
# point-forecast targets and their priority (how much we trust the model there x popularity)
POINT_TARGETS = {("temp_mean_c", "ANN"): 1.0, ("temp_mean_max_c", "ANN"): 0.9,
                 ("temp_mean_min_c", "ANN"): 0.9, ("temp_mean_c", "JJA"): 0.85,
                 ("precip_sum_mm", "ANN"): 0.8, ("temp_mean_c", "JF"): 0.6,
                 ("hot_days_ge_30c", "ANN"): 0.5, ("rainy_days_ge_1mm", "ANN"): 0.5}
MODELS = ["clim", "trend", "full", "tele", "tele2"]
LAMBDA = {"trend": 0.05, "full": 0.2, "tele": 0.4, "tele2": 0.8}
LAMBDA = {k: v * float(os.environ.get("P8_LSCALE", "1")) for k, v in LAMBDA.items()}


# ---------------------------------------------------------------- data

def load(data_dir, year_x):
    days = defaultdict(lambda: defaultdict(dict))  # loc -> year -> "MM-DD" -> vals
    meta = {}
    with open(os.path.join(data_dir, "daily_history.csv"), newline="") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            if int(d[:4]) > year_x:  # causality guard
                continue
            vals = []
            for c in COLS:
                v = r.get(c, "")
                vals.append(float(v) if v not in ("", None, "nan", "NaN") else None)
            loc = r["location_id"]
            meta[loc] = (float(r["lat"]), float(r["lon"]))
            days[loc][int(d[:4])][d[5:]] = vals
    return days, meta


def load_indices(data_dir, year_x):
    """{name: {(year, month): value}} restricted to months <= X-12."""
    p = os.path.join(data_dir, "indices_monthly.json")
    out = {}
    if not os.path.exists(p):
        return out
    try:
        with open(p) as f:
            ser = json.load(f).get("series", {})
    except (OSError, ValueError):
        return out
    for name, rows in ser.items():
        d = {}
        for r in rows:
            y, m = int(r["date"][:4]), int(r["date"][5:7])
            if y <= year_x and r.get("value") is not None:
                d[(y, m)] = float(r["value"])
        out[name] = d
    return out


def window_mds(year, win):
    m0, d0, m1, d1 = WINDOWS[win]
    d1 = d1 or calendar.monthrange(year, m1)[1]
    return "%02d-%02d" % (m0, d0), "%02d-%02d" % (m1, d1)


def window_rows(by_md, year, win):
    a, b = window_mds(year, win)
    rows = [v for md, v in by_md.items() if a <= md <= b]
    n_exp = sum(calendar.monthrange(year, m)[1] for m in range(int(a[:2]), int(b[:2]) + 1))
    if len(rows) < n_exp:
        return None
    return rows


def agg(rows, col, how):
    v = [r[col] for r in rows if r[col] is not None]
    if len(v) < len(rows):
        return None
    if how == "mean":
        return sum(v) / len(v)
    if how == "max":
        return max(v)
    if how == "min":
        return min(v)
    if how == "sum":
        return sum(v)
    if how == "ge1":
        return float(sum(1 for x in v if x >= 1.0))
    if how == "ge30":
        return float(sum(1 for x in v if x >= 30.0))
    if how == "lt0":
        return float(sum(1 for x in v if x < 0.0))
    raise ValueError(how)


def build_series(days, years):
    """series[var][(loc, win)] = {year: value}"""
    series = {v: {} for v in VARS}
    for loc, by_year in days.items():
        for win in WINDOWS:
            per_year = {y: window_rows(by_year.get(y, {}), y, win) for y in years}
            for var, (col, how) in VARS.items():
                s = {}
                for y, rows in per_year.items():
                    if rows is not None:
                        val = agg(rows, col, how)
                        if val is not None:
                            s[y] = val
                series[var][(loc, win)] = s
    return series


# ---------------------------------------------------------------- stats helpers

def mean(a):
    return sum(a) / len(a)


def sd(a):
    m = mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / max(1, len(a) - 1))


def clip(x, lim=3.0):
    return max(-lim, min(lim, x))


def quantile(sorted_a, q):
    p = q * (len(sorted_a) - 1)
    i = int(math.floor(p))
    j = min(i + 1, len(sorted_a) - 1)
    return sorted_a[i] + (sorted_a[j] - sorted_a[i]) * (p - i)


def prob_gt(sorted_resid, x):
    """P(resid > x) from the empirical residual distribution (smoothed)."""
    lo, hi = 0, len(sorted_resid)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_resid[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return (len(sorted_resid) - lo + 0.5) / (len(sorted_resid) + 1.0)


def std_stats(s, t):
    """climatology, spread and own-trend extrapolation from years < t only."""
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
    return c, sdev, past, c + slope * (t - ym)


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
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
                mr, mc = m[r], m[c]
                for k in range(c, n + 1):
                    mr[k] -= f * mc[k]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge(rows, y, k, lam):
    """rows: sparse [(idx, val)], ridge with unpenalized intercept at index 0."""
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi in zip(rows, y):
        for i, xv in xi:
            b[i] += xv * yi
            ai = a[i]
            for j, xw in xi:
                ai[j] += xv * xw
    n = len(rows)
    for i in range(1, k):
        a[i][i] += lam * n
    return solve(a, b)


def sdot(w, xi):
    return sum(w[i] * v for i, v in xi)


# ---------------------------------------------------------------- features

class Featurizer:
    def __init__(self, series, meta, indices):
        self.series = series
        self.meta = meta
        self.locs = sorted(meta)
        self.li = {l: i for i, l in enumerate(self.locs)}
        self.nbrs = {}
        for l in self.locs:
            d = sorted((haversine(meta[l], meta[o]), o) for o in self.locs if o != l)[:3]
            self.nbrs[l] = [(1.0 / max(dist, 100.0), o) for dist, o in d]
        self.idx = {name: self._index_anoms(indices.get(name), months)
                    for name, months in (("nino34_anom", (10, 11, 12)), ("pdo", (10, 11, 12)),
                                         ("nao", (11, 12)))}
        if not self.idx["nino34_anom"] and "lima_pe" in meta:
            self.idx["nino34_anom"] = None  # fallback to Lima late-year anomaly
        self._late, self._lag1, self._pslope = {}, {}, {}

    @staticmethod
    def _index_anoms(d, months):
        """year -> standardized mean of the index over `months` of that year."""
        if not d:
            return {}
        raw = {}
        for y in sorted({k[0] for k in d}):
            v = [d[(y, m)] for m in months if (y, m) in d]
            if len(v) == len(months):
                raw[y] = mean(v)
        if len(raw) < 5:
            return {}
        s = sd(list(raw.values())) or 1.0
        mu = mean(list(raw.values()))
        return {y: clip((v - mu) / s) for y, v in raw.items()}

    def index_feat(self, name, t):
        d = self.idx.get(name)
        if d is None:  # Nino fallback
            return self.late_anom("lima_pe", t)
        return d.get(t - 1, 0.0)

    def pooled_slope(self, var, grp, t):
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
        key = (loc, t)
        if key not in self._late:
            son = self.series["temp_mean_c"].get((loc, "SON"), {})
            dec = self.series["temp_mean_c"].get((loc, "M12"), {})
            s = {y: 0.75 * son[y] + 0.25 * dec[y] for y in son if y in dec}
            st = std_stats(s, t - 1)
            self._late[key] = clip((s[t - 1] - st[0]) / st[1]) if st and (t - 1) in s else 0.0
        return self._late[key]

    def lag1(self, var, loc, win, t):
        key = (var, loc, win, t)
        if key not in self._lag1:
            s = self.series[var].get((loc, win), {})
            st = std_stats(s, t)
            self._lag1[key] = clip((s[t - 1] - st[0]) / st[1]) if st and (t - 1) in s else None
        return self._lag1[key]

    def row(self, var, loc, win, t):
        """(features, clim, sd) for target year t, or None."""
        s = self.series[var].get((loc, win), {})
        st = std_stats(s, t)
        if st is None:
            return None
        c, sdev, past, trend = st
        if var in COUNTS and sum(1 for y in past if s[y] == 0) > 0.3 * len(past):
            return None
        zs = [clip((s[y] - c) / sdev) for y in past]
        f_trend = clip((trend - c) / sdev)
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
        f_ptrend = self.pooled_slope(var, wgroup(win), t) * (t - mean(past))
        feats = [f_ptrend, f_trend, f_lag1, f_lag3, f_net, f_nbr, self.late_anom(loc, t),
                 self.index_feat("nino34_anom", t), self.index_feat("pdo", t), self.index_feat("nao", t)]
        prior = WARMING_C_PER_YR * (t - mean(past)) / sdev if var in TEMPS else 0.0
        return feats, c, sdev, prior

    def design(self, model, feats, loc):
        """sparse row [(idx, val)] and dimension."""
        if model == "trend":
            return [(0, 1.0), (1, feats[0])], 2
        base = [(0, 1.0)] + [(i + 1, v) for i, v in enumerate(feats[:7])]
        if model == "full":
            return base, 8
        n = len(self.locs)
        li = self.li[loc]
        base.append((8 + li, feats[7]))
        if model == "tele":
            return base, 8 + n
        base.append((8 + n + li, feats[8]))
        base.append((8 + 2 * n + li, feats[9]))
        return base, 8 + 3 * n


# ---------------------------------------------------------------- per-variable modelling

def fit_variable(var, grp, fz, years, year_x):
    keys = [k for k in fz.series[var] if wgroup(k[1]) == grp]
    samples = []  # (t, loc, win, feats, z)
    for t in range(years[0] + MIN_PAST, year_x + 1):
        for loc, win in keys:
            s = fz.series[var][(loc, win)]
            if t not in s:
                continue
            r = fz.row(var, loc, win, t)
            if r:
                samples.append((t, loc, win, r[0], clip((s[t] - r[1]) / r[2] - r[3], 4.0), r[3]))
    tyears = sorted({s[0] for s in samples})
    eval_years = [t for t in tyears if t - tyears[0] >= 3] if tyears else []
    if len(eval_years) < 4:
        return None
    designs = {m: [fz.design(m, s[3], s[1]) for s in samples] for m in MODELS[1:]}
    cands = MODELS + ["ens"]
    res = {m: [] for m in cands}  # (t, loc, win, residual, zhat incl. prior)
    for t in eval_years:
        tr = [i for i, s in enumerate(samples) if s[0] < t]
        te = [i for i, s in enumerate(samples) if s[0] == t]
        preds = {}
        for m in MODELS[1:]:
            k = designs[m][0][1]
            w = ridge([designs[m][i][0] for i in tr], [samples[i][4] for i in tr], k, LAMBDA[m])
            preds[m] = [sdot(w, designs[m][i][0]) for i in te]
        preds["clim"] = [0.0] * len(te)
        preds["ens"] = [mean([preds[m][j] for m in MODELS[1:]]) for j in range(len(te))]
        for m in cands:
            for j, i in enumerate(te):
                s = samples[i]
                res[m].append((t, s[1], s[2], s[4] - preds[m][j], preds[m][j] + s[5]))
    mse = {m: mean([r[3] ** 2 for r in res[m]]) for m in cands}
    # The ensemble is fixed: selecting per X among models on ~5-10 CV years was noisy and,
    # in cross-year backtests (X=1996..2023), lost to the plain ensemble on every variable checked.
    m = os.environ.get("P8_FORCE_MODEL", "ens")
    clim_y, m_y = defaultdict(list), defaultdict(list)
    for r in res["clim"]:
        clim_y[r[0]].append(r[3] ** 2)
    for r in res[m]:
        m_y[r[0]].append(r[3] ** 2)
    diffs = [mean(clim_y[t]) - mean(m_y[t]) for t in clim_y]
    sdd = sd(diffs) if len(diffs) > 1 else 0.0
    tstat = mean(diffs) / (sdd / math.sqrt(len(diffs))) if sdd > 0 else 0.0
    members = []
    for mm in (MODELS[1:] if m == "ens" else ([m] if m != "clim" else [])):
        k = designs[mm][0][1]
        members.append((mm, ridge([d[0] for d in designs[mm]], [s[4] for s in samples], k, LAMBDA[mm])))
    return {"model": m, "members": members, "skill": 1 - mse[m] / mse["clim"], "tstat": tstat,
            "resid": sorted(r[3] for r in res[m]), "cv": res[m], "n_eval_years": len(eval_years)}


class Forecaster:
    def __init__(self, fz, fits, series, X):
        self.fz, self.fits, self.series, self.X = fz, fits, series, X

    def get(self, var, loc, win):
        """(zhat, clim, sd, sorted z-residuals, fit) for X+1, or None."""
        fit = self.fits.get((var, wgroup(win)))
        if not fit:
            return None
        r = self.fz.row(var, loc, win, self.X + 1)
        if r is None:
            return None
        feats, c, sdev, prior = r
        mem = fit["members"]
        zhat = prior + (mean([sdot(w, self.fz.design(m, feats, loc)[0]) for m, w in mem]) if mem else 0.0)
        return zhat, c, sdev, fit["resid"], fit

    def value_quantile(self, f, q):
        zhat, c, sdev, resid, _ = f
        return c + sdev * (zhat + quantile(resid, q))


def boolean_cv_accuracy(var, fit, series, pmin):
    hits = n = 0
    resid = fit["resid"]
    for t, loc, win, r, zhat in fit["cv"]:
        s = series[var][(loc, win)]
        st = std_stats(s, t)
        if st is None or (t - 1) not in s or t not in s:
            continue
        p = prob_gt(resid, (s[t - 1] - st[0]) / st[1] - zhat)
        if max(p, 1 - p) < pmin:
            continue
        n += 1
        hits += int((p > 0.5) == (s[t] > s[t - 1]))
    return hits, n


# ---------------------------------------------------------------- claims

def make_claims(days, meta, years, X, indices, log):
    T = X + 1
    series = build_series(days, years)
    fz = Featurizer(series, meta, indices)
    fits = {}
    for var in VARS:
        for grp in ("ANN", "SEAS", "MON"):
            f = fits[var, grp] = fit_variable(var, grp, fz, years, X)
            if f:
                log("%-18s %-4s model=%-5s skill=%+.3f t=%+.2f" % (var, grp, f["model"], f["skill"], f["tstat"]))
    fc = Forecaster(fz, fits, series, X)
    rng = random.Random(X)
    cands = []  # (priority, loc, claim)

    def target(loc, win, var):
        a, b = window_mds(T, win)
        lat, lon = meta[loc]
        return {"location_id": loc, "lat": lat, "lon": lon, "window_start": "%d-%s" % (T, a),
                "window_end": "%d-%s" % (T, b), "variable": var}

    # 0. point forecasts: the forecast median, which minimises expected absolute error and so
    #    is the value most likely to be closer to the truth than a rival's point
    point_cands = []
    for (var, win), ww in POINT_TARGETS.items():
        for loc in fz.locs:
            f = fc.get(var, loc, win)
            if f is None:
                continue
            fit = f[4]
            med = fc.value_quantile(f, 0.5)
            blend = PT_BLEND["temp" if var in TEMPS else "other"]
            if blend:
                s = series[var].get((loc, win), {})
                ys = sorted(s)
                if len(ys) >= 8:
                    m = mean([s[y] for y in ys])
                    ym = mean(ys)
                    sl = sum((y - ym) * (s[y] - m) for y in ys) / sum((y - ym) ** 2 for y in ys)
                    prior = WARMING_C_PER_YR if var in TEMPS else 0.0
                    sl = (sl * len(ys) + prior * 8.0) / (len(ys) + 8.0)
                    med = (1 - blend) * med + blend * (m + sl * (T - ym))
            if var in NONNEG:
                med = max(0.0, med)
            point_cands.append((ww * (1 + max(0.0, fit["skill"])), loc, {
                "target": target(loc, win, var),
                "assertion": {"kind": "point", "value": round(med, 2)},
                "confidence": 0.5,
                "rationale": "Ridge-ensemble median blended with shrunk own trend (clim %.2f, CV skill %+.0f%% vs climatology); "
                             "central 50%% band %.2f..%.2f" % (f[1], 100 * fit["skill"],
                                                             fc.value_quantile(f, 0.25), fc.value_quantile(f, 0.75))}))

    # 1. warmer/wetter than year X
    for bvar, var in (("warmer_than_year_x", "temp_mean_c"), ("wetter_than_year_x", "precip_sum_mm")):
        acc = {}
        for grp in ("ANN", "SEAS", "MON"):
            fit = fits.get((var, grp))
            if fit:
                h, n = boolean_cv_accuracy(var, fit, series, BOOL_MIN)
                acc[grp] = (h + 1) / (n + 2)
        for loc in fz.locs:
            for win, ww in BOOL_WINDOWS.items():
                f = fc.get(var, loc, win)
                s = series[var].get((loc, win), {})
                if f is None or X not in s or acc.get(wgroup(win), 0) < 0.55:
                    continue
                zhat, c, sdev, resid, fit = f
                pg = prob_gt(resid, (s[X] - c) / sdev - zhat)
                p = max(pg, 1 - pg)
                if p < BOOL_MIN:
                    continue
                conf = min(p, acc[wgroup(win)])
                cands.append((ww * (2 * conf - 1), loc, {
                    "target": target(loc, win, bvar),
                    "assertion": {"kind": "boolean", "value": pg > 0.5},
                    "confidence": round(conf, 3),
                    "rationale": "Year X %s %.1f vs forecast median %.1f (%s model, CV skill %+.0f%%); "
                                 "P=%.2f" % (var, s[X], fc.value_quantile(f, 0.5), fit["model"],
                                             100 * fit["skill"], p)}))

    # 2. thresholds anchored at the trailing-history baseline (the value naive and
    #    trend-only predictors anchor on), on the side our model favours, gated per side by THRESH_SIDES
    for (var, win), ww in anchor_targets(T):
        if var not in THRESH_SIDES:
            continue
        for loc in fz.locs:
            f = fc.get(var, loc, win)
            s = series[var].get((loc, win), {})
            if f is None or X not in s or len(s) < 10:
                continue
            zhat, c, sdev, resid, fit = f
            base = mean(list(s.values()))
            if var in COUNTS:
                if base < 3.0:
                    continue
                if var == "frost_days_lt_0c":
                    v = float(math.ceil(base))
                    p_hi = prob_gt(resid, (v - 0.5 - c) / sdev - zhat)  # P(count >= v)
                    hi_a, lo_a = ("ge", v), ("lt", v)
                else:
                    v = float(math.floor(base))
                    p_hi = prob_gt(resid, (v + 0.5 - c) / sdev - zhat)  # P(count > v)
                    hi_a, lo_a = ("gt", v), ("le", v)
            else:
                v = round(base, 2)
                p_hi = prob_gt(resid, (v - c) / sdev - zhat)
                hi_a, lo_a = ("gt", v), ("le", v)
            up = p_hi >= 0.5
            p = p_hi if up else 1 - p_hi
            op, val = hi_a if up else lo_a
            if p < THRESH_SIDES.get(var, {}).get(op, 2.0):
                continue
            # thresholds are where rivals' trend/persistence anchors disagree with us, so they get
            # double priority; stated confidence is shrunk 30% toward 0.5, matching the
            # cross-year hit rate of these claims (raw model P was ~8 points overconfident)
            cands.append((2 * ww * (2 * p - 1), loc, {
                "target": target(loc, win, var),
                "assertion": {"kind": "threshold", "op": op, "value": val},
                "confidence": round(0.5 + 0.7 * (p - 0.5), 3),
                "rationale": "%s model forecast median %.2f vs %d-yr baseline %.2f (year X %.2f); "
                             "P(%s %.2f)=%.2f" % (fit["model"], fc.value_quantile(f, 0.5), len(s), base,
                                                  s[X], op, val, p)}))

    # 3. hottest / wettest month over the year via Monte Carlo over CV residuals
    for cvar, var, minp in (("hottest_month", "temp_mean_c", 0.7), ("wettest_month", "precip_sum_mm", 0.7)):
        for loc in fz.locs:
            fs = {m: fc.get(var, loc, "M%02d" % m) for m in range(1, 13)}
            if any(v is None for v in fs.values()):
                continue
            resid = fs[1][3]
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
                best = max(fs, key=lambda m: fs[m][1] + fs[m][2] * (fs[m][0] + rng.choice(resid)))
                wins_ct[best] += 1
            mbest = max(wins_ct, key=wins_ct.get)
            pm = wins_ct[mbest] / N
            if hist_n:
                pm = min(pm, (hist_hits + 1) / (hist_n + 2))
            if pm < minp:
                continue
            cands.append((0.8 * pm, loc, {
                "target": target(loc, "ANN", cvar),
                "assertion": {"kind": "categorical", "value": "%02d" % mbest},
                "confidence": round(pm, 3),
                "rationale": "Monte Carlo over monthly %s forecasts: P=%.2f; climatological argmax "
                             "right %d/%d past years" % (var, wins_ct[mbest] / N, hist_hits, hist_n)}))

    if os.environ.get("P8_DUMP"):  # analysis hook: every candidate before selection
        with open(os.environ["P8_DUMP"], "w") as f:
            json.dump([cl for _, _, cl in point_cands + cands], f)

    # points first (top POINTS_PER_LOC per city), then the rest round-robin across cities,
    # never more than MAX_PER_LOC claims per city or MAX_CLAIMS in total
    out, seen, per_loc = [], set(), defaultdict(int)

    def take(cl):
        tg = cl["target"]
        key = (tg["location_id"], tg["window_start"], tg["window_end"], tg["variable"])
        if key in seen or per_loc[tg["location_id"]] >= MAX_PER_LOC or len(out) >= MAX_CLAIMS:
            return
        seen.add(key)
        per_loc[tg["location_id"]] += 1
        out.append(dict(claim_id="", **cl))

    pts = defaultdict(list)
    for pr, loc, cl in point_cands:
        pts[loc].append((pr, cl))
    for loc in sorted(pts):
        for _, cl in sorted(pts[loc], key=lambda x: -x[0])[:POINTS_PER_LOC]:
            take(cl)
    by_loc = defaultdict(list)
    for pr, loc, cl in cands:
        by_loc[loc].append((pr, cl))
    for l in by_loc:
        by_loc[l].sort(key=lambda x: -x[0])
    i = 0
    while len(out) < MAX_CLAIMS and any(i < len(v) for v in by_loc.values()):
        for l in sorted(by_loc, key=lambda l: -by_loc[l][i][0] if i < len(by_loc[l]) else 0):
            if i < len(by_loc[l]):
                take(by_loc[l][i][1])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    X = a.year_x
    log = (lambda *_: None) if a.quiet else (lambda m: print(m, file=sys.stderr))
    days, meta = load(a.data_dir, X)
    indices = load_indices(a.data_dir, X)
    years = sorted({y for l in days for y in days[l]})
    claims = make_claims(days, meta, years, X, indices, log) if len(years) >= MIN_PAST + 4 else []
    claims = claims[:MAX_CLAIMS]
    for i, c in enumerate(claims, 1):
        c["claim_id"] = "p8_%03d" % i
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)


if __name__ == "__main__":
    main()
