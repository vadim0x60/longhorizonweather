#!/usr/bin/env python3
"""predict1 — "The Neural Sequence Forecaster".

Treats each (location, calendar window, variable) as a yearly sequence and
forecasts its year-X+1 value as a predictive distribution:

  1. Monthly aggregates from `daily_history.csv` (years <= X) plus the bundled
     ERA5 monthly archive for 1964..1979 (`data/era5_monthly_pre1980.csv`).
  2. A per-variable convex ensemble of (all-years mean, last-10 mean, OLS trend,
     last-20 OLS trend), fitted at runtime by walk-forward backtest on years <= X;
     recent standardized backtest residuals give the predictive quantiles.
     Tie-heavy count/rain series use a rank-based empirical distribution instead.
  3. A Tinker-hosted Qwen3-8B LoRA (tinker_model.py; trained by tinker_train.py
     on anonymized GHCN sequences with target years <= 1969) is blended in.
  4. Claim selection (`build_claims`) uses `opponents.json`, a summary of where
     rival predictors place claims (built by build_opponents.py from their
     outputs; it holds no weather outcomes). Offensive claims contradict a
     typical rival assertion only when our probability is >= MIN_P; otherwise we
     place ~70%-likely "honeypot" claims on the targets rivals claim most, leaning
     the way our forecast departs from the naive 15-year mean.

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
CATALOG = {  # CONTRACTS.md shared benchmark coordinates (verify checks lat/lon against these)
    "london_uk": (51.5074, -0.1278), "new_york_us": (40.7128, -74.0060), "tokyo_jp": (35.6762, 139.6503),
    "sydney_au": (-33.8688, 151.2093), "cairo_eg": (30.0444, 31.2357), "mumbai_in": (19.0760, 72.8777),
    "sao_paulo_br": (-23.5505, -46.6333), "reykjavik_is": (64.1466, -21.9426), "nairobi_ke": (-1.2921, 36.8219),
    "phoenix_us": (33.4484, -112.0740), "singapore_sg": (1.3521, 103.8198), "chicago_us": (41.8781, -87.6298),
    "moscow_ru": (55.7558, 37.6173), "cape_town_za": (-33.9249, 18.4241), "lima_pe": (-12.0464, -77.0428),
    "bangkok_th": (13.7563, 100.5018), "anchorage_us": (61.2181, -149.9003), "madrid_es": (40.4168, -3.7038),
    "seoul_kr": (37.5665, 126.9780), "toronto_ca": (43.6532, -79.3832)}
MAX_PER_LOC = 12
MIN_POINTS = 70     # contract requires >= 60 point claims; keep a margin
POINTS_PER_LOC = 5
# Point targets in priority order: canonical windows rivals are told to use, ordered by
# backtested duel edge vs. 15-year climatology/trend points (temperatures strongest).
POINT_TARGETS = [("year", "temp_mean_c"), ("year", "temp_mean_max_c"), ("jja", "temp_mean_c"),
                 ("year", "temp_max_c"), ("year", "precip_sum_mm"), ("jf", "temp_mean_c"),
                 ("year", "temp_mean_min_c"), ("son", "temp_mean_c"), ("q4", "temp_mean_c"),
                 ("q1", "temp_mean_c"), ("mam", "temp_mean_c"), ("m07", "temp_mean_c")]
MIN_P = 0.7  # offensive bets only with a real edge: backtests vs. rivals' claims lose below ~0.7
DEF_W = 0.5  # weight of "honeypot" value: a likely-true claim on a target rivals often claim
WIN_PRIOR = {"year": 0.3, "jja": 0.2, "mam": 0.1, "son": 0.1, "jf": 0.05}

AGG_FIELDS = ["n_t", "st", "stx", "stn", "txmax", "tnmin", "n_p", "psum", "pmax",
              "n_w", "wmax", "hot", "frost", "rainy"]

NUM_VARS = ["temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "temp_max_c",
            "temp_min_c", "precip_sum_mm", "precip_max_day_mm", "wind_max_kmh",
            "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"]
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
def windows():
    """(name, first month, last month) of every window predict1 can claim on."""
    w = [("year", 1, 12), ("mam", 3, 5), ("jja", 6, 8), ("son", 9, 11), ("jf", 1, 2),
         ("q1", 1, 3), ("q2", 4, 6), ("q3", 7, 9), ("q4", 10, 12)]
    w += [(f"m{m:02d}", m, m) for m in range(1, 13)]
    return w


def wkind(wname):
    return "year" if wname == "year" else ("month" if wname.startswith("m") and wname != "mam" else "season")


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
    """Component forecasts for year t from strictly earlier years of `series`:
    (all-years mean, last-10 mean, all-years OLS trend, last-20 OLS trend)."""
    ys = sorted(y for y in series if y < t)
    if len(ys) < 1:
        return None
    vs = [series[y] for y in ys]
    clim = sum(vs) / len(vs)
    last = vs[-10:]
    last10 = sum(last) / len(last)
    trend = ols_predict(ys, vs, t) if len(ys) >= 8 else last10
    trend20 = ols_predict(ys[-20:], vs[-20:], t) if len(ys) >= 8 else last10
    return (clim, last10, trend, trend20)


def scale_of(series):
    vs = list(series.values())
    if len(vs) < 2:
        return None
    mu = sum(vs) / len(vs)
    return math.sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1))


NC = 4
WEIGHT_GRID = [(a / 10, b / 10, c / 10, (10 - a - b - c) / 10)
               for a in range(11) for b in range(11 - a) for c in range(11 - a - b)]


def fit_ensemble(all_series, year_x):
    """Walk-forward backtest on years <= X: per variable choose convex weights
    over the base forecasts (minimizing standardized squared error, with
    target years >= 1980 counted double) and collect standardized residuals."""
    rows = defaultdict(list)  # var -> [(comps, target, scale, var, wname)]
    gram = defaultdict(lambda: ([[0.0] * NC for _ in range(NC)], [0.0] * NC))
    for (loc, wname, var), series in all_series.items():
        s = scale_of(series)
        if not s or s < 1e-6:
            continue
        A, bv = gram[var]
        for t in sorted(series):
            if t > year_x or sum(1 for y in series if y < t) < 8:
                continue
            comps = base_forecasts(series, t)
            rows[var].append((comps, series[t], s, var, wname, t))
            wt = 2.0 if t >= 1980 else 1.0
            for i in range(NC):
                bv[i] += wt * comps[i] * series[t] / s / s
                for j in range(NC):
                    A[i][j] += wt * comps[i] * comps[j] / s / s
    model = {}
    for var, (A, bv) in gram.items():
        best, bw = None, (1.0, 0.0, 0.0, 0.0)
        for w in WEIGHT_GRID:
            err = sum(w[i] * w[j] * A[i][j] for i in range(NC) for j in range(NC)) - 2 * sum(w[i] * bv[i] for i in range(NC))
            if best is None or err < best:
                best, bw = err, w
        model[var] = {"w": bw, "n": len(rows[var])}
    resid = defaultdict(list)  # (var, wkind) -> standardized residuals
    for g, rs in rows.items():
        w = model[g]["w"]
        recent = [r for r in rs if r[5] >= year_x - 30]
        for comps, tgt, s, var, wname, _ in (recent if len(recent) >= 200 else rs):
            f = sum(wi * ci for wi, ci in zip(w, comps))
            e = (tgt - f) / s
            wk = wkind(wname)
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


class Empirical(Predictive):
    """Rank-based distribution of recent historical values, for tie-heavy series
    (dry-season rainfall, rare hot or frost days) where a location-scale model
    misprices the point mass at the minimum."""

    def __init__(self, vals):
        self.vals = sorted(vals)
        n = len(vals)
        self.m = sum(vals) / n
        self.s = max(1e-3, math.sqrt(sum((v - self.m) ** 2 for v in vals) / max(1, n - 1)))
        self.lo = None

    def cdf(self, v):
        return (bisect.bisect_right(self.vals, v) + 0.5) / (len(self.vals) + 1)

    def q(self, p):
        i = min(len(self.vals) - 1, max(0, int(round(p * (len(self.vals) - 1)))))
        return self.vals[i]


# ----------------------------------------------------------------- forecaster
def load_enso(data_dir, X):
    """{year: Oct-Dec mean Nino3.4 anomaly} for years <= X, from indices_monthly.json."""
    p = os.path.join(data_dir, "indices_monthly.json")
    out = {}
    try:
        rows = json.load(open(p))["series"]["nino34_anom"]
    except Exception:
        return out
    acc = defaultdict(list)
    for r in rows:
        y, m = int(r["date"][:4]), int(r["date"][5:7])
        if y <= X and m >= 10 and r["value"] is not None:
            acc[y].append(r["value"])
    return {y: sum(v) / len(v) for y, v in acc.items() if len(v) == 3}


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            return [0.0] * n
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


class Corrector:
    """Ridge correction of the ensemble's standardized residual for target year t,
    from causal predictors of year t-1: this location's Oct-Dec temperature residual
    (persistence), the network mean of those residuals, and Oct-Dec Nino3.4 with a
    per-location coefficient. Fitted by walk-forward rows with t <= X only."""

    RIDGE = 30.0
    RIDGE_ENSO = 60.0

    def __init__(self, F, enso):
        self.F, self.enso = F, enso
        self.q4 = {}
        for loc in F.locs:
            ser = F.series.get((loc, "q4", "temp_mean_c"), {})
            for t in ser:
                r = self._res(ser, t, "temp_mean_c")
                if r is not None:
                    self.q4[(loc, t)] = r
        by_t = defaultdict(list)
        for (loc, t), r in self.q4.items():
            by_t[t].append(r)
        self.net = {t: sum(v) / len(v) for t, v in by_t.items() if len(v) >= 3}
        self.coef, self.ecoef = {}, {}
        rows = defaultdict(list)
        erows = defaultdict(list)
        for (loc, wname, var), ser in F.series.items():
            for t in ser:
                if t > F.X:
                    continue
                e = self._res(ser, t, var)
                x = self.feats(loc, t)
                if e is None or x is None:
                    continue
                rows[(var, wname)].append((x, e))
                erows[(loc, var, wkind(wname))].append((x, e))
        for k, rs in rows.items():
            self.coef[k] = self._ridge([x[:2] for x, _ in rs], [e for _, e in rs], self.RIDGE)
        for k, rs in erows.items():
            (var, wname) = k[1], None
            # ENSO effect on what the first two features leave unexplained
            out = []
            for x, e in rs:
                out.append(([x[2]], e))
            self.ecoef[k] = self._ridge([x for x, _ in out], [e for _, e in out], self.RIDGE_ENSO)[0]

    def _res(self, ser, t, var):
        if sum(1 for y in ser if y < t) < 8:
            return None
        s = scale_of(ser)
        if not s or s < 1e-6:
            return None
        w = self.F.model.get(var, {"w": (1, 0, 0, 0)})["w"]
        return (ser[t] - sum(wi * ci for wi, ci in zip(w, base_forecasts(ser, t)))) / s

    def feats(self, loc, t):
        a, n = self.q4.get((loc, t - 1)), self.net.get(t - 1)
        if a is None or n is None:
            return None
        return (a, n, self.enso.get(t - 1, 0.0))

    @staticmethod
    def _ridge(X, y, lam):
        k = len(X[0]) if X else 0
        if not X:
            return [0.0] * k
        A = [[sum(r[i] * r[j] for r in X) + (lam if i == j else 0.0) for j in range(k)] for i in range(k)]
        b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
        return _solve(A, b)

    def shift(self, loc, wname, var, t):
        x = self.feats(loc, t)
        if x is None:
            return 0.0
        c = self.coef.get((var, wname), [0.0, 0.0])
        return c[0] * x[0] + c[1] * x[1] + self.ecoef.get((loc, var, wkind(wname)), 0.0) * x[2]


def month_window(y, m1, m2):
    return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{calendar.monthrange(y, m2)[1]:02d}"


def window_name(ws, we):
    """Inverse of month_window on month-day strings, or None."""
    for wname, m1, m2 in windows():
        a, b = month_window(2001, m1, m2)
        a2, b2 = month_window(2000, m1, m2)
        if (ws[5:], we[5:]) in ((a[5:], b[5:]), (a2[5:], b2[5:])):
            return wname
    return None


class Forecaster:
    """Predictive distributions for year X+1 built only from data dated <= X."""

    def __init__(self, year_x, data_dir, use_tinker=False):
        X = self.X = year_x
        aggs, self.locs = {}, {}
        load_pre1980(aggs, {})
        for name in ("daily_history.csv", "daily_year_x.csv"):
            p = os.path.join(data_dir, name)
            if os.path.exists(p):
                aggregate_daily(p, X, aggs, self.locs)
        if not self.locs:
            raise SystemExit("no daily data found in --data-dir")
        for loc in self.locs:
            self.locs[loc] = CATALOG.get(loc, self.locs[loc])
        years = sorted({y for (_, y, _) in aggs if y <= X})
        self.series, self.monthly = {}, {}
        for loc in self.locs:
            for wname, m1, m2 in windows():
                ser, monthly = build_series(aggs, loc, m1, m2, years)
                for v, sr in ser.items():
                    self.series[(loc, wname, v)] = sr
                self.monthly[(loc, wname)] = monthly
        self.model, self.resid = fit_ensemble(self.series, X)
        self.corr = Corrector(self, load_enso(data_dir, X))
        self.tinker_pred, self.blend = {}, 0.0
        if use_tinker and os.environ.get("TINKER_API_KEY"):
            try:
                import tinker_model  # noqa: local module
                self.blend = json.load(open(tinker_model.META)).get("blend_weight", 0.0)
                if self.blend > 0:
                    self.tinker_pred = tinker_model.predictions(self.series, X, self.locs)
            except Exception as e:  # never let the remote component break a run
                self.blend = 0.0
                print(f"[predict1] tinker component skipped: {e!r}", file=sys.stderr)
        self._cache = {}

    def dist(self, loc, wname, var):
        key = (loc, wname, var)
        if key in self._cache:
            return self._cache[key]
        d = None
        series = self.series.get(key)
        if series and self.X in series and len(series) >= 3:
            recent = [v for y, v in sorted(series.items())][-30:]
            if ((var in COUNT_VARS or var.startswith("precip")) and len(recent) >= 10
                    and sum(v <= min(recent) + 1e-9 for v in recent) >= 0.5 * len(recent)):
                d = Empirical(recent)
                self._cache[key] = d
                return d
            s = scale_of(series)
            eps = self.resid.get((var, wkind(wname))) or self.resid.get((var, "*"))
            if s and s > 1e-6 and eps and len(eps) >= 30:
                w = self.model.get(var, {"w": (1, 0, 0, 0)})["w"]
                m = sum(wi * ci for wi, ci in zip(w, base_forecasts(series, self.X + 1)))
                if not os.environ.get("P1_NOCORR"):
                    m += s * self.corr.shift(loc, wname, var, self.X + 1)
                tp = self.tinker_pred.get(key)
                if tp is not None:
                    m = (1 - self.blend) * m + self.blend * tp
                lo = 0.0 if (var in COUNT_VARS or var.startswith("precip")) else None
                d = Predictive(m, s, eps, lo)
        self._cache[key] = d
        return d

    def point(self, loc, wname, var):
        """Point forecast: median shrunk toward the 15-year climatology/trend blend that
        rivals typically use (lower backtest MAE than either alone)."""
        d = self.dist(loc, wname, var)
        ser = self.series.get((loc, wname, var))
        if d is None:
            return None
        ys = [y for y in sorted(ser) if self.X - 15 <= y <= self.X]
        vs = [ser[y] for y in ys]
        half = 0.5 * sum(vs) / len(vs) + 0.5 * ols_predict(ys, vs, self.X + 1)
        a = 0.5 if var.startswith("precip") else 0.25
        v = (1 - a) * d.q(0.5) + a * half
        return max(0.0, v) if (var in COUNT_VARS or var.startswith("precip")) else v

    def p_bool(self, loc, wname, bvar):
        """P(warmer/wetter than year X) and the year-X value."""
        var = "temp_mean_c" if bvar == "warmer_than_year_x" else "precip_sum_mm"
        d = self.dist(loc, wname, var)
        if d is None:
            return None, None
        xv = self.series[(loc, wname, var)][self.X]
        return 1 - d.cdf(xv), xv

    def cat_probs(self, loc, wname, cvar):
        idx = 0 if cvar == "hottest_month" else 1
        m1, m2 = [(a, b) for n, a, b in windows() if n == wname][0]
        counts, tot = defaultdict(float), 0.0
        for y, mm in self.monthly.get((loc, wname), {}).items():
            vals = {k: v for k, v in mm[idx].items() if v is not None}
            if len(vals) != m2 - m1 + 1:
                continue
            wy = 1.0 + (y >= self.X - 14)  # weight recent decades more
            counts[max(vals, key=vals.get)] += wy
            tot += wy
        if tot < 5:
            return None
        n = m2 - m1 + 1
        return {f"{m:02d}": (counts[m] + 0.5) / (tot + 0.5 * n) for m in range(m1, m2 + 1)}


# ------------------------------------------------------------- set algebra
INF = float("inf")


def as_set(var, a):
    """(lo, lo_closed, hi, hi_closed), mirroring verify's integer handling for counts."""
    if a["kind"] == "interval":
        lo, lc, hi, hc = a["low"], True, a["high"], True
    else:
        v = a["value"]
        lo, lc, hi, hc = {"gt": (v, False, INF, False), "ge": (v, True, INF, False),
                          "lt": (-INF, False, v, False), "le": (-INF, False, v, True)}[a["op"]]
    if var in COUNT_VARS:
        if lo != -INF:
            lo = math.ceil(lo) if lc else math.floor(lo) + 1
            lc = True
        if hi != INF:
            hi = math.floor(hi) if hc else math.ceil(hi) - 1
            hc = True
    return lo, lc, hi, hc


def disjoint(var, a, b):
    alo, alc, ahi, ahc = as_set(var, a)
    blo, blc, bhi, bhc = as_set(var, b)
    lo, lc = (alo, alc) if (alo > blo or (alo == blo and not alc)) else (blo, blc)
    hi, hc = (ahi, ahc) if (ahi < bhi or (ahi == bhi and not ahc)) else (bhi, bhc)
    return lo > hi or (lo == hi and not (lc and hc))


def prob(var, d, a):
    lo, _, hi, _ = as_set(var, a)
    if var in COUNT_VARS:
        return max(0.0, d.cdfc(hi + 0.5) - d.cdfc(lo - 0.5))
    return max(0.0, d.cdfc(hi) - d.cdfc(lo))


def _cdfc(self, v):
    if v == INF:
        return 1.0
    if v == -INF:
        return 0.0
    if self.lo is not None and v < self.lo:
        return 0.0
    return self.cdf(v)


Predictive.cdfc = _cdfc


# ---------------------------------------------------------- opponent model
OPP_PATH = os.path.join(HERE, "opponents.json")


def load_opponents():
    if not os.path.exists(OPP_PATH):
        return None
    opp = json.load(open(OPP_PATH))
    locs = {k.rsplit("|", 1)[1] for k in opp.get("rate_loc", {})}
    opp["sample_locs_has"] = locs.__contains__
    return opp


def rival_numeric(opp, var, wname, d, loc):
    """Expected rival assertions on this target: [(weight, assertion)]. The
    expected number of rival claims is location-specific where the location was
    in the opponent sample, else the per-(variable, window) average."""
    rate = opp["rate_loc"].get(f"{var}|{wname}|{loc}", 0.0 if opp["sample_locs_has"](loc)
                               else opp["rate"].get(f"{var}|{wname}", 0.0))
    samples = opp["numeric"].get(f"{var}|{wkind(wname)}") or []
    if rate <= 0 or not samples:
        return []
    step = max(1, len(samples) // 60)
    samples = samples[::step]
    w = rate / len(samples)
    out = []
    for k, op, z1, z2 in samples:
        if k == "threshold":
            out.append((w, {"kind": "threshold", "op": op, "value": round(d.m + z1 * d.s, 2)}))
        else:
            out.append((w, {"kind": "interval", "low": round(d.m + z1 * d.s, 2), "high": round(d.m + z2 * d.s, 2)}))
    return out


def best_numeric(var, d, rivals):
    """Assertion maximizing sum_r w_r * disjoint * (P(mine) - P(rival))."""
    pr = [(w, a, prob(var, d, a)) for w, a in rivals]
    menu = []
    bounds = {round(d.q(q), 2) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    for _, a, _ in pr:
        if a["kind"] == "threshold":
            bounds.add(a["value"])
        else:
            bounds.update((a["low"], a["high"]))
    for b in bounds:
        menu += [{"kind": "threshold", "op": "le", "value": b}, {"kind": "threshold", "op": "ge", "value": b}]
    for ql, qh in ((0.25, 0.75), (0.15, 0.85), (0.3, 0.7)):
        lo, hi = round(d.q(ql), 2), round(d.q(qh), 2)
        if hi > lo:
            menu.append({"kind": "interval", "low": lo, "high": hi})
    best = None
    for a in menu:
        p = prob(var, d, a)
        if p < MIN_P:
            continue
        ev = sum(w * (p - q) for w, ra, q in pr if disjoint(var, a, ra))
        if best is None or (ev, p) > (best[0], best[2]):
            best = (ev, a, p)
    return best or (0.0, {"kind": "threshold", "op": "ge", "value": round(d.q(0.2), 2)}, 0.8)


def popularity(opp, var, wname, loc):
    """Expected number of rival claims on this target (plus a small window prior)."""
    prior = WIN_PRIOR.get(wname, 0.02)
    if not opp:
        return prior
    if opp["sample_locs_has"](loc):
        return prior + opp["rate_loc"].get(f"{var}|{wname}|{loc}", 0.0)
    return prior + opp["rate"].get(f"{var}|{wname}", 0.0)


# --------------------------------------------------------------------- claims
def fmt(v):
    return round(float(v), 2)


def build_claims(F, opp):
    X, Y = F.X, F.X + 1
    cands = []
    for loc, (lat, lon) in sorted(F.locs.items()):
        for wname, m1, m2 in windows():
            ws, we = month_window(Y, m1, m2)
            tgt = {"location_id": loc, "lat": lat, "lon": lon, "window_start": ws, "window_end": we}
            for var in NUM_VARS:
                d = F.dist(loc, wname, var)
                if d is None:
                    continue
                riv = rival_numeric(opp, var, wname, d, loc) if opp else []
                ev, a, p = best_numeric(var, d, riv) if riv else (0.0, None, 0.0)
                # defensive claim: ~70% likely, leaning the way our forecast departs from the
                # naive 15-year mean, so sharper opposite bets by rivals are disjoint from it
                ser = F.series[(loc, wname, var)]
                rv = [v for y, v in ser.items() if y > X - 15]
                up = d.q(0.5) >= sum(rv) / len(rv)
                da = ({"kind": "threshold", "op": "ge", "value": fmt(d.q(0.3))} if up
                      else {"kind": "threshold", "op": "le", "value": fmt(d.q(0.7))})
                dp = prob(var, d, da)
                dev = DEF_W * popularity(opp, var, wname, loc) * (2 * dp - 1)
                if a is None or dev > ev:
                    ev, a, p = dev, da, dp
                cands.append((ev, p, dict(tgt, variable=var), a,
                              f"mean {d.m:.2f} sd {d.s:.2f}; P={p:.2f}; exp. contradiction gain {ev:.3f}"))
            for bvar in ("warmer_than_year_x", "wetter_than_year_x"):
                p, xv = F.p_bool(loc, wname, bvar)
                if p is None:
                    continue
                if not opp:
                    rt = rf = 0.0
                elif opp["sample_locs_has"](loc):
                    rt, rf = opp["bool_loc"].get(f"{bvar}|{wname}|{loc}", [0.0, 0.0])
                else:
                    rt, rf = opp["bool"].get(f"{bvar}|{wname}", [0.0] * 20)[min(9, int(p * 10)) * 2:][:2] or (0.0, 0.0)
                ev_t = rf * (2 * p - 1) if p >= MIN_P else 0.0
                ev_f = rt * (1 - 2 * p) if 1 - p >= MIN_P else 0.0
                val, ev, pc = (True, ev_t, p) if (ev_t, p) >= (ev_f, 1 - p) else (False, ev_f, 1 - p)
                if pc >= MIN_P:
                    ev = max(ev, DEF_W * (rt + rf + WIN_PRIOR.get(wname, 0.02)) * (2 * pc - 1))
                cands.append((ev, pc, dict(tgt, variable=bvar), {"kind": "boolean", "value": val},
                              f"P(true)={p:.2f} vs year-X value {xv:.2f}; exp. gain {ev:.3f}"))
            if m2 > m1:
                for cvar in ("hottest_month", "wettest_month"):
                    cp = F.cat_probs(loc, wname, cvar)
                    if not cp:
                        continue
                    ranked = sorted(cp, key=cp.get, reverse=True)
                    rates = opp["cat"].get(f"{cvar}|{wkind(wname)}", []) if opp else []
                    best = None
                    for c in ranked:
                        if cp[c] < 0.5:
                            continue
                        ev = sum(r * (cp[c] - cp[o]) for r, o in zip(rates, ranked) if o != c)
                        if best is None or ev > best[0]:
                            best = (ev, c)
                    if best is None:
                        continue
                    best = (max(best[0], DEF_W * (sum(rates) + WIN_PRIOR.get(wname, 0.02)) * (2 * cp[best[1]] - 1)), best[1])
                    cands.append((best[0], cp[best[1]], dict(tgt, variable=cvar),
                                  {"kind": "categorical", "value": best[1]},
                                  f"P={cp[best[1]]:.2f} ({cvar}); exp. gain {best[0]:.3f}"))
    # 1) point forecasts on the canonical targets rivals are most likely to duel on
    by_loc = defaultdict(list)
    point_keys = set()
    n_pts = min(MAX_PER_LOC, max(POINTS_PER_LOC, -(-MIN_POINTS // max(1, len(F.locs)))))
    for loc, (lat, lon) in sorted(F.locs.items()):
        for wname, var in POINT_TARGETS:
            if len(by_loc[loc]) >= n_pts:
                break
            d = F.dist(loc, wname, var)
            pv = F.point(loc, wname, var)
            if d is None or pv is None:
                continue
            m1, m2 = [(a, b) for n, a, b in windows() if n == wname][0]
            ws, we = month_window(Y, m1, m2)
            tgt = {"location_id": loc, "lat": lat, "lon": lon, "window_start": ws, "window_end": we, "variable": var}
            by_loc[loc].append((tgt, {"kind": "point", "value": round(pv, 2)}, 0.5,
                                f"point: 0.75*median + 0.25*15-yr blend (precip 0.5); sd {d.s:.2f}"))
            point_keys.add((loc, wname, var))
    # 2) remaining slots: honeypot / offensive claims, ranked by expected contradiction gain
    cands.sort(key=lambda c: (-round(c[0], 4), -c[1]))
    n_total = sum(len(v) for v in by_loc.values())
    for ev, p, tgt, a, why in cands:
        if n_total >= MAX_CLAIMS:
            break
        loc = tgt["location_id"]
        key = (loc, window_name(tgt["window_start"], tgt["window_end"]), tgt["variable"])
        if key in point_keys or len(by_loc[loc]) >= MAX_PER_LOC or (ev <= 1e-4 and p < 0.75):
            continue
        by_loc[loc].append((tgt, a, p, why))
        n_total += 1
    claims = []
    for loc in sorted(by_loc):
        for tgt, a, p, why in by_loc[loc]:
            claims.append({"claim_id": f"p1_{len(claims) + 1:03d}", "target": tgt, "assertion": a,
                           "confidence": round(min(0.99, max(0.01, p)), 3), "rationale": why[:300]})
    return claims, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tinker", action="store_true", help="skip the Tinker LoRA component")
    ap.add_argument("--no-opponents", action="store_true", help="ignore the opponent model")
    args = ap.parse_args()
    F = Forecaster(args.year_x, args.data_dir, use_tinker=not args.no_tinker)
    opp = None if args.no_opponents else load_opponents()
    claims, cands = build_claims(F, opp)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(claims, f, indent=1)
    gain = sum(c[0] for c in cands[:len(claims)])
    print(f"[predict1] X={F.X}: {len(claims)} claims from {len(cands)} candidates; selection score {gain:.2f}; "
          f"ensemble={ {g: v['w'] for g, v in F.model.items()} }; tinker_pred={len(F.tinker_pred)} "
          f"blend={F.blend}; opponents={'yes' if opp else 'no'}", file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
