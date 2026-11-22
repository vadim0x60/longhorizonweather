"""Analog-ensemble core for predict2 ("Regime-Analogue" predictor).

Every past year y (<= X) is an analog for year X+1. Its daily series is
warmed by the local temperature trend times (X+1 - y), plus a damped share of
year X's anomaly (persistence) and an optional regime shift from the Tinker
model, then jittered by a small kernel. Every window statistic is computed on
each shifted analog, which yields a weighted ensemble per target.
Only data dated <= X-12-31 is ever read by the callers of this module.
"""
import calendar
import csv
import gzip
import math
import os
from collections import defaultdict
from datetime import date
from statistics import NormalDist

FIELDS = ("temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max")
TEMP_VARS = ("temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c")
NUM_VARS = TEMP_VARS + ("precip_sum_mm", "precip_max_day_mm", "wind_max_kmh",
                        "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm")
COUNT_VARS = ("hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm")
JITTER_Z = (-1.2816, -0.5244, 0.0, 0.5244, 1.2816)  # quintile midpoints of N(0,1)


def _f(v):
    if v is None or v == "" or v.lower() == "nan":
        return None
    return float(v)


def load_csv(path, max_year):
    """Return {loc: {"lat":, "lon":, "days": {year: [(month, day, tmean, tmax, tmin, pr, wind)]}}}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    locs = {}
    with opener(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            d = r["date"]
            y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
            if y > max_year:
                continue  # temporal causality guard
            loc = locs.setdefault(r["location_id"], {"lat": float(r["lat"]), "lon": float(r["lon"]), "days": defaultdict(dict)})
            loc["days"][y][(m, dd)] = tuple(_f(r[k]) for k in FIELDS)
    for loc in locs.values():
        loc["days"] = {y: [(m, d) + v for (m, d), v in sorted(dd.items())] for y, dd in loc["days"].items()}
    return locs


def merge(*sources):
    out = {}
    for src in sources:
        for lid, loc in src.items():
            o = out.setdefault(lid, {"lat": loc["lat"], "lon": loc["lon"], "days": {}})
            o["days"].update(loc["days"])
    return out


def complete_years(days, min_days=330):
    return sorted(y for y, rows in days.items() if sum(1 for r in rows if r[2] is not None) >= min_days)


def annual_mean(rows):
    v = [r[2] for r in rows if r[2] is not None]
    return sum(v) / len(v)


def window_md(ws, we):
    return (int(ws[5:7]), int(ws[8:10])), (int(we[5:7]), int(we[8:10]))


def window_stats(rows, start_md, end_md, shift, pscale=1.0):
    """All numeric stats plus monthly maps for one analog year and temperature shift."""
    sel = [r for r in rows if start_md <= (r[0], r[1]) <= end_md]
    tm = [r[2] for r in sel if r[2] is not None]
    tx = [r[3] for r in sel if r[3] is not None]
    tn = [r[4] for r in sel if r[4] is not None]
    pr = [r[5] for r in sel if r[5] is not None]
    wd = [r[6] for r in sel if r[6] is not None]
    if len(tm) < 0.8 * max(1, len(sel)) or not pr or not sel:
        return None
    s = {
        "temp_mean_c": sum(tm) / len(tm) + shift,
        "temp_max_c": max(tx) + shift,
        "temp_min_c": min(tn) + shift,
        "temp_mean_max_c": sum(tx) / len(tx) + shift,
        "temp_mean_min_c": sum(tn) / len(tn) + shift,
        "precip_sum_mm": pscale * sum(pr) * len(sel) / len(pr),
        "precip_max_day_mm": pscale * max(pr),
        "wind_max_kmh": max(wd) if wd else None,
        "hot_days_ge_30c": sum(1 for v in tx if v + shift >= 30.0),
        "frost_days_lt_0c": sum(1 for v in tn if v + shift < 0.0),
        "rainy_days_ge_1mm": sum(1 for v in pr if v >= 1.0),
    }
    mt, mp = defaultdict(list), defaultdict(float)
    for r in sel:
        if r[2] is not None:
            mt[r[0]].append(r[2])
        if r[5] is not None:
            mp[r[0]] += r[5]
    if len(mt) >= 2:
        s["hottest_month"] = "%02d" % max(mt, key=lambda m: sum(mt[m]) / len(mt[m]))
        s["wettest_month"] = "%02d" % max(mp, key=lambda m: mp[m])
    return s


def theil_sen(xs, ys):
    sl = sorted((ys[j] - ys[i]) / (xs[j] - xs[i]) for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i])
    return sl[len(sl) // 2] if sl else 0.0


class Params:
    tau = 25.0           # recency e-folding (years) for analog weights
    trend_prior = float(os.environ.get("P2_TP", "0.03"))  # degC/yr prior for local warming
    trend_years = 35     # trailing years used for the local trend
    trend_shrink = 30.0  # pseudo-years pulling the local slope to the prior
    phi = 0.15           # persistence of year-X annual anomaly
    jitter = 0.5         # degC kernel sd on top of analog spread
    precip_jitter = 0.12 # log-sd multiplicative precip kernel
    min_years_for_x = 1
    max_age = int(os.environ.get("P2_MAXAGE", "999"))  # oldest analog year allowed, in years before X


def build_ensemble(loc, year_x, windows, params=Params, regime_shift=0.0):
    """Return ({window: [(weight, stats)]}, info) for X+1 using analog years <= X."""
    days = loc["days"]
    yrs = [y for y in complete_years(days) if year_x - params.max_age <= y <= year_x]
    ann = {y: annual_mean(days[y]) for y in yrs}
    tyrs = [y for y in yrs if y > year_x - params.trend_years]
    if len(tyrs) >= 8:
        local = theil_sen(tyrs, [ann[y] for y in tyrs])
        n = len(tyrs)
        slope = (local * n + params.trend_prior * params.trend_shrink) / (n + params.trend_shrink)
    else:
        slope = params.trend_prior
    # trend line anchored on the recent decade (robust to the slope)
    rec = [y for y in yrs if y > year_x - 10] or yrs
    anchor_y = sum(rec) / len(rec)
    anchor_t = sum(ann[y] for y in rec) / len(rec)
    resid_x = (ann[year_x] - (anchor_t + slope * (year_x - anchor_y))) if year_x in ann else 0.0
    target_level = anchor_t + slope * (year_x + 1 - anchor_y) + params.phi * resid_x + regime_shift
    ens = {w: [] for w in windows}
    zs, jit, pjit = JITTER_Z, params.jitter, params.precip_jitter
    if len(yrs) < 8:
        # few analogs (early years): widen the kernel by typical interannual spread
        zs = tuple(NormalDist().inv_cdf((i + 0.5) / 15) for i in range(15))
        jit, pjit = math.hypot(jit, 0.6), math.hypot(pjit, 0.25)
    for y in yrs:
        # shift that moves analog year y onto the X+1 trend level, keeping its own anomaly
        base_shift = target_level - (anchor_t + slope * (y - anchor_y))
        wgt = math.exp(-(year_x - y) / params.tau)
        for z in zs:
            sh = base_shift + z * jit
            for w in windows:
                smd, emd = window_md(*w)
                st = window_stats(days[y], smd, emd, sh, math.exp(z * pjit))
                if st is not None:
                    st["year"] = y
                    ens[w].append((wgt, st))
    info = {"slope": slope, "resid_x": resid_x, "target_level": target_level, "n_years": len(yrs)}
    return ens, info


def wquantile(pairs, q):
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * tot:
            return v
    return pairs[-1][0]


def wprob(pairs, pred):
    tot = sum(w for _, w in pairs)
    return sum(w for v, w in pairs if pred(v)) / tot if tot else 0.5


def year_x_actual(loc, year_x, window, var):
    """Value of a window stat in year X itself (Feb 29 clipped when X is not leap)."""
    rows = loc["days"].get(year_x)
    if not rows:
        return None
    (sm, sd), (em, ed) = window_md(*window)
    if not calendar.isleap(year_x):
        sd = min(sd, 28) if sm == 2 else sd
        ed = min(ed, 28) if em == 2 else ed
    st = window_stats(rows, (sm, sd), (em, ed), 0.0)
    return None if st is None else st.get(var)


def default_windows(year_next):
    yn = year_next
    w = [("%d-01-01" % yn, "%d-12-31" % yn),
         ("%d-03-01" % yn, "%d-05-31" % yn), ("%d-06-01" % yn, "%d-08-31" % yn),
         ("%d-09-01" % yn, "%d-11-30" % yn)]
    for m in range(1, 13):
        w.append((date(yn, m, 1).isoformat(), date(yn, m, calendar.monthrange(yn, m)[1]).isoformat()))
    return w
