#!/usr/bin/env python3
"""predict6 — The Spectral & Oscillatory Cycle Theorist.

Models each yearly target series y_t (one value per trailing year <= X) as

    y_t = level + trend * t + cycle(t) + AR(1) anomaly + ENSO(t) + noise

* trend: Theil-Sen slope, shrunk toward a physical prior (warming for temps).
* AR(1): lag-1 autocorrelation of detrended residuals, heavily shrunk, so an
  extreme year X mostly mean-reverts in X+1.
* cycle: strongest periodogram harmonic with period 2..8 years, added only
  when it passes a Fisher g-test, with amplitude shrunk.
* ENSO: ridge regression of residuals on the preceding Oct-Dec Nino3.4 anomaly
  (from indices_monthly.json, months <= X-12), used for temperatures at
  |lat| < 35 deg, where the lagged ENSO teleconnection showed out-of-sample skill.
  A late-X El Nino makes us call a warmer X+1 where mean-reversion models call
  cooling; a La Nina makes us call "le baseline" against warming extrapolators.

* Sep-Dec persistence (Round 4): one coefficient per (window, temperature
  variable), pooled across all locations and ridge-shrunk, on the detrended
  Sep-Dec anomaly of the preceding year.

Point claims: 140 (7 per location: annual temp_mean/max/min, annual precip_sum,
JJA, MAM and Jan-Feb temp_mean), values from point_value(), tuned in duels against
the real Round 5 rivals (X=1985..2009, 2010..2023 held out). Then the best-ranked
thresholds/booleans below, capped at 12 claims per location_id.

Other claims are thresholds at the naive 15/16-year climatologies that trend and
climatology rivals anchor on (so a disagreement is an exact contradiction),
plus warmer/wetter-than-year-X booleans. Candidates are ranked by
(2p - 1) x how often rivals take the opposite side of that kind of target.
Stdlib only; reads only --data-dir (dates <= X-12-31).
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


def load_enso(data_dir, year_x):
    """{year: Oct-Dec mean Nino3.4 anomaly} for years <= X, from indices_monthly.json if present."""
    path = os.path.join(data_dir, "indices_monthly.json")
    if not os.path.exists(path):
        return {}
    try:
        rows = json.load(open(path)).get("series", {}).get("nino34_anom", [])
    except (ValueError, AttributeError):
        return {}
    m = {r["date"]: r["value"] for r in rows if r.get("date", "9999") <= f"{year_x}-12" and r.get("value") is not None}
    out = {}
    for y in range(1900, year_x + 1):
        v = [m.get(f"{y}-{k}") for k in ("10", "11", "12")]
        if None not in v:
            out[y] = sum(v) / 3
    return out


ENSO_MAX_LAT = 35.0  # ENSO teleconnection used only for tropical/subtropical temperatures (backtested)


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


ENSO_LAMBDA = 2.0  # ridge pseudo-variance (degC^2 of Nino3.4) shrinking the ENSO coefficient


def enso_term(years, rc, enso, year_next):
    """Ridge regression of detrended residuals on Oct-Dec Nino3.4 of the preceding year."""
    pr = [(enso[y - 1], r) for y, r in zip(years, rc) if (y - 1) in enso]
    if not enso or len(pr) < 8 or (year_next - 1) not in enso:
        return 0.0, 0.0, [0.0] * len(rc)
    me = sum(e for e, _ in pr) / len(pr)
    beta = sum((e - me) * r for e, r in pr) / (sum((e - me) ** 2 for e, _ in pr) + ENSO_LAMBDA)
    fit = [beta * (enso[y - 1] - me) if (y - 1) in enso else 0.0 for y in years]
    return beta * (enso[year_next - 1] - me), beta, fit


def forecast(years, vals, var, year_next, enso=None):
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
    ens, beta, fit = enso_term(years, rc, enso or {}, year_next)
    mu = a + b * tn + mr + phi * rc[-1] + cyc + ens
    sd = math.sqrt(den / max(n - 2, 1)) if den > 0 else 0.0
    sigma = sd * math.sqrt(1 + 1.0 / n + (tn - sum(t) / n) ** 2 / max(sum((ti - sum(t) / n) ** 2 for ti in t), 1e-9)) * INFLATE
    return {"mu": mu, "sigma": sigma, "trend": b, "phi": phi, "period": period, "cyc": cyc,
            "enso": ens, "beta": beta, "last": vals[-1], "clim": sum(vals) / n, "n": n, "resid": rc}


# ---------------------------------------------------------------- claims
# Windows that rivals also use (exact canonical targets are needed for contradictions).
WINDOWS = {"year": (1, 1, 12, 31), "jf": (1, 1, 2, 31), "mam": (3, 1, 5, 31), "jja": (6, 1, 8, 31),
           "son": (9, 1, 11, 30), "q1": (1, 1, 3, 31), "q2": (4, 1, 6, 30), "q3": (7, 1, 9, 30),
           "q4": (10, 1, 12, 31)}
WINDOWS.update({f"m{m:02d}": (m, 1, m, 31) for m in range(1, 13)})

COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}

# Selection weight per (kind, variable, window class, side): roughly how often a rival
# claims the disjoint side of that target (rolling backtest of all ten round-1
# predictors over X=1995..2023), plus a small floor so accurate filler still ranks.
# Contradictions concentrate where warming extrapolators claim "gt baseline" and we
# expect cooling (ENSO / mean reversion), and on whole-year booleans.
def cat_weight(cat, side):
    kind, var, wc = cat
    if kind == "thr":
        if var.startswith("temp"):
            if side == "le":
                return {"year": 0.30, "season": 0.06, "month": 0.012}[wc]
            return 0.004
        return {"year": 0.08, "season": 0.01, "month": 0.002}[wc]
    return {"year": 0.08, "season": 0.02, "month": 0.004}[wc]


PMIN = 0.60
MAX_PER_LOC = 12  # CONTRACTS.md Round 3
# Point forecasts (Round 3 duels, closer to truth wins). Backtest X=2000..2023: on whole-year
# windows the model median beats 15/16-y climatology, OLS trend and persistence in ~60-64%
# of duels with gap >= min_gap; on seasons a 50/50 blend with the 10-y mean duels best.
BASE_VARS = ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "precip_sum_mm")
POINT_SPECS = [("year", "temp_mean_c"), ("year", "temp_mean_max_c"), ("year", "temp_mean_min_c"),
               ("year", "precip_sum_mm"), ("jja", "temp_mean_c"), ("mam", "temp_mean_c"),
               ("jf", "temp_mean_c")]
PRECIP_RECENT_W = 0.3
BOOL_PMIN = 0.70  # booleans near 0.6 lost more contradictions than they won in backtests


def wclass(win):
    return "year" if win == "year" else ("month" if win.startswith("m") else "season")


MIN_YEARS = 1   # X can be as early as 1980; short series still get shrunk point forecasts
SOND_MAX = 0.6  # cap on the pooled Sep-Dec persistence coefficient


def sond_anomalies(rows, X):
    """{loc: {year: detrended Sep-Dec mean temperature anomaly}} for years <= X."""
    out = {}
    for loc, lr in rows.items():
        years = sorted(y for y in {int(d[:4]) for d in lr} if y <= X)
        ys, vs = yearly_series(lr, years, (9, 1, 12, 31), "temp_mean_c")
        if len(ys) < 6:
            continue
        b = theil_sen(ys, vs)
        a = median([v - b * y for y, v in zip(ys, vs)])
        out[loc] = {y: v - (a + b * y) for y, v in zip(ys, vs)}
    return out


def fit_all(rows, meta, X, enso):
    """{(loc, win, var): (years, values, forecast)}. Temperatures get one Sep-Dec persistence
    coefficient per (window, variable), pooled across all locations (per-location fits on
    <= 16 years are noise) and shrunk toward 0, so a warm autumn in X carries into X+1."""
    F = {}
    for loc in sorted(rows):
        lr = rows[loc]
        years = sorted(y for y in {int(d[:4]) for d in lr} if y <= X)
        for win in WINDOWS:
            for var in BASE_VARS:
                ys, vs = yearly_series(lr, years, WINDOWS[win], var)
                if len(vs) < MIN_YEARS or ys[-1] != X:
                    continue
                use_enso = var in TEMP_VARS and abs(meta[loc][0]) < ENSO_MAX_LAT
                F[(loc, win, var)] = (ys, vs, forecast(ys, vs, var, X + 1, enso if use_enso else None))
    sond = sond_anomalies(rows, X)
    for win in WINDOWS:
        for var in TEMP_VARS & {"temp_mean_c", "temp_mean_max_c", "temp_mean_min_c"}:
            num = den = 0.0
            keys = [k for k in F if k[1] == win and k[2] == var and k[0] in sond]
            for k in keys:
                ys, _, f = F[k]
                for y, r in zip(ys, f["resid"]):
                    a = sond[k[0]].get(y - 1)
                    if a is not None:
                        num += a * r
                        den += a * a
            phi = max(0.0, min(SOND_MAX, num / (den + SOND_RIDGE))) if den > 0 else 0.0
            for k in keys:
                f = F[k][2]
                f["sond"] = phi * sond[k[0]].get(X, 0.0)
                f["mu"] += f["sond"]
    return F


SOND_RIDGE = 5.0


def shrunk_line(ys, vs, X, prior, k=40.0):
    """(value at X+1, residual of year X) of an OLS line whose slope is shrunk toward prior
    with weight k / (n + k)."""
    n = len(ys)
    tb, yb = sum(ys) / n, sum(vs) / n
    sxx = sum((y - tb) ** 2 for y in ys)
    bo = sum((y - tb) * (v - yb) for y, v in zip(ys, vs)) / sxx if sxx else 0.0
    b = n / (n + k) * bo + k / (n + k) * prior
    return yb + b * (X + 1 - tb), vs[-1] - (yb + b * (X - tb))


def sond_series(lr, X, var):
    years = sorted(y for y in {int(d[:4]) for d in lr} if y <= X)
    ys, vs = yearly_series(lr, years, (9, 1, 12, 31), var)
    return dict(zip(ys, vs))


def point_value(ys, vs, var, f, X, win, sond=None):
    """Point forecast for a duel (closer to truth wins). Built from three pieces, mixed per
    target by duels against the real Round 5 rivals' points (tuned on X=1985..2009, checked
    on 2010..2023 held out, verify duel rule, local truth):
    * mine: 16-y climatology clim + lam * (model - clim), lam = 0.05 n capped at 0.8 (annual)
      or 0.6 (seasons); precipitation clim + 0.3 * (last-3-year mean - clim);
    * line: 16-y OLS line with slope shrunk toward 0.03 C/y (temps) or 0 (precip), k = 40,
      plus a damped (0.2) year-X residual (predict4's form);
    * sond: 0.2 x year X's Sep-Dec residual of the same variable from that line.
    Annual temps use 0.5 mine + 0.5 (line + sond), annual temp_mean_min 0.3 mine + 0.7 (line + sond),
    JJA/Jan-Feb mine + sond, MAM mine, annual precipitation 0.7 mine + 0.3 line."""
    sel = [(y, v) for y, v in zip(ys, vs) if y >= X - 15]
    r = [v for _, v in sel]
    clim = sum(r) / len(r)
    if var == "precip_sum_mm":
        recent = r[-3:]
        mine = clim + PRECIP_RECENT_W * (sum(recent) / len(recent) - clim)
        line, res = shrunk_line([y for y, _ in sel], r, X, 0.0)
        return max(0.0, 0.7 * mine + 0.3 * (line + 0.2 * res))
    mine = clim + min(0.8 if win == "year" else 0.6, 0.05 * len(r)) * (f["mu"] - clim)
    carry = 0.0
    if sond and X in sond and len(sond) >= 3:
        sy = sorted(y for y in sond if y >= X - 15)
        carry = 0.2 * shrunk_line(sy, [sond[y] for y in sy], X, 0.03)[1]
    if win == "mam":
        return mine
    if win != "year":
        return mine + carry
    line, res = shrunk_line([y for y, _ in sel], r, X, 0.03)
    w = 0.3 if var == "temp_mean_min_c" else 0.5
    return w * mine + (1 - w) * (line + 0.2 * res + carry)


def candidates(rows, meta, X, enso):
    Y1 = X + 1
    out = []

    def target(loc, win, var):
        lat, lon = meta[loc]
        ms, ds, me, de = WINDOWS[win]
        de = min(de, calendar.monthrange(Y1, me)[1])
        return {"location_id": loc, "lat": lat, "lon": lon,
                "window_start": f"{Y1}-{ms:02d}-{ds:02d}", "window_end": f"{Y1}-{me:02d}-{de:02d}",
                "variable": var}

    F = fit_all(rows, meta, X, enso)
    for loc in sorted(rows):
        for win in WINDOWS:
            for var in BASE_VARS:
                if (loc, win, var) not in F:
                    continue
                ys, vs, f = F[(loc, win, var)]
                if var == "precip_sum_mm" and f["clim"] < 10:
                    continue  # dry-season totals: skewed, ties near zero
                cyc = f" cycle P={f['period']:.1f}y {f['cyc']:+.2f};" if f["period"] else ""
                if f.get("sond"):
                    cyc += f" Sep-Dec persistence {f['sond']:+.2f};"
                if f["enso"]:
                    cyc += f" ENSO(Nino3.4 OND {X}={enso[X]:+.2f}) {f['enso']:+.2f};"
                model = (f"trend {f['trend']:+.3f}/y, AR phi={f['phi']:.2f},{cyc} fcst {f['mu']:.2f}"
                         f"±{f['sigma']:.2f}")
                if (win, var) in POINT_SPECS:
                    pv = point_value(ys, vs, var, f, X, win, sond_series(rows[loc], X, var))
                    out.append((1.0, ("point", var, wclass(win)), {
                        "target": target(loc, win, var),
                        "assertion": {"kind": "point", "value": round(pv, 2 if var in TEMP_VARS else 1)},
                        "confidence": 0.5, "rationale": "Point = 16-y climatology shrunk toward model; " + model + "."}))
                if f["sigma"] <= 0 or len(vs) < 8:
                    continue  # probabilistic claims need a spread estimate from >= 8 years
                cdf = lambda v: ND.cdf((v - f["mu"]) / f["sigma"])
                # thresholds at the naive climatologies a trend/climatology rival anchors on
                b16 = sum(vs) / len(vs)
                r15 = [v for y, v in zip(ys, vs) if y >= X - 14]
                b15 = sum(r15) / len(r15)
                lo_b, hi_b = round(min(b15, b16), 2), round(max(b15, b16), 2)
                p_le = cdf(lo_b)
                p_gt = 1 - cdf(hi_b)
                if p_le >= p_gt:
                    a, p = {"kind": "threshold", "op": "le", "value": lo_b}, p_le
                    why = f"{var} at or below naive climatology {lo_b}: {model}."
                else:
                    a, p = {"kind": "threshold", "op": "gt", "value": hi_b}, p_gt
                    why = f"{var} above naive climatology {hi_b}: {model}."
                out.append((p, ("thr", var, wclass(win)), {"target": target(loc, win, var), "assertion": a,
                            "confidence": round(min(p, 0.9), 3), "rationale": "Spectral/AR damped-cycle model; " + why}))
                if var in ("temp_mean_c", "precip_sum_mm"):
                    bvar = "warmer_than_year_x" if var == "temp_mean_c" else "wetter_than_year_x"
                    pu = 1 - cdf(f["last"])
                    val = pu >= 0.5
                    p = max(pu, 1 - pu)
                    out.append((p, ("bool", bvar, wclass(win)), {
                        "target": target(loc, win, bvar), "assertion": {"kind": "boolean", "value": val},
                        "confidence": round(min(p, 0.9), 3),
                        "rationale": (f"Year X {var}={f['last']:.2f} vs {model}: P(X+1 > X)={pu:.2f}"
                                      f"{'; year-X extreme mean-reverts' if p > 0.7 else ''}.")}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="debug: dump every candidate (not contract-compliant)")
    args = ap.parse_args()
    X = args.year_x
    rows, meta = load(args.data_dir, X)
    cands = candidates(rows, meta, X, load_enso(args.data_dir, X))
    out = []
    if args.all:
        for p, cat, c in cands:
            out.append({"claim_id": f"p6_{len(out) + 1:04d}", **c, "_cat": list(cat), "_p": p})
    else:
        def side(c):
            a = c["assertion"]
            return a.get("op", a.get("value"))
        points = [c for p, cat, c in cands if cat[0] == "point"]
        ranked = sorted(((2 * p - 1) * (cat_weight(cat, side(c)) + 0.01), i)
                        for i, (p, cat, c) in enumerate(cands)
                        if cat[0] != "point" and p >= (BOOL_PMIN if cat[0] == "bool" else PMIN))
        per_loc, seen = defaultdict(int), set()
        for c in points + [cands[i][2] for _, i in reversed(ranked)]:
            t = c["target"]
            key = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
            if per_loc[t["location_id"]] >= MAX_PER_LOC or key in seen:
                continue  # one claim per target: the point claim wins over a threshold
            seen.add(key)
            per_loc[t["location_id"]] += 1
            out.append({"claim_id": f"p6_{len(out) + 1:03d}", **c})
            if len(out) >= MAX_CLAIMS:
                break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"predict6: wrote {len(out)} claims for X+1={X + 1} -> {args.out}")


if __name__ == "__main__":
    main()
