#!/usr/bin/env python3
"""predict2: Regime-Analogue predictor.

Builds a trend-adjusted analog ensemble for year X+1 from data dated <= X-12-31
(core.py), optionally nudged by a Tinker LoRA "regime" model (regime.py), and
emits <=150 claims (see CONTRACTS.md).

Point claims (Round 3+) come first: up to 6 per location on the annual and seasonal
targets rivals duel on, valued clim16 + lambda * (ensemble median - clim16), optionally
blended with the Tinker anomaly-sequence LoRA (seqmodel.py). Other points are only
scored when another predictor claims the same target with a disjoint assertion, so claims sit on "focal" targets: standard calendar windows
and thresholds at the naive 15/16-year climatological mean, the value most
climatology-anchored predictors also use. The side of each threshold (and each
boolean) is whichever the ensemble finds more likely; candidates are ranked by
that probability times how often the claim type has historically been
contested (TYPE_W, fitted on pre-1995 replays).
"""
import argparse
import calendar
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

MAX_CLAIMS = 150
PRE1980 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "daily_pre1980.csv.gz")
MAX_PER_LOC = 12
POINTS_FIRST = 6  # point claims reserved per location before the shared ranking (>= 60 overall)
MIN_GAP = {"temp_mean_c": 0.25, "temp_mean_max_c": 0.25, "temp_mean_min_c": 0.25, "temp_max_c": 0.5,
           "temp_min_c": 0.5, "precip_sum_mm": 20.0, "precip_max_day_mm": 3.0, "wind_max_kmh": 2.0,
           "hot_days_ge_30c": 2.0, "frost_days_lt_0c": 2.0, "rainy_days_ge_1mm": 2.0}
# Point targets in priority order: (window, variable, duel value). Rivals' point claims sit
# mostly on annual windows; in a replay of X=1981..1994 against the Round 3 rival code these
# targets were net positive (annual temp mean/max/min, precip, hot days).
POINTS = [("year", "temp_mean_c", 1.0), ("year", "precip_sum_mm", 1.0), ("year", "temp_mean_min_c", 0.95),
          ("year", "temp_mean_max_c", 0.95), ("year", "hot_days_ge_30c", 0.9), ("jja", "temp_mean_c", 0.8),
          ("jja", "precip_sum_mm", 0.7), ("mam", "temp_mean_c", 0.6), ("son", "temp_mean_c", 0.6),
          ("jf", "temp_mean_c", 0.5)]
# Point value = clim16 + LAMBDA * (ensemble median - clim16). Rivals shrink toward the trailing
# 16-year mean too; in point duels against them these weights beat both the raw median and clim16.
LAMBDA = {"precip_sum_mm": float(os.environ.get("P2_LP", "0.5")), "hot_days_ge_30c": 1.0}
LAMBDA_TEMP = float(os.environ.get("P2_LT", "0.4"))  # 0.25 -> 0.4: replay +55 -> +63, X=1995-2023 MAE 0.533 -> 0.519
POINT_BASE = os.environ.get("P2_BASE", "focal")
PRECIP_RECENT = ("precip_sum_mm",)
PRECIP_BASE = os.environ.get("P2_PB", "clim16")  # short post-1980 means mislead early years; replay -14 -> +10
PRECIP_DRIFT = float(os.environ.get("P2_PD", "0.3"))  # + drift * (last3 - clim16), predict6-style
TREND_ADJ = float(os.environ.get("P2_TA", "1.0"))  # late-years replay (7 stations, X=1995-2023): -159 -> -32 with TP 0.03
SOND_CARRY = float(os.environ.get("P2_SC", "0.1"))  # + carry * year-X Sep-Dec anomaly, predict4-style; 0.3 lost badly in 1981-94
PRECIP_MIN_YEAR = int(os.environ.get("P2_PMY", "0"))
EPS = 0.01  # threshold margin so "le m" / "gt m" stay disjoint from rivals' gt/le at m
TEMP3 = ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c")
# relative value of a claim type: how likely it is to be contested and won
TYPE_W = {
    ("precip_sum_mm", "le"): 0.3, ("precip_sum_mm", "gt"): 0.5,
    ("temp", "le"): 0.8, ("temp", "gt"): 0.5,
    ("hot_days_ge_30c", "le"): 0.4, ("hot_days_ge_30c", "gt"): 0.5,
    ("frost_days_lt_0c", "ge"): 0.8, ("frost_days_lt_0c", "lt"): 0.5,
    ("rainy_days_ge_1mm", "le"): 0.6, ("rainy_days_ge_1mm", "gt"): 0.5,
    ("warmer_than_year_x", True): 0.9, ("warmer_than_year_x", False): 0.6,
    ("wetter_than_year_x", True): 0.7, ("wetter_than_year_x", False): 1.0,
}
DRY_MM = 5.0
DRY_MIN_P = float(os.environ.get("P2_DRY_MIN_P", "0.8"))
MIN_P = float(os.environ.get("P2_MIN_P", "0.64"))  # contested claims below this lost more than they won


def load(data_dir, year_x):
    srcs = []
    for name in ("daily_history.csv", "daily_year_x.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            srcs.append(core.load_csv(p, year_x))
    if not srcs:
        raise SystemExit("no daily_*.csv in %s" % data_dir)
    locs = core.merge(*srcs)
    if os.path.exists(PRE1980) and os.environ.get("P2_PRE1980", "1") == "1":
        # bundled ERA5 daily archive, 1964..1979 only: before every scored X, so leak-free
        pre = core.load_csv(PRE1980, 1979)
        for lid, loc in locs.items():
            for y, rows in pre.get(lid, {}).get("days", {}).items():
                loc["days"].setdefault(y, rows)
    return locs


def clim16(loc, X, w, var):
    """Mean of a window stat over X-15..X (uses the bundled pre-1980 years when X < 1995)."""
    md = core.window_md(*w)
    v = []
    for y in range(X - 15, X + 1):
        st = core.window_stats(loc["days"][y], *md, 0.0) if y in loc["days"] else None
        if st and st.get(var) is not None:
            v.append(st[var])
    return sum(v) / len(v) if len(v) >= 5 else None


def windows(Y):
    """(name, start, end) for X+1: calendar windows plus Feb-28 variants used by some rivals."""
    def mw(m1, m2):
        return date(Y, m1, 1).isoformat(), date(Y, m2, calendar.monthrange(Y, m2)[1]).isoformat()
    w = [("year",) + mw(1, 12), ("mam",) + mw(3, 5), ("jja",) + mw(6, 8), ("son",) + mw(9, 11),
         ("jf",) + mw(1, 2), ("q1",) + mw(1, 3), ("q2",) + mw(4, 6), ("q3",) + mw(7, 9), ("q4",) + mw(10, 12)]
    w += [("m%02d" % m,) + mw(m, m) for m in range(1, 13)]
    if calendar.isleap(Y):
        w += [("jf28", "%d-01-01" % Y, "%d-02-28" % Y), ("m02_28", "%d-02-01" % Y, "%d-02-28" % Y)]
    return w


WIN_W = {"year": 1.0, "jja": 0.9, "mam": 0.8, "son": 0.8, "jf": 0.7, "jf28": 0.7}


def focal_means(loc, X, w, var):
    """Naive climatological means of a window stat: years max(1980,X-15)..X and max(1980,X-14)..X."""
    md = core.window_md(w[1], w[2])
    vals = {}
    for y, rows in loc["days"].items():
        if max(1980, X - 15) <= y <= X:
            st = core.window_stats(rows, *md, 0.0)
            if st and st.get(var) is not None:
                vals[y] = st[var]
    out = []
    for y0 in (max(1980, X - 15), max(1980, X - 14)):
        v = [x for y, x in vals.items() if y >= y0]
        if len(v) >= 5 and X in vals:
            out.append(sum(v) / len(v))
    return out


def seq_history(loc, X, w, var):
    """The 16 yearly values X-15..X of a window stat (input of the sequence LoRA), or None."""
    md = core.window_md(*w)
    out = []
    for y in range(X - 15, X + 1):
        st = core.window_stats(loc["days"][y], *md, 0.0) if y in loc["days"] else None
        if not st or st.get(var) is None:
            return None
        out.append(st[var])
    return out


def _log_seq_inference(n_points):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinker_budget.json")
    try:
        with open(path) as fh:
            b = json.load(fh)
        b["inference_runs"] = b.get("inference_runs", 0) + 1
        b["inference_usd_est"] = round(b.get("inference_usd_est", 0.0) + n_points * 9 * 170 * 0.195 / 1e6, 4)
        b["total_usd_est"] = round(b["training_usd_est"] + b["inference_usd_est"], 4)
        with open(path, "w") as fh:
            json.dump(b, fh, indent=1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tinker", action="store_true")
    a = ap.parse_args()
    X, Y = a.year_x, a.year_x + 1
    locs = load(a.data_dir, X)
    W = windows(Y)
    shifts, regime_note = {}, "no regime model"
    if not a.no_tinker and os.environ.get("TINKER_API_KEY"):
        try:
            import regime
            index = None
            ip = os.path.join(a.data_dir, "indices_monthly.json")
            if os.path.exists(ip):
                with open(ip) as fh:
                    js = json.load(fh)
                index = {k: {r["date"]: r["value"] for r in v if r["date"] <= "%d-12" % X}
                         for k, v in js.get("series", {}).items()}
            shifts, regime_note = regime.regime_shifts(locs, X, index)
        except Exception as e:  # never fail the run on the optional LLM step
            print("regime model skipped: %r" % (e,), file=sys.stderr)

    cands, seq_req = [], []
    for lid, loc in sorted(locs.items()):
        if X not in core.complete_years(loc["days"]):
            continue
        ens, info = core.build_ensemble(loc, X, [(s, e) for _, s, e in W], regime_shift=shifts.get(lid, 0.0))
        base = {"location_id": lid, "lat": loc["lat"], "lon": loc["lon"]}
        tag = "trend %+.3fC/yr, X-anom %+.2fC" % (info["slope"], info["resid_x"])

        def add(score, wname, w, var, asr, p, why, kind):
            cands.append((score * WIN_W.get(wname, 0.55), p, kind, {
                "target": dict(base, window_start=w[0], window_end=w[1], variable=var),
                "assertion": asr, "confidence": round(min(0.97, max(0.5, p)), 3),
                "rationale": "%s; %s" % (why, tag)}))

        wmap = {wname: (s, e) for wname, s, e in W}
        for rank, (wname, var, val) in enumerate(POINTS):
            w = wmap[wname]
            pairs = [(st[var], wt) for wt, st in ens[w]
                     if st.get(var) is not None and (var not in PRECIP_RECENT or st["year"] >= PRECIP_MIN_YEAR)]
            if len(pairs) < 15:
                continue
            if var in core.COUNT_VARS and core.wquantile(pairs, 0.8) < MIN_GAP[var]:
                continue  # near-zero count: no rival can be 2 days away on the right side
            fm = focal_means(loc, X, (wname,) + w, var)
            med = core.wquantile(pairs, 0.5)
            use16 = POINT_BASE == "clim16" or (var == "precip_sum_mm" and PRECIP_BASE == "clim16")
            base_v = clim16(loc, X, w, var) if use16 else (fm[0] if fm else None)
            if base_v is not None and var in TEMP3 and TREND_ADJ:
                # the 16-yr mean is centred ~8 years before X+1: move it onto the local trend
                y0 = X - 15 if use16 else max(1980, X - 15)
                base_v += TREND_ADJ * info["slope"] * (X + 1 - (y0 + X) / 2)
            if base_v is not None:
                med = base_v + LAMBDA.get(var, LAMBDA_TEMP) * (med - base_v)
                c16 = clim16(loc, X, w, var)
                if var == "precip_sum_mm" and PRECIP_DRIFT and c16 is not None:
                    last3 = [core.window_stats(loc["days"][y], *core.window_md(*w), 0.0) for y in (X - 2, X - 1, X)
                             if y in loc["days"]]
                    last3 = [st[var] for st in last3 if st and st.get(var) is not None]
                    if len(last3) == 3:
                        med += PRECIP_DRIFT * (sum(last3) / 3 - c16)
                if var in TEMP3 and SOND_CARRY:
                    sond = clim16(loc, X - 1, (date(X, 9, 1).isoformat(), date(X, 12, 31).isoformat()), var)
                    stx = core.window_stats(loc["days"][X], (9, 1), (12, 31), 0.0)
                    if sond is not None and stx and stx.get(var) is not None:
                        med += SOND_CARRY * (stx[var] - sond)
            hist = seq_history(loc, X, w, var)
            if hist:
                seq_req.append((len(cands), hist, var, wname, loc["lat"]))
            v = round(med) if var in core.COUNT_VARS else round(med, 2)
            g = MIN_GAP[var]
            p = core.wprob(pairs, lambda x: abs(x - v) < g)
            cands.append((None, rank, val, {
                "target": dict(base, window_start=w[0], window_end=w[1], variable=var),
                "assertion": {"kind": "point", "value": v}, "confidence": round(min(0.97, max(0.05, p)), 3),
                "rationale": "clim16 shrunk toward analog-ensemble median; P(|err|<%g)=%.2f; %s" % (g, p, tag)}))

        for wname, s, e in W:
            w = (s, e)
            members = ens[w]
            if len(members) < 15:
                continue
            for var in TEMP3 + ("precip_sum_mm", "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"):
                pairs = [(st[var], wt) for wt, st in members if st.get(var) is not None]
                fm = focal_means(loc, X, (wname, s, e), var)
                if not pairs or not fm:
                    continue
                if var in core.COUNT_VARS:
                    b = fm[0]
                    if b < 3:
                        continue
                    if var == "frost_days_lt_0c":
                        v = math.ceil(b)
                        p_hi = core.wprob(pairs, lambda x: x >= v)
                        opts = (("ge", v, p_hi), ("lt", v, 1 - p_hi))
                    else:
                        v = math.floor(b)
                        p_hi = core.wprob(pairs, lambda x: x > v)
                        opts = (("gt", v, p_hi), ("le", v, 1 - p_hi))
                else:
                    lo, hi = round(min(fm) - EPS, 2), round(max(fm) + EPS, 2)
                    opts = (("gt", hi, core.wprob(pairs, lambda x: x > hi)),
                            ("le", lo, core.wprob(pairs, lambda x: x <= lo)))
                op, v, p = max(opts, key=lambda o: o[2])
                if p < MIN_P:
                    continue
                tw = TYPE_W.get(("temp" if var in TEMP3 else var, op), 0.5)
                add(tw * (2 * p - 1), wname, w, var, {"kind": "threshold", "op": op, "value": v}, p,
                    "P=%.2f vs naive clim %.2f" % (p, fm[0]), (var, op))
            for var, stat in (("warmer_than_year_x", "temp_mean_c"), ("wetter_than_year_x", "precip_sum_mm")):
                xv = core.year_x_actual(loc, X, w, stat)
                if xv is None:
                    continue
                pairs = [(st[stat], wt) for wt, st in members if st.get(stat) is not None]
                pt = core.wprob(pairs, lambda v: v > xv)
                val, p = (True, pt) if pt >= 0.5 else (False, 1 - pt)
                # trace-rain windows: one drizzle event flips every rival pair at once
                dry = stat == "precip_sum_mm" and min(xv, core.wquantile(pairs, 0.5)) < DRY_MM
                if p < (DRY_MIN_P if dry else MIN_P):
                    continue
                add(TYPE_W[(var, val)] * (2 * p - 1), wname, w, var, {"kind": "boolean", "value": val}, p,
                    "year X %s=%.1f, P(exceed)=%.2f" % (stat, xv, pt), (var, val))

    # 1) the first POINTS_FIRST point claims of every location; 2) everything else by score
    # (points valued at 0.35 * priority weight, comparable to a p~0.75 boolean), under the caps.
    if seq_req and not a.no_tinker and os.environ.get("TINKER_API_KEY"):
        try:
            import seqmodel
            preds, cfg = seqmodel.expected_points(seq_req)
            b = cfg.get("blend", 0.0) if cfg.get("use") else 0.0
            for i, pv in preds.items():
                asr = cands[i][3]["assertion"]
                var = cands[i][3]["target"]["variable"]
                v = (1 - b) * asr["value"] + b * pv
                asr["value"] = round(v) if var in core.COUNT_VARS else round(v, 2)
            if preds:
                regime_note += "; seq LoRA blend %.2f on %d points" % (b, len(preds))
                _log_seq_inference(len(preds))
        except Exception as e:  # never fail the run on the optional LLM step
            print("seq model skipped: %r" % (e,), file=sys.stderr)
    points = [c for c in cands if c[0] is None]
    first = [c for c in points if c[1] < POINTS_FIRST]
    rest = [(0.35 * c[2], c[3]) for c in points if c[1] >= POINTS_FIRST]
    rest += [(c[0], c[3]) for c in cands if c[0] is not None]
    rest.sort(key=lambda c: -c[0])
    out, seen, per_loc = [], set(), {}
    for c in [c[3] for c in first] + [c for _, c in rest]:
        t = c["target"]
        k = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if k in seen or per_loc.get(t["location_id"], 0) >= MAX_PER_LOC:
            continue
        seen.add(k)
        per_loc[t["location_id"]] = per_loc.get(t["location_id"], 0) + 1
        out.append({"claim_id": "p2_%03d" % (len(out) + 1), **c})
        if len(out) >= MAX_CLAIMS:
            break
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("predict2: %d claims for %d (%s)" % (len(out), Y, regime_note))


if __name__ == "__main__":
    main()
