#!/usr/bin/env python3
"""predict2: Regime-Analogue predictor.

Builds a trend-adjusted analog ensemble for year X+1 from data dated <= X-12-31
(core.py), optionally nudged by a Tinker LoRA "regime" model (regime.py), and
emits <=150 claims (see CONTRACTS.md).

Points are only scored when another predictor claims the same target with a
disjoint assertion, so claims sit on "focal" targets: standard calendar windows
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
    return core.merge(*srcs)


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


EXTRA = ("temp_max_c", "temp_min_c", "precip_max_day_mm", "wind_max_kmh")


def rnd_out(lo, hi, var):
    if var in core.COUNT_VARS:
        return math.floor(lo), math.ceil(hi)
    return math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10


def shapes(pairs, var):
    """Interval and one-sided candidates from the ensemble quantiles: [(shape, assertion, p)]."""
    out = []
    for cov in (0.5, 0.65, 0.8):
        lo, hi = rnd_out(core.wquantile(pairs, 0.5 - cov / 2), core.wquantile(pairs, 0.5 + cov / 2), var)
        if hi > lo:
            out.append(("iv%d" % (cov * 100), {"kind": "interval", "low": lo, "high": hi},
                        core.wprob(pairs, lambda x: lo <= x <= hi)))
    for q in (0.2, 0.3):
        lo = rnd_out(core.wquantile(pairs, q), 0, var)[0]
        hi = rnd_out(0, core.wquantile(pairs, 1 - q), var)[1]
        out.append(("gtq%d" % (q * 100), {"kind": "threshold", "op": "gt", "value": lo}, core.wprob(pairs, lambda x: x > lo)))
        out.append(("ltq%d" % (q * 100), {"kind": "threshold", "op": "lt", "value": hi}, core.wprob(pairs, lambda x: x < hi)))
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tinker", action="store_true")
    ap.add_argument("--pool", action="store_true", help="dump every candidate (analysis only, not a valid submission)")
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

    cands = []
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
                if p < MIN_P and not a.pool:
                    continue
                tw = TYPE_W.get(("temp" if var in TEMP3 else var, op), 0.5)
                add(tw * (2 * p - 1), wname, w, var, {"kind": "threshold", "op": op, "value": v}, p,
                    "P=%.2f vs naive clim %.2f" % (p, fm[0]), (var, op))
            if a.pool:
                for var in TEMP3 + ("precip_sum_mm", "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm") + EXTRA:
                    pairs = [(st[var], wt) for wt, st in members if st.get(var) is not None]
                    if len(pairs) < 15 or core.wquantile(pairs, 0.9) == core.wquantile(pairs, 0.1):
                        continue
                    for shp, asr, p in shapes(pairs, var):
                        add(0, wname, w, var, asr, p, shp, (var, shp))
            for var, stat in (("warmer_than_year_x", "temp_mean_c"), ("wetter_than_year_x", "precip_sum_mm")):
                xv = core.year_x_actual(loc, X, w, stat)
                if xv is None:
                    continue
                pairs = [(st[stat], wt) for wt, st in members if st.get(stat) is not None]
                pt = core.wprob(pairs, lambda v: v > xv)
                val, p = (True, pt) if pt >= 0.5 else (False, 1 - pt)
                # trace-rain windows: one drizzle event flips every rival pair at once
                dry = stat == "precip_sum_mm" and min(xv, core.wquantile(pairs, 0.5)) < DRY_MM
                if p < (DRY_MIN_P if dry else MIN_P) and not a.pool:
                    continue
                add(TYPE_W[(var, val)] * (2 * p - 1), wname, w, var, {"kind": "boolean", "value": val}, p,
                    "year X %s=%.1f, P(exceed)=%.2f" % (stat, xv, pt), (var, val))

    cands.sort(key=lambda c: -c[0])
    out, seen = [], set()
    for _, _, kind, c in cands:
        t = c["target"]
        k = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if k in seen and not a.pool:
            continue
        seen.add(k)
        c = {"claim_id": "p2_%03d" % (len(out) + 1), **c}
        if a.pool:
            c["kind_key"] = str(kind)
        out.append(c)
        if len(out) >= MAX_CLAIMS and not a.pool:
            break
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("predict2: %d claims for %d (%s)" % (len(out), Y, regime_note))


if __name__ == "__main__":
    main()
