#!/usr/bin/env python3
"""predict2: Regime-Analogue predictor.

Builds a trend-adjusted analog ensemble for year X+1 from data dated <= X-12-31,
optionally nudged by a Tinker LoRA "regime" model (see regime.py), and emits
<=150 calibrated claims (see CONTRACTS.md).
"""
import argparse
import json
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

MAX_CLAIMS = 150
# (window kind, variable) in priority order; window kind indexes default_windows
SEASONS = {"annual": 0, "mam": 1, "jja": 2, "son": 3}


def load(data_dir, year_x):
    srcs = []
    for name in ("daily_history.csv", "daily_year_x.csv", "daily_history.csv.gz"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            srcs.append(core.load_csv(p, year_x))
    if not srcs:
        raise SystemExit("no daily_*.csv in %s" % data_dir)
    return core.merge(*srcs)


def clim_median(loc, year_x, window, var):
    """Plain (untrended) climatology median of a stat over the available years <= X."""
    vals = []
    for y in core.complete_years(loc["days"]):
        if y > year_x:
            continue
        st = core.window_stats(loc["days"][y], *core.window_md(*window), 0.0)
        if st and st.get(var) is not None:
            vals.append(st[var])
    return median(vals) if vals else None


def numeric_claim(pairs, clim, var, q_edge=0.38):
    """Threshold on the side our ensemble leans away from naive climatology."""
    med = core.wquantile(pairs, 0.5)
    if clim is None:
        clim = med
    is_count = var in core.COUNT_VARS
    if med >= clim:
        v = core.wquantile(pairs, q_edge)
        if is_count:
            v = int(v)  # gt int -> at least v+1
            p = core.wprob(pairs, lambda x: x > v)
        else:
            v = round(v, 1)
            p = core.wprob(pairs, lambda x: x > v)
        return {"kind": "threshold", "op": "gt", "value": v}, p, med
    v = core.wquantile(pairs, 1 - q_edge)
    if is_count:
        v = int(v) + 1
        p = core.wprob(pairs, lambda x: x < v)
    else:
        v = round(v, 1)
        p = core.wprob(pairs, lambda x: x < v)
    return {"kind": "threshold", "op": "lt", "value": v}, p, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tinker", action="store_true")
    a = ap.parse_args()
    X, Y = a.year_x, a.year_x + 1
    locs = load(a.data_dir, X)
    windows = core.default_windows(Y)
    shifts, regime_note = {}, "no regime model"
    if not a.no_tinker and os.environ.get("TINKER_API_KEY"):
        try:
            import regime
            shifts, regime_note = regime.regime_shifts(locs, X)
        except Exception as e:  # never fail the run on the optional LLM step
            print("regime model skipped: %r" % (e,), file=sys.stderr)

    cands = []
    for lid, loc in locs.items():
        if X not in core.complete_years(loc["days"]):
            continue
        ens, info = core.build_ensemble(loc, X, windows, regime_shift=shifts.get(lid, 0.0))
        base = {"location_id": lid, "lat": loc["lat"], "lon": loc["lon"]}
        tag = "trend %+.3fC/yr, X-anom %+.2fC, regime %+.2fC" % (info["slope"], info["resid_x"], shifts.get(lid, 0.0))

        def add(prio, w, var, assertion, p, why):
            cands.append((prio, p, {
                "target": dict(base, window_start=w[0], window_end=w[1], variable=var),
                "assertion": assertion, "confidence": round(min(0.97, max(0.5, p)), 3),
                "rationale": "%s; %s" % (why, tag)}))

        for wname, prio_base, vars_ in (
                ("annual", 0, ["temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "precip_sum_mm"]),
                ("jja", 1, ["temp_mean_c"]), ("mam", 1, ["temp_mean_c"]), ("son", 1, ["temp_mean_c"]),
                ("annual", 2, ["hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"])):
            w = windows[SEASONS[wname]]
            for var in vars_:
                pairs = [(st[var], wt) for wt, st in ens[w] if st.get(var) is not None]
                if len(pairs) < 15:
                    continue
                if var in ("hot_days_ge_30c", "frost_days_lt_0c") and core.wquantile(pairs, 0.5) < 3:
                    continue  # degenerate count, skip
                clim = clim_median(loc, X, w, var)
                asr, p, med = numeric_claim(pairs, clim, var)
                add(prio_base, w, var, asr, p, "analog median %.1f vs clim %.1f" % (med, clim if clim is not None else med))
        # boolean comparisons with year X (regression of X's anomaly toward trend)
        for w in windows[:4]:
            for var, stat in (("warmer_than_year_x", "temp_mean_c"), ("wetter_than_year_x", "precip_sum_mm")):
                xv = core.year_x_actual(loc, X, w, stat)
                if xv is None:
                    continue
                pairs = [(st[stat], wt) for wt, st in ens[w] if st.get(stat) is not None]
                p = core.wprob(pairs, lambda v: v > xv)
                if abs(p - 0.5) >= 0.12:
                    add(1.5 if var.startswith("warm") else 2.5, w, var,
                        {"kind": "boolean", "value": p > 0.5}, max(p, 1 - p),
                        "year X %s=%.1f, P(exceed)=%.2f" % (stat, xv, p))

    # priority first, then confidence; keep at most one claim per target
    cands.sort(key=lambda c: (c[0], -c[1]))
    out, seen = [], set()
    for _, _, c in cands:
        t = c["target"]
        k = (t["location_id"], t["window_start"], t["window_end"], t["variable"])
        if k in seen:
            continue
        seen.add(k)
        c = {"claim_id": "p2_%03d" % (len(out) + 1), **c}
        out.append(c)
        if len(out) >= MAX_CLAIMS:
            break
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("predict2: %d claims for %d (%s)" % (len(out), Y, regime_note))


if __name__ == "__main__":
    main()
