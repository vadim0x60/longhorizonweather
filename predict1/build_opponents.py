#!/usr/bin/env python3
"""Offline: summarize how rival predictors place claims, as `opponents.json`.

For each sample year X we need rivals' claim files (produced by running their
run.py on a datacollect dir for X) and the same data dir. Rival assertions are
stored relative to predict1's own predictive distribution for that target
(z = (value - mean) / sd), so the summary describes *where rivals bet*, never
what the weather turned out to be. No ground truth is read here.

usage: build_opponents.py --years 1983,1986 --claims-root /tmp/cl --data-root /tmp/dc
       (claims in <claims-root>/<X>/predict{2..10}.json, data in <data-root>/<X>/)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402

MAX_SAMPLES = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True)
    ap.add_argument("--claims-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=run.OPP_PATH)
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    count = defaultdict(float)          # var|wname -> rival claims
    count_loc = defaultdict(float)      # var|wname|loc -> rival claims
    slots_loc = defaultdict(float)
    bool_loc = defaultdict(lambda: [0.0, 0.0])  # bvar|wname|loc -> [true, false] claims
    slots = defaultdict(float)          # var|wname -> (year, loc) opportunities
    numeric = defaultdict(list)         # var|wkind -> [kind, op, z1, z2]
    bool_n = defaultdict(lambda: [0.0] * 20)   # bvar|wname -> per p-bin [true, false] claims
    bool_slots = defaultdict(lambda: [0.0] * 10)
    cat_rank = defaultdict(lambda: defaultdict(float))  # cvar|wkind -> rank -> claims
    cat_slots = defaultdict(float)
    for X in years:
        F = run.Forecaster(X, os.path.join(a.data_root, str(X)))
        for loc in F.locs:
            for wname, m1, m2 in run.windows():
                for var in run.NUM_VARS:
                    if F.dist(loc, wname, var) is not None:
                        slots[f"{var}|{wname}"] += 1
                        slots_loc[f"{var}|{wname}|{loc}"] += 1
                for bvar in ("warmer_than_year_x", "wetter_than_year_x"):
                    p, _ = F.p_bool(loc, wname, bvar)
                    if p is not None:
                        bool_slots[f"{bvar}|{wname}"][min(9, int(p * 10))] += 1
                        slots_loc[f"{bvar}|{wname}|{loc}"] += 1
                if m2 > m1:
                    for cvar in ("hottest_month", "wettest_month"):
                        if F.cat_probs(loc, wname, cvar):
                            cat_slots[f"{cvar}|{run.wkind(wname)}"] += 1
        for i in range(2, 11):
            f = os.path.join(a.claims_root, str(X), f"predict{i}.json")
            if not os.path.exists(f):
                continue
            for c in json.load(open(f)):
                t, asr = c["target"], c["assertion"]
                loc, var = t["location_id"], t["variable"]
                wname = run.window_name(t["window_start"], t["window_end"])
                if loc not in F.locs or wname is None or not t["window_start"].startswith(str(X + 1)):
                    continue
                if var in run.NUM_VARS:
                    d = F.dist(loc, wname, var)
                    if d is None:
                        continue
                    count[f"{var}|{wname}"] += 1
                    count_loc[f"{var}|{wname}|{loc}"] += 1
                    if asr["kind"] == "threshold":
                        numeric[f"{var}|{run.wkind(wname)}"].append(
                            ["threshold", asr["op"], round((asr["value"] - d.m) / d.s, 3), None])
                    else:
                        numeric[f"{var}|{run.wkind(wname)}"].append(
                            ["interval", None, round((asr["low"] - d.m) / d.s, 3), round((asr["high"] - d.m) / d.s, 3)])
                elif var in ("warmer_than_year_x", "wetter_than_year_x"):
                    p, _ = F.p_bool(loc, wname, var)
                    if p is None:
                        continue
                    b = min(9, int(p * 10))
                    bool_n[f"{var}|{wname}"][2 * b + (0 if asr["value"] else 1)] += 1
                    bool_loc[f"{var}|{wname}|{loc}"][0 if asr["value"] else 1] += 1
                elif var in ("hottest_month", "wettest_month"):
                    cp = F.cat_probs(loc, wname, var)
                    if not cp or asr["value"] not in cp:
                        continue
                    ranked = sorted(cp, key=cp.get, reverse=True)
                    cat_rank[f"{var}|{run.wkind(wname)}"][ranked.index(asr["value"])] += 1
    opp = {"years": years, "rate": {}, "rate_loc": {}, "numeric": {}, "bool": {}, "bool_loc": {}, "cat": {}}
    for k, n in slots_loc.items():
        if k in count_loc:
            opp["rate_loc"][k] = round(count_loc[k] / n, 4)
        if k in bool_loc:
            opp["bool_loc"][k] = [round(x / n, 4) for x in bool_loc[k]]
    for k, n in slots.items():
        opp["rate"][k] = round(count[k] / n, 4)
    for k, v in numeric.items():
        step = max(1, len(v) // MAX_SAMPLES)
        opp["numeric"][k] = v[::step]
    for k, arr in bool_n.items():
        s = bool_slots[k]
        opp["bool"][k] = [round(arr[j] / s[j // 2], 4) if s[j // 2] else 0.0 for j in range(20)]
    for k, ranks in cat_rank.items():
        n = cat_slots[k]
        opp["cat"][k] = [round(ranks[r] / n, 4) if n else 0.0 for r in range(12)]
    json.dump(opp, open(a.out, "w"))
    print(f"opponents: {len(opp['rate'])} rate keys, {sum(len(v) for v in opp['numeric'].values())} numeric samples",
          file=sys.stderr)


if __name__ == "__main__":
    main()
