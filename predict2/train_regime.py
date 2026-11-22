#!/usr/bin/env python3
"""Train the predict2 regime LoRA on Tinker (offline, one-off).

Usage: python3 predict2/train_regime.py --json-dir <dir of <location_id>.json Open-Meteo daily archives>

Examples use years X in [1945, 1970] for training and [1971, 1978] for
validation, so every label year (X+1) is <= 1979 and the scored range
(X in 1980..2024 -> X+1 in 1981..2025) never enters training. Each example uses
only the 16 years X-15..X, matching what datacollect provides at run time.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
import regime  # noqa: E402

BASE = "Qwen/Qwen3-8B"
PRICE = {"train": 0.44, "prefill": 0.195}  # USD per 1M tokens (Tinker list price)


def load_json_dir(d):
    locs = {}
    for lid in regime.CATALOG:
        fn = os.path.join(d, lid + ".json")
        if not os.path.exists(fn):
            continue
        js = json.load(open(fn))
        if js["daily"]["time"][-1] < "1979-12-31":
            continue
        dd, days = js["daily"], {}
        for i, t in enumerate(dd["time"]):
            y, m, da = int(t[:4]), int(t[5:7]), int(t[8:10])
            days.setdefault(y, []).append((m, da) + tuple(dd[k][i] for k in core.FIELDS))
        cat = {"lat": js["latitude"], "lon": js["longitude"]}
        locs[lid] = {"lat": cat["lat"], "lon": cat["lon"], "days": days}
    return locs


def examples(all_locs, xs, index=None):
    out = []
    for X in xs:
        view = {lid: {"lat": l["lat"], "lon": l["lon"],
                      "days": {y: r for y, r in l["days"].items() if X - 15 <= y <= X}} for lid, l in all_locs.items()}
        summ = {lid: regime.station_summary(view[lid], X) for lid in view}
        for lid, loc in view.items():
            if not summ.get(lid) or X + 1 not in core.complete_years(all_locs[lid]["days"]):
                continue
            _, info = core.build_ensemble(loc, X, [])
            resid = core.annual_mean(all_locs[lid]["days"][X + 1]) - info["target_level"]
            out.append({"X": X, "lid": lid, "prompt": regime.prompt_for(view, lid, X, list(view), summ, index and {k: {d: v for d, v in s.items() if d <= "%d-12" % X} for k, s in index.items()}), "resid": resid,
                        "label": regime.label(resid)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--indices", help="indices_monthly.json-style file (series name -> [{date, value}])")
    ap.add_argument("--name", default="p2-regime-r2")
    a = ap.parse_args()
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    locs = load_json_dir(a.json_dir)
    index = None
    if a.indices:
        js = json.load(open(a.indices))
        index = {k: {r["date"]: r["value"] for r in v} for k, v in js.get("series", js).items()}
    tr = examples(locs, range(1951, 1971), index)
    va = examples(locs, range(1971, 1979), index)
    assert max(e["X"] for e in tr + va) + 1 <= 1979
    tok = AutoTokenizer.from_pretrained(BASE)
    counts = {L: sum(e["label"] == L for e in tr) for L in regime.LETTERS}
    print("train", len(tr), "val", len(va), "labels", counts, flush=True)

    def datum(e):
        p = tok.encode(e["prompt"], add_special_tokens=False)
        t = tok.encode(" " + e["label"], add_special_tokens=False)
        toks = p + t
        w = [0.0] * (len(p) - 1) + [1.0] * len(t)
        return types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                           loss_fn_inputs={"target_tokens": toks[1:], "weights": w}), len(toks)

    data = [datum(e) for e in tr]
    ntok = sum(n for _, n in data) * a.epochs
    est = ntok * PRICE["train"] / 1e6 + len(va) * 3 * 800 * PRICE["prefill"] / 1e6
    print("estimated cost $%.3f (%d train tokens)" % (est, ntok), flush=True)
    if est > a.max_usd:
        raise SystemExit("over budget")

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE, rank=16).result()
    rng = random.Random(0)
    bs, step = 32, 0
    for ep in range(a.epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = [data[j][0] for j in idx[i:i + bs]]
            fb = tc.forward_backward(batch, "cross_entropy")
            op = tc.optim_step(types.AdamParams(learning_rate=a.lr))
            r = fb.result()
            op.result()
            lp = sum(float(sum(x["logprobs"].to_numpy() * d.loss_fn_inputs["weights"].to_numpy()))
                     for x, d in zip(r.loss_fn_outputs, batch))
            step += 1
            print("epoch %d step %d nll/ex %.3f" % (ep, step, -lp / len(batch)), flush=True)
    path = tc.save_weights_for_sampler(name=a.name).result().path
    print("sampler", path, flush=True)

    samp = tc.create_sampling_client(path)
    ans = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in regime.LETTERS]
    xs, ys, correct = [], [], 0
    for e in va:
        p = tok.encode(e["prompt"], add_special_tokens=False)
        futs = [samp.compute_logprobs(types.ModelInput.from_ints(p + [t]), prompt_logprobs_last_n=1) for t in ans]
        lps = [f.result()[-1] for f in futs]
        m = max(lps)
        ps = [math.exp(v - m) for v in lps]
        ps = [v / sum(ps) for v in ps]
        xs.append(ps[2] - ps[0])
        ys.append(e["resid"])
        correct += regime.LETTERS[ps.index(max(ps))] == e["label"]
    # least-squares scale of residual on (pC - pA), clipped to [0, 0.6] degC
    sxx = sum(x * x for x in xs) or 1e-9
    scale = max(0.0, min(0.6, sum(x * y for x, y in zip(xs, ys)) / sxx))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys) or 1e-9)
    vcounts = {L: sum(e["label"] == L for e in va) for L in regime.LETTERS}
    print("val acc %.3f (majority %.3f) corr %.3f scale %.3f" %
          (correct / len(va), max(vcounts.values()) / len(va), corr, scale), flush=True)
    json.dump({"base_model": BASE, "stations": list(locs), "sampler_path": path, "scale": round(scale, 4), "val_corr": round(corr, 4),
               "val_acc": round(correct / len(va), 4), "uses_indices": bool(index), "train_years": [1951, 1970], "val_years": [1971, 1978],
               "prefill_usd_per_m": PRICE["prefill"]},
              open(regime.MODEL_FILE, "w"), indent=1)
    bpath = os.path.join(HERE, "tinker_budget.json")
    b = json.load(open(bpath)) if os.path.exists(bpath) else {"cap_usd": 50.0, "training_runs": [], "training_usd_est": 0.0}
    b["training_runs"].append({"name": a.name, "base_model": BASE, "train_tokens": ntok,
                               "usd_est": round(est, 4), "sampler_path": path})
    b["training_usd_est"] = round(b["training_usd_est"] + est, 4)
    b["total_usd_est"] = round(b["training_usd_est"] + b.get("inference_usd_est", 0.0), 4)
    json.dump(b, open(bpath, "w"), indent=1)


if __name__ == "__main__":
    main()
