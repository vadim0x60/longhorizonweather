#!/usr/bin/env python3
"""Train the predict2 anomaly-sequence LoRA (seqmodel.py) on Tinker (offline, one-off).

Usage: python3 predict2/train_seq.py --json-dir <dir of <location_id>.json Open-Meteo daily archives>

Labels are target years T in 1956..1975 (train) and 1976..1979 (validation); each example
sees only T-16..T-1. Every label year is <= 1979, so the scored range X+1 >= 1981 is never
trained on. Validation decides whether run.py uses the model (seq_model.json "use").
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
import seqmodel as S  # noqa: E402
from train_regime import load_json_dir  # noqa: E402

BASE = "Qwen/Qwen3-8B"
PRICE = {"train": 0.44, "prefill": 0.195}
WINS = {"year": ((1, 1), (12, 31)), "jja": ((6, 1), (8, 31)), "mam": ((3, 1), (5, 31)),
        "son": ((9, 1), (11, 30)), "jf": ((1, 1), (2, 28))}
WINS.update({"m%02d" % m: ((m, 1), (m, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])) for m in range(1, 13)})
VARS = ("temp_mean_c", "temp_mean_max_c", "temp_mean_min_c", "precip_sum_mm", "hot_days_ge_30c")


def examples(locs, years, wins=WINS):
    out = []
    for lid, loc in locs.items():
        for wname, (s, e) in wins.items():
            st = {y: core.window_stats(r, s, e, 0.0) for y, r in loc["days"].items()}
            for var in VARS:
                ser = {y: v[var] for y, v in st.items() if v and v.get(var) is not None}
                for T in years:
                    hist = [ser.get(y) for y in range(T - 16, T)]
                    if T not in ser or None in hist:
                        continue
                    if var == "hot_days_ge_30c" and sorted(hist)[12] < 2:
                        continue
                    out.append({"T": T, "var": var, "w": wname, "hist": hist, "next": ser[T],
                                "prompt": S.prompt(hist, var, wname, loc["lat"]), "label": S.label(hist, ser[T])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-usd", type=float, default=4.0)
    ap.add_argument("--name", default="p2-seq-r4")
    a = ap.parse_args()
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    locs = load_json_dir(a.json_dir)
    tr = examples(locs, range(1956, 1976))
    va = examples(locs, range(1976, 1980), {k: v for k, v in WINS.items() if not k.startswith("m")})
    assert max(e["T"] for e in tr + va) <= 1979
    tok = AutoTokenizer.from_pretrained(BASE)

    def datum(e):
        p = tok.encode(e["prompt"], add_special_tokens=False)
        t = tok.encode(" " + e["label"], add_special_tokens=False)
        toks = p + t
        return types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                           loss_fn_inputs={"target_tokens": toks[1:],
                                           "weights": [0.0] * (len(p) - 1) + [1.0] * len(t)}), len(toks)

    data = [datum(e) for e in tr]
    ntok = sum(n for _, n in data) * a.epochs
    nval = sum(n for _, n in (datum(e) for e in va)) * 9
    est = ntok * PRICE["train"] / 1e6 + nval * PRICE["prefill"] / 1e6
    print("train %d val %d, est $%.3f" % (len(tr), len(va), est), flush=True)
    if est > a.max_usd:
        raise SystemExit("over budget")
    tc = tinker.ServiceClient().create_lora_training_client(base_model=BASE, rank=16).result()
    rng, bs, step = random.Random(0), 64, 0
    for ep in range(a.epochs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = [data[j][0] for j in idx[i:i + bs]]
            fb = tc.forward_backward(batch, "cross_entropy")
            op = tc.optim_step(types.AdamParams(learning_rate=a.lr))
            r = fb.result()
            op.result()
            step += 1
            if step % 10 == 0:
                lp = sum(float(sum(x["logprobs"].to_numpy() * d.loss_fn_inputs["weights"].to_numpy()))
                         for x, d in zip(r.loss_fn_outputs, batch))
                print("epoch %d step %d nll %.3f" % (ep, step, -lp / len(batch)), flush=True)
    path = tc.save_weights_for_sampler(name=a.name).result().path
    print("sampler", path, flush=True)
    json.dump({"base_model": BASE, "sampler_path": path, "use": True}, open(S.MODEL_FILE, "w"))
    preds, _ = S.expected_points([(i, e["hist"], e["var"], e["w"], 40.0) for i, e in enumerate(va)], timeout=1800)
    # validation: MAE of the 16-yr mean vs the LoRA point, per variable
    res = {}
    for var in VARS:
        es = [(i, e) for i, e in enumerate(va) if e["var"] == var and i in preds]
        if not es:
            continue
        clim = sum(abs(sum(e["hist"]) / 16 - e["next"]) for _, e in es) / len(es)
        llm = sum(abs(preds[i] - e["next"]) for i, e in es) / len(es)
        half = sum(abs((sum(e["hist"]) / 16 + preds[i]) / 2 - e["next"]) for i, e in es) / len(es)
        res[var] = {"n": len(es), "mae_clim16": round(clim, 4), "mae_lora": round(llm, 4), "mae_half": round(half, 4)}
    print(json.dumps(res, indent=1), flush=True)
    cfg = {"base_model": BASE, "sampler_path": path, "train_years": [1956, 1975], "val_years": [1976, 1979],
           "validation": res, "use": False, "blend": 0.0}
    json.dump(cfg, open(S.MODEL_FILE, "w"), indent=1)
    bpath = os.path.join(HERE, "tinker_budget.json")
    b = json.load(open(bpath))
    b["training_runs"].append({"name": a.name, "base_model": BASE, "train_tokens": ntok, "usd_est": round(est, 4),
                               "sampler_path": path})
    b["training_usd_est"] = round(b["training_usd_est"] + est, 4)
    b["total_usd_est"] = round(b["training_usd_est"] + b.get("inference_usd_est", 0.0), 4)
    json.dump(b, open(bpath, "w"), indent=1)


if __name__ == "__main__":
    main()
