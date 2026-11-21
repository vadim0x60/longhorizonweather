#!/usr/bin/env python3
"""Offline (one-time) LoRA training of the anomaly-sequence corrector on Tinker.

Data: GHCN-Daily station files (TMAX/TMIN/PRCP), target years <= 1969 for
training and 1970..1979 for out-of-time validation, so the model never sees any
year >= 1980 (the earliest allowed X). Examples are anonymized by seqfeat.encode.

usage:
  tinker_train.py prep  --ghcn-dir DIR --out examples.jsonl
  tinker_train.py train --examples examples.jsonl [--n-train N] [--steps-epochs E]
"""
import argparse
import calendar
import csv
import glob
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seqfeat  # noqa: E402

WINDOWS = [("year", 1, 12), ("mam", 3, 5), ("jja", 6, 8), ("son", 9, 11), ("jf", 1, 2)] + \
          [(f"m{m:02d}", m, m) for m in range(1, 13)]
TRAIN_MAX, VAL_MAX = 1969, 1979
BUDGET = os.path.join(HERE, "tinker_budget.json")
MODEL_META = os.path.join(HERE, "tinker_model.json")
BASE_MODEL = "Qwen/Qwen3-8B"
PRICE = {"train": 0.44, "prefill": 0.195, "sample": 0.60}  # USD per 1M tokens (Qwen3-8B)


def station_monthly(path, max_year):
    lat = None
    tsum, tn, psum, pn = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
    tx, tm = {}, {}
    with gzip.open(path, "rt") as f:
        for row in csv.reader(f):
            sid, d, el, v, _, q = row[:6]
            if q.strip():
                continue
            y = int(d[:4])
            if y > max_year:
                continue
            key = (y, int(d[4:6]))
            if el == "PRCP":
                psum[key] += int(v) / 10; pn[key] += 1
            elif el == "TMAX":
                tx[d] = int(v) / 10
            elif el == "TMIN":
                tm[d] = int(v) / 10
    for d, a in tx.items():
        b = tm.get(d)
        if b is not None:
            key = (int(d[:4]), int(d[4:6]))
            tsum[key] += (a + b) / 2; tn[key] += 1
    out = {}
    for key in set(tn) | set(pn):
        nd = calendar.monthrange(*key)[1]
        t = tsum[key] / tn[key] if tn[key] >= 0.9 * nd else None
        p = psum[key] * nd / pn[key] if pn[key] >= 0.9 * nd else None
        out[key] = (t, p, nd)
    return out


def prep(args):
    st = {}
    for line in open(args.stations):
        st[line[:11]] = float(line[12:20])
    rng = random.Random(0)
    n = 0
    with open(args.out, "w") as fo:
        for path in sorted(glob.glob(os.path.join(args.ghcn_dir, "*.csv.gz"))):
            sid = os.path.basename(path)[:11]
            mon = station_monthly(path, VAL_MAX)
            years = sorted({y for y, _ in mon})
            for wname, m1, m2 in WINDOWS:
                for idx, kind in ((0, "temp_mean_c"), (1, "precip_sum_mm")):
                    ser = {}
                    for y in years:
                        parts = [mon.get((y, m)) for m in range(m1, m2 + 1)]
                        if any(p is None or p[idx] is None for p in parts):
                            continue
                        if idx == 0:
                            ser[y] = sum(p[0] * p[2] for p in parts) / sum(p[2] for p in parts)
                        else:
                            ser[y] = sum(p[1] for p in parts)
                    for t in sorted(ser):
                        enc = seqfeat.encode({y: v for y, v in ser.items() if y < t and y >= t - 40}, t,
                                             kind, wname, abs(st.get(sid, 0.0)))
                        if enc is None:
                            continue
                        prompt, f, sd = enc
                        split = "train" if t <= TRAIN_MAX else "val"
                        fo.write(json.dumps({"split": split, "prompt": prompt,
                                             "label": seqfeat.label(ser[t], f, sd),
                                             "r": (ser[t] - f) / sd, "kind": kind, "win": wname,
                                             "r_rand": rng.random()}) + "\n")
                        n += 1
    print("examples", n)


def log_spend(kind, tokens, note):
    b = json.load(open(BUDGET)) if os.path.exists(BUDGET) else {"cap_usd": 50.0, "entries": [], "total_usd": 0.0}
    cost = tokens * PRICE[kind] / 1e6
    b["entries"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind,
                         "model": BASE_MODEL, "tokens": tokens, "usd": round(cost, 4), "note": note})
    b["total_usd"] = round(sum(e["usd"] for e in b["entries"]), 4)
    json.dump(b, open(BUDGET, "w"), indent=1)
    return b["total_usd"]


def train(args):
    import tinker
    from tinker import types
    rows = [json.loads(l) for l in open(args.examples)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    rng = random.Random(1)
    rng.shuffle(tr); rng.shuffle(va)
    tr, va = tr[:args.n_train], va[:args.n_val]
    # empirical residual mean per bin (train) -> used as bin values at inference
    centers = {}
    for lab in seqfeat.BIN_LABELS:
        rs = [r["r"] for r in tr if r["label"] == lab]
        centers[lab] = sum(rs) / len(rs)
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=16).result()
    tok = tc.get_tokenizer()

    def datum(r):
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["label"], add_special_tokens=False)
        toks = p + c
        w = [0.0] * (len(p) - 1) + [1.0] * len(c)
        return types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                           loss_fn_inputs={"target_tokens": toks[1:], "weights": w}), len(toks)

    total_tok = 0
    bs = args.batch
    steps = (len(tr) // bs) * args.epochs
    for step in range(steps):
        i = (step * bs) % len(tr)
        batch = [datum(r) for r in tr[i:i + bs]]
        total_tok += sum(n for _, n in batch)
        lr = args.lr * (1 - step / steps)
        fb = tc.forward_backward([d for d, _ in batch], "cross_entropy")
        op = tc.optim_step(types.AdamParams(learning_rate=lr))
        out = fb.result(); op.result()
        if step % 10 == 0:
            print(f"step {step}/{steps} tokens {total_tok}", flush=True)
    spent = log_spend("train", total_tok, f"LoRA r16 {len(tr)} ex x{args.epochs} ep, bs {bs}")
    path = tc.save_weights_for_sampler(name=f"p1-seqcorr-{int(time.time())}").result().path
    print("sampler path", path, "total spend", spent)
    meta = {"base_model": BASE_MODEL, "sampler_path": path, "bin_centers": centers,
            "train_max_target_year": TRAIN_MAX, "n_train": len(tr), "epochs": args.epochs}
    json.dump(meta, open(MODEL_META, "w"), indent=1)
    evaluate(path, va, centers)


def evaluate(path, va, centers):
    import tinker_model
    probs, ptoks = tinker_model.bin_probs(path, [r["prompt"] for r in va])
    log_spend("prefill", ptoks, f"validation on {len(va)} examples (1970-1979 targets)")
    ll, ll0, se, se0, sgn = 0.0, 0.0, 0.0, 0.0, 0
    prior = {lab: sum(r["label"] == lab for r in va) / len(va) for lab in seqfeat.BIN_LABELS}
    for r, pr in zip(va, probs):
        ll += math.log(max(1e-6, pr.get(r["label"], 1e-6)))
        ll0 += math.log(prior[r["label"]])
        ez = sum(pr[k] * centers[k] for k in pr)
        se += (r["r"] - ez) ** 2; se0 += r["r"] ** 2
        sgn += (ez > 0) == (r["r"] > 0)
    n = len(va)
    rep = {"n_val": n, "logloss_model": -ll / n, "logloss_marginal": -ll0 / n,
           "mse_model": se / n, "mse_baseline": se0 / n, "sign_acc": sgn / n}
    print(json.dumps(rep, indent=1))
    meta = json.load(open(MODEL_META))
    meta["validation_1970_1979"] = rep
    json.dump(meta, open(MODEL_META, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--ghcn-dir", required=True)
    p.add_argument("--stations", required=True); p.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--examples", required=True)
    t.add_argument("--n-train", type=int, default=16000); t.add_argument("--n-val", type=int, default=1500)
    t.add_argument("--epochs", type=int, default=1); t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else train(a)
