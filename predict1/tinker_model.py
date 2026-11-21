"""Inference side of the Tinker LoRA anomaly-sequence corrector.

Reads `tinker_model.json` (written by tinker_train.py) for the sampler path and
bin centres. Every prompt is built by seqfeat.encode from values of years <= X
only, so the remote model never sees year-X+1 information (or any calendar year).
"""
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "tinker_model.json")
BUDGET = os.path.join(HERE, "tinker_budget.json")
KINDS = ("temp_mean_c", "precip_sum_mm")
PREFILL_USD_PER_M = 0.195  # Qwen3-8B


def _client(path):
    import tinker
    return tinker.ServiceClient().create_sampling_client(model_path=path).result()


def bin_probs(path, prompts, timeout=120):
    """Return ([{label: prob}], prompt_tokens) for each prompt."""
    import seqfeat
    from tinker import types
    sc = _client(path)
    tok = sc.get_tokenizer()
    lab_ids = {}
    for lab in seqfeat.BIN_LABELS:
        ids = tok.encode(lab, add_special_tokens=False)
        assert len(ids) == 1, (lab, ids)
        lab_ids[ids[0]] = lab
    params = types.SamplingParams(max_tokens=1, temperature=1.0)
    futs, ntok = [], 0
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        ntok += len(ids)
        futs.append(sc.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                              sampling_params=params, topk_sample_logprobs=20))
    t0, out = time.time(), []
    for f in futs:
        res = f.result(timeout=max(1.0, timeout - (time.time() - t0)))
        top = res.sequences[0].topk_logprobs[0] or []
        pr = {}
        for tid, lp in top:
            if tid in lab_ids:
                pr[lab_ids[tid]] = pr.get(lab_ids[tid], 0.0) + math.exp(lp)
        z = sum(pr.values())
        out.append({k: v / z for k, v in pr.items()} if z > 0 else {})
    return out, ntok


def _log(tokens, note):
    try:
        b = json.load(open(BUDGET))
        b["entries"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": "prefill",
                             "model": "Qwen/Qwen3-8B", "tokens": tokens,
                             "usd": round(tokens * PREFILL_USD_PER_M / 1e6, 4), "note": note})
        b["total_usd"] = round(sum(e["usd"] for e in b["entries"]), 4)
        json.dump(b, open(BUDGET, "w"), indent=1)
    except Exception:
        pass


def predictions(all_series, year_x, locs):
    """{(loc, window, var): predicted value for year X+1} from the LoRA model."""
    import seqfeat
    if not os.path.exists(META):
        return {}
    meta = json.load(open(META))
    centers = meta["bin_centers"]
    keys, prompts, base = [], [], []
    for (loc, wname, var), series in all_series.items():
        if var not in KINDS:
            continue
        prior = {y: v for y, v in series.items() if y <= year_x and y > year_x - 40}
        enc = seqfeat.encode(prior, year_x + 1, var, wname, abs(locs[loc][0]))
        if enc is None:
            continue
        prompt, f, sd = enc
        keys.append((loc, wname, var)); prompts.append(prompt); base.append((f, sd))
    if not prompts:
        return {}
    probs, ntok = bin_probs(meta["sampler_path"], prompts)
    _log(ntok, f"run.py inference X={year_x}, {len(prompts)} prompts")
    out = {}
    for k, pr, (f, sd) in zip(keys, probs, base):
        if pr:
            out[k] = f + sd * sum(p * centers[lab] for lab, p in pr.items())
    return out
