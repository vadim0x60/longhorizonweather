"""Tinker LoRA "anomaly-sequence" point model for predict2.

For one (location, window, variable) the model sees the trailing 16 yearly values as
standardized anomalies from their own mean (oldest first; no station name, no years), and
predicts which of 9 bins the next year's standardized anomaly falls in. The expected bin
centre gives a point forecast clim + sd * E[z]. It is trained by train_seq.py only on
target years <= 1979, so no scored year (X+1 >= 1981) is ever a training label.
"""
import asyncio
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "seq_model.json")
EDGES = (-1.5, -1.0, -0.5, -0.17, 0.17, 0.5, 1.0, 1.5)
CENTRES = (-1.9, -1.25, -0.75, -0.33, 0.0, 0.33, 0.75, 1.25, 1.9)
LETTERS = "ABCDEFGHI"
VAR_TEXT = {"temp_mean_c": "mean temperature", "temp_mean_max_c": "mean daily maximum temperature",
            "temp_mean_min_c": "mean daily minimum temperature", "precip_sum_mm": "total precipitation",
            "hot_days_ge_30c": "count of days >= 30C"}
WIN_TEXT = {"year": "full year", "jja": "Jun-Aug", "mam": "Mar-May", "son": "Sep-Nov", "jf": "Jan-Feb"}
WIN_TEXT.update({"m%02d" % m: n for m, n in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)})


def standardize(values):
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / max(1, len(values) - 1))
    return m, max(sd, 1e-6)


def prompt(values, var, wname, lat):
    m, sd = standardize(values)
    z = " ".join("%+.1f" % ((v - m) / sd) for v in values)
    return ("Yearly %s, %s, %s-hemisphere %s-latitude station. Standardized anomalies vs the "
            "series mean, oldest first:\n%s\nBin of next year's anomaly (A<-1.5 B<-1 C<-0.5 D<-0.17 "
            "E near 0 F>0.17 G>0.5 H>1 I>1.5):" % (
                VAR_TEXT[var], WIN_TEXT[wname], "northern" if lat >= 0 else "southern",
                "high" if abs(lat) > 45 else ("mid" if abs(lat) > 23 else "low"), z))


def label(values, nxt):
    m, sd = standardize(values)
    z = (nxt - m) / sd
    return LETTERS[sum(z > e for e in EDGES)]


def expected_points(requests, timeout=150):
    """requests: [(key, values, var, wname, lat)] -> {key: clim + sd*E[z]}. Raises if unavailable."""
    with open(MODEL_FILE) as fh:
        cfg = json.load(fh)
    if not cfg.get("use"):
        return {}, cfg
    import tinker
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    samp = tinker.ServiceClient().create_sampling_client(model_path=cfg["sampler_path"]).result()
    ans = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]

    async def one(key, values, var, wname, lat):
        p = tok.encode(prompt(values, var, wname, lat), add_special_tokens=False)
        lps = await asyncio.gather(*[samp.compute_logprobs(tinker.ModelInput.from_ints(p + [a]),
                                                           prompt_logprobs_last_n=1) for a in ans])
        lp = [x[-1] for x in lps]
        mx = max(lp)
        w = [math.exp(v - mx) for v in lp]
        ez = sum(c * x for c, x in zip(CENTRES, w)) / sum(w)
        m, sd = standardize(values)
        return key, m + sd * ez

    async def run():
        return await asyncio.wait_for(asyncio.gather(*[one(*r) for r in requests]), timeout)

    return dict(asyncio.run(run())), cfg
