"""Tinker LoRA "regime" model for predict2.

The analog ensemble (core.py) already captures trend + climatology. This model
reads a synoptic summary of year X across all benchmark stations (tropical
Pacific/Indian Ocean proxies such as Lima, Singapore, Sydney, Mumbai carry the
ENSO/IOD signal; Reykjavik/London/Moscow the NAO) and classifies whether the
target station's year X+1 annual mean will land BELOW / NEAR / ABOVE the
ensemble's own trend-level forecast. The class probabilities become a small
temperature shift. It is trained only on years whose targets are <= 1979
(see train_regime.py), so it cannot memorise any scored year (X+1 >= 1981).
"""
import asyncio
import json
import os

import core

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "regime_model.json")
NEAR_BAND = 0.3  # degC: |residual| below this is NEAR
LETTERS = ("A", "B", "C")  # BELOW, NEAR, ABOVE
CATALOG = ["london_uk", "new_york_us", "tokyo_jp", "sydney_au", "cairo_eg", "mumbai_in",
           "sao_paulo_br", "reykjavik_is", "nairobi_ke", "phoenix_us", "singapore_sg",
           "chicago_us", "moscow_ru", "cape_town_za", "lima_pe", "bangkok_th",
           "anchorage_us", "madrid_es", "seoul_kr", "toronto_ca"]
SEAS = {"JF": ((1, 1), (2, 29)), "MAM": ((3, 1), (5, 31)), "JJA": ((6, 1), (8, 31)),
        "SON": ((9, 1), (11, 30)), "D": ((12, 1), (12, 31))}


def station_summary(loc, year_x):
    """Seasonal temp anomalies (degC) and annual precip anomaly (%) of year X vs years < X."""
    days = loc["days"]
    yrs = [y for y in core.complete_years(days) if y <= year_x]
    if year_x not in yrs or len(yrs) < 5:
        return None
    out = {}
    for name, (s, e) in SEAS.items():
        vals = {y: core.window_stats(days[y], s, e, 0.0) for y in yrs}
        vals = {y: v["temp_mean_c"] for y, v in vals.items() if v}
        past = [v for y, v in vals.items() if y < year_x]
        if year_x in vals and past:
            out[name] = vals[year_x] - sum(past) / len(past)
    pr = {y: sum(r[5] or 0 for r in days[y]) for y in yrs}
    past = [pr[y] for y in yrs if y < year_x]
    clim = sum(past) / len(past)
    out["P%"] = 100 * (pr[year_x] - clim) / clim if clim > 1 else 0.0
    ann = [core.annual_mean(days[y]) for y in yrs]
    out["trend10"] = (sum(ann[-5:]) / 5 - sum(ann[-10:-5]) / 5) if len(ann) >= 10 else 0.0
    return out


def fmt(s):
    return " ".join("%s%+.1f" % (k, v) for k, v in s.items() if k != "P%") + " P%+.0f%%" % s["P%"]


def index_line(index, year_x):
    """Seasonal means of climate indices in year X; `index` = {name: {'YYYY-MM': value}} with months <= X."""
    parts = []
    for name, label in (("nino34_anom", "Nino3.4"), ("nao", "NAO"), ("ao", "AO"), ("pdo", "PDO")):
        ser = index.get(name) or {}
        seas = []
        for tag, ms in (("JFM", (1, 2, 3)), ("AMJ", (4, 5, 6)), ("JAS", (7, 8, 9)), ("OND", (10, 11, 12))):
            v = [ser.get("%d-%02d" % (year_x, m)) for m in ms]
            if all(x is not None for x in v):
                seas.append("%s%+.1f" % (tag, sum(v) / 3))
        if seas:
            parts.append("%s %s" % (label, " ".join(seas)))
    return "; ".join(parts)


def prompt_for(locs, target, year_x, stations, summaries=None, index=None):
    summaries = summaries or {lid: station_summary(locs[lid], year_x) for lid in stations if lid in locs}
    lines = []
    if index:
        lines.append("Climate indices in year X: " + index_line(index, year_x))
    lines.append("Synoptic summary of year X (temperature anomalies in C vs prior years by season; precip anomaly %):")
    for lid in stations:
        if summaries.get(lid):
            lines.append("%s: %s" % (lid, fmt(summaries[lid])))
    loc = locs[target]
    lines.append("Target station: %s (lat %.1f, lon %.1f)." % (target, loc["lat"], loc["lon"]))
    lines.append("Relative to its warming-trend forecast, the target's annual mean temperature in year X+1 will be "
                 "A) below (< -%.1fC), B) near, C) above (> +%.1fC).\nAnswer:" % (NEAR_BAND, NEAR_BAND))
    return "\n".join(lines)


def label(residual):
    return "A" if residual < -NEAR_BAND else ("C" if residual > NEAR_BAND else "B")


def regime_shifts(locs, year_x, index=None, timeout=120):
    """Return ({location_id: shift_degC}, note). Raises if the model is unavailable."""
    with open(MODEL_FILE) as fh:
        cfg = json.load(fh)
    if not cfg.get("scale"):
        return {}, "regime model %s has no validated skill (scale 0); skipped" % cfg["sampler_path"].split("/")[-1]
    import tinker
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(model_path=cfg["sampler_path"]).result()
    ans = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    stations = cfg["stations"]  # the stations the LoRA was trained on
    summaries = {lid: station_summary(locs[lid], year_x) for lid in stations if lid in locs}
    targets = [lid for lid in stations if summaries.get(lid)]

    async def score(lid):
        ptoks = tok.encode(prompt_for(locs, lid, year_x, stations, summaries, index if cfg.get("uses_indices") else None), add_special_tokens=False)
        lps = await asyncio.gather(*[
            samp.compute_logprobs_async(tinker.ModelInput.from_ints(ptoks + [a]), prompt_logprobs_last_n=1)
            for a in ans])
        return lid, [lp[-1] for lp in lps]

    async def all_():
        return await asyncio.wait_for(asyncio.gather(*[score(l) for l in targets]), timeout)

    import math
    res = asyncio.run(all_())
    shifts = {}
    for lid, lps in res:
        m = max(lps)
        ps = [math.exp(v - m) for v in lps]
        z = sum(ps)
        pa, _, pc = (p / z for p in ps)
        shifts[lid] = cfg["scale"] * (pc - pa)
    _log_inference_cost(len(targets) * 3 * 700, cfg)
    return shifts, "regime model %s" % cfg["sampler_path"].split("/")[-1]


def _log_inference_cost(prefill_tokens, cfg):
    path = os.path.join(HERE, "tinker_budget.json")
    try:
        with open(path) as fh:
            b = json.load(fh)
        b.setdefault("inference_runs", 0)
        b["inference_runs"] += 1
        b["inference_usd_est"] = round(b.get("inference_usd_est", 0.0) + prefill_tokens * cfg["prefill_usd_per_m"] / 1e6, 4)
        b["total_usd_est"] = round(b["training_usd_est"] + b["inference_usd_est"], 4)
        with open(path, "w") as fh:
            json.dump(b, fh, indent=1)
    except Exception:
        pass
