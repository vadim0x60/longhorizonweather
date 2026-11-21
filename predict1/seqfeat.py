"""Shared, anonymized sequence encoding for the Tinker LoRA corrector.

A training/inference example is one (series, target year t) pair, encoded only
from values strictly before t: no calendar years, no station or city names.
The label is the bin of the residual of y_t against a fixed baseline forecast
F_t = 0.5 * mean(all prior) + 0.5 * mean(last 10 prior), in units of the prior
standard deviation.
"""
import math

MAX_LEN = 20
BIN_EDGES = (-1.0, -0.35, 0.35, 1.0)
BIN_LABELS = (" A", " B", " C", " D", " E")  # single tokens in Qwen BPE


def baseline(prior_vals):
    clim = sum(prior_vals) / len(prior_vals)
    last = prior_vals[-10:]
    f = 0.5 * clim + 0.5 * sum(last) / len(last)
    mu = clim
    sd = math.sqrt(sum((v - mu) ** 2 for v in prior_vals) / max(1, len(prior_vals) - 1))
    return f, sd


def encode(prior, t, kind, window, abs_lat):
    """prior: {year: value} with all years < t. Returns (prompt, F, sd) or None."""
    ys = sorted(y for y in prior if y < t)
    if len(ys) < 8:
        return None
    vals = [prior[y] for y in ys]
    f, sd = baseline(vals)
    if sd < 1e-6:
        return None
    mu = sum(vals) / len(vals)
    toks = []
    for y in range(t - MAX_LEN, t):
        if y in prior:
            toks.append(f"{int(round(10 * (prior[y] - mu) / sd)):+d}")
        else:
            toks.append("_")
    fz = int(round(10 * (f - mu) / sd))
    prompt = (f"var={kind} win={window} lat={int(abs_lat // 10) * 10} n={len(ys)}\n"
              f"z: {' '.join(toks)}\nbase: {fz:+d}\nbin:")
    return prompt, f, sd


def label(y, f, sd):
    r = (y - f) / sd
    for edge, lab in zip(BIN_EDGES, BIN_LABELS):
        if r < edge:
            return lab
    return BIN_LABELS[-1]
