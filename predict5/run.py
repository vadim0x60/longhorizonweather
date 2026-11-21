#!/usr/bin/env python3
"""predict5 — The Secular Warming & Trend Extrapolator.

Bets that year X+1 continues the multi-decadal warming signal: seasonal means
and minima above the trailing 15-year baseline, more hot days, fewer frost
days. Each candidate target gets a per-location OLS trend over the trailing
history (shrunk toward a global warming prior), a predictive Normal
distribution, and a probability that the warm-side claim holds. The most
probable claims (at most 150, one per target) are emitted.

Offline, stdlib only, reads only --data-dir (dates <= X-12-31).
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import date, timedelta

MAX_CLAIMS = 150
MIN_CONF = 0.55
PRIOR_SLOPE_C_PER_YR = 0.025  # global-mean land warming since 1980, roughly
PRIOR_WEIGHT_YEARS = 8.0       # prior counts like this many years of slope evidence

# (name, start_mmdd, end_mmdd); windows that do not cross a year boundary.
WINDOWS = [
    ("year", "01-01", "12-31"),
    ("JF", "01-01", "02-28"),
    ("MAM", "03-01", "05-31"),
    ("JJA", "06-01", "08-31"),
    ("SON", "09-01", "11-30"),
    ("Q1", "01-01", "03-31"),
    ("Q2", "04-01", "06-30"),
    ("Q3", "07-01", "09-30"),
    ("Q4", "10-01", "12-31"),
] + [
    (f"M{m:02d}", f"{m:02d}-01", f"{m:02d}-{[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]:02d}")
    for m in range(1, 13)
]

FIELDS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max"]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_daily(data_dir, max_date):
    """{location_id: {"lat","lon","days": {date: tuple}}} from history + year_x, deduped."""
    locs = {}
    for name in ("daily_history.csv", "daily_year_x.csv"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                d = r["date"]
                if d > max_date:
                    continue  # never look past X-12-31
                loc = locs.setdefault(r["location_id"], {"lat": float(r["lat"]), "lon": float(r["lon"]), "days": {}})
                vals = []
                for c in FIELDS:
                    v = r.get(c, "")
                    vals.append(float(v) if v not in ("", None, "nan", "NaN") else None)
                loc["days"][d] = tuple(vals)
    return locs


def window_days(year, s, e):
    d0 = date.fromisoformat(f"{year}-{s}")
    d1 = date.fromisoformat(f"{year}-{e}")
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def stat(days, ds, var):
    rows = [days.get(d) for d in ds]
    if sum(r is not None for r in rows) < 0.9 * len(ds):
        return None
    rows = [r for r in rows if r is not None]
    if var == "temp_mean_c":
        xs = [r[0] for r in rows if r[0] is not None]
        return sum(xs) / len(xs) if xs else None
    if var == "temp_mean_max_c":
        xs = [r[1] for r in rows if r[1] is not None]
        return sum(xs) / len(xs) if xs else None
    if var == "temp_mean_min_c":
        xs = [r[2] for r in rows if r[2] is not None]
        return sum(xs) / len(xs) if xs else None
    if var == "temp_min_c":
        xs = [r[2] for r in rows if r[2] is not None]
        return min(xs) if xs else None
    if var == "hot_days_ge_30c":
        return float(sum(1 for r in rows if r[1] is not None and r[1] >= 30.0))
    if var == "frost_days_lt_0c":
        return float(sum(1 for r in rows if r[2] is not None and r[2] < 0.0))
    raise ValueError(var)


def trend_forecast(years, ys, target_year, prior_slope):
    """OLS slope shrunk toward prior_slope; returns (pred, predictive_sd, slope)."""
    n = len(ys)
    ybar = sum(ys) / n
    tbar = sum(years) / n
    sxx = sum((t - tbar) ** 2 for t in years)
    b_ols = sum((t - tbar) * (y - ybar) for t, y in zip(years, ys)) / sxx if sxx else 0.0
    # Precision-weighted shrinkage: sxx is the data's slope information, prior acts like extra years.
    w_data = sxx
    w_prior = PRIOR_WEIGHT_YEARS ** 3 / 12.0  # sxx of an 8-year evenly spaced series
    b = (w_data * b_ols + w_prior * prior_slope) / (w_data + w_prior)
    resid = [y - (ybar + b * (t - tbar)) for t, y in zip(years, ys)]
    s = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    h = target_year - tbar
    sd = s * math.sqrt(1.0 + 1.0 / n + h * h / (sxx + w_prior if sxx else 1.0))
    return ybar + b * h, max(sd, 1e-6), b


def candidates(locs, x):
    target_year = x + 1
    hist_years = list(range(max(1980, x - 15), x + 1))
    out = []
    for lid, loc in locs.items():
        days = loc["days"]
        for wname, s, e in WINDOWS:
            span = {yr: window_days(yr, s, e) for yr in hist_years}
            for var in ("temp_mean_c", "temp_mean_min_c", "temp_mean_max_c",
                        "hot_days_ge_30c", "frost_days_lt_0c"):
                series = [(yr, stat(days, span[yr], var)) for yr in hist_years]
                series = [(t, v) for t, v in series if v is not None]
                if len(series) < 10 or series[-1][0] != x:
                    continue
                yrs = [t for t, _ in series]
                ys = [v for _, v in series]
                base = sum(ys) / len(ys)
                ndays = len(span[x])
                if var.startswith("temp"):
                    prior = PRIOR_SLOPE_C_PER_YR * (1.3 if var == "temp_mean_min_c" else 1.0)
                    pred, sd, b = trend_forecast(yrs, ys, target_year, prior)
                    p = 1.0 - phi((base - pred) / sd)
                    out.append(dict(lid=lid, loc=loc, w=(s, e), wname=wname, var=var,
                                    assertion={"kind": "threshold", "op": "gt", "value": round(base, 2)},
                                    p=p, why=f"15y baseline {base:.2f}C, trend {b * 10:+.2f}C/decade -> {pred:.2f}C (sd {sd:.2f})"))
                    if var == "temp_mean_c":
                        yx = ys[-1]
                        pw = 1.0 - phi((yx - pred) / sd)
                        out.append(dict(lid=lid, loc=loc, w=(s, e), wname=wname, var="warmer_than_year_x",
                                        assertion={"kind": "boolean", "value": True},
                                        p=pw, why=f"{x} was {yx:.2f}C vs trend forecast {pred:.2f}C (sd {sd:.2f}); regression to rising trend"))
                elif var == "hot_days_ge_30c":
                    if base < 3.0 or base > 0.9 * ndays:
                        continue
                    # counts scale with warming: ~ hot-day sensitivity from the series itself, prior +2%/yr
                    pred, sd, b = trend_forecast(yrs, ys, target_year, 0.02 * base)
                    thr = math.floor(base)
                    p = 1.0 - phi((thr + 0.5 - pred) / sd)
                    out.append(dict(lid=lid, loc=loc, w=(s, e), wname=wname, var=var,
                                    assertion={"kind": "threshold", "op": "gt", "value": thr},
                                    p=p, why=f"15y mean {base:.1f} hot days, trend {b * 10:+.1f}/decade -> {pred:.1f} (sd {sd:.1f})"))
                else:  # frost days
                    if base < 3.0 or base > 0.9 * ndays:
                        continue
                    pred, sd, b = trend_forecast(yrs, ys, target_year, -0.02 * base)
                    thr = math.ceil(base)
                    p = phi((thr - 0.5 - pred) / sd)
                    out.append(dict(lid=lid, loc=loc, w=(s, e), wname=wname, var=var,
                                    assertion={"kind": "threshold", "op": "lt", "value": thr},
                                    p=p, why=f"15y mean {base:.1f} frost days, trend {b * 10:+.1f}/decade -> {pred:.1f} (sd {sd:.1f})"))
    return out


def select(cands, x):
    seen = set()
    claims = []
    # Rank by probability; prefer broad windows on ties so they survive the cap.
    cands.sort(key=lambda c: (-c["p"], c["wname"] != "year", c["lid"], c["wname"], c["var"]))
    per_loc = defaultdict(int)
    for c in cands:
        if c["p"] < MIN_CONF:
            break
        key = (c["lid"], c["w"], c["var"])
        if key in seen or per_loc[c["lid"]] >= 12:
            continue
        seen.add(key)
        per_loc[c["lid"]] += 1
        y1 = x + 1
        end = c["w"][1]
        if end == "02-28" and (y1 % 4 == 0 and (y1 % 100 != 0 or y1 % 400 == 0)):
            end = "02-29"
        claims.append({
            "claim_id": f"p5_{len(claims) + 1:03d}",
            "target": {
                "location_id": c["lid"], "lat": c["loc"]["lat"], "lon": c["loc"]["lon"],
                "window_start": f"{y1}-{c['w'][0]}", "window_end": f"{y1}-{end}",
                "variable": c["var"],
            },
            "assertion": c["assertion"],
            # Model p runs hot (interannual noise is fat-tailed); shrink toward 0.5 for reported confidence.
            "confidence": round(0.5 + 0.7 * (min(c["p"], 0.99) - 0.5), 3),
            "rationale": f"Secular warming extrapolation ({c['wname']}): {c['why']}",
        })
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    max_date = f"{a.year_x}-12-31"
    man_path = os.path.join(a.data_dir, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if man.get("max_date", max_date) < max_date:
            max_date = man["max_date"]
    locs = load_daily(a.data_dir, max_date)
    claims = select(candidates(locs, a.year_x), a.year_x)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(claims, f, indent=1)
    print(f"predict5: wrote {len(claims)} claims for {a.year_x + 1} to {a.out}")


if __name__ == "__main__":
    main()
