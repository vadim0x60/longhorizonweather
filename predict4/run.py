#!/usr/bin/env python3
"""predict4: calibrated base rates with Bayesian shrinkage and anomaly reversion.

    python3 predict4/run.py --year-x <X> --data-dir <data_dir> --out <claims_json>

For each location and window, the year-by-year history of a variable (from
daily_history.csv, all dates <= X-12-31) gives a predictive distribution for X+1:

  mean   = climatology + shrunken linear trend + shrunken AR(1) carry-over of year X's anomaly
  spread = residual standard deviation, inflated for parameter uncertainty

The trend is shrunk toward a weak prior (slow warming for temperatures, none otherwise) and
the carry-over toward zero, so one hot or wet year X moves the forecast only a little.
Claims sit where this model disagrees with naive persistence (repeating year X), with the
threshold halfway between the two, so the claim holds with probability > 0.5 under the model
and contradicts a rival that extrapolates the last year. Each claim states its model probability.

Runs offline, Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CATALOG = {
    "london_uk": (51.5074, -0.1278), "new_york_us": (40.7128, -74.0060),
    "tokyo_jp": (35.6762, 139.6503), "sydney_au": (-33.8688, 151.2093),
    "cairo_eg": (30.0444, 31.2357), "mumbai_in": (19.0760, 72.8777),
    "sao_paulo_br": (-23.5505, -46.6333), "reykjavik_is": (64.1466, -21.9426),
    "nairobi_ke": (-1.2921, 36.8219), "phoenix_us": (33.4484, -112.0740),
    "singapore_sg": (1.3521, 103.8198), "chicago_us": (41.8781, -87.6298),
    "moscow_ru": (55.7558, 37.6173), "cape_town_za": (-33.9249, 18.4241),
    "lima_pe": (-12.0464, -77.0428), "bangkok_th": (13.7563, 100.5018),
    "anchorage_us": (61.2181, -149.9003), "madrid_es": (40.4168, -3.7038),
    "seoul_kr": (37.5665, 126.9780), "toronto_ca": (43.6532, -79.3832),
}
FIELDS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max"]
MAX_CLAIMS = 150
MIN_YEARS = 6

# Prior trend per year and its strength in "pseudo-years of evidence".
PRIOR_TREND = {"temp_mean_c": 0.025, "temp_mean_max_c": 0.025, "temp_mean_min_c": 0.025}
TREND_PRIOR_STRENGTH = 20.0
AR_PRIOR_STRENGTH = 10.0   # shrink the lag-1 autocorrelation toward 0
AR_CAP = 0.5
SD_INFLATION = 1.15        # calibration guard: rivals' overconfidence is our edge, not ours
MIN_PROB = 0.58            # only claim what the model believes clearly
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_daily(data_dir: Path, year_x: int):
    """{location_id: {date: {field: float|None}}} and {location_id: (lat, lon)}, dates <= X."""
    daily, coords = defaultdict(dict), {}
    cutoff = f"{year_x}-12-31"
    for name in ("daily_history.csv", "daily_year_x.csv"):
        p = data_dir / name
        if not p.exists():
            continue
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                d = row.get("date", "")
                if not d or d > cutoff:
                    continue  # never read anything after year X
                loc = row["location_id"]
                rec = {}
                for f in FIELDS:
                    v = row.get(f, "")
                    try:
                        rec[f] = float(v) if v not in ("", None) else None
                    except ValueError:
                        rec[f] = None
                    if rec[f] is not None and not math.isfinite(rec[f]):
                        rec[f] = None
                daily[loc][d] = rec
                if loc not in coords:
                    try:
                        coords[loc] = (float(row["lat"]), float(row["lon"]))
                    except (KeyError, ValueError):
                        pass
    return daily, coords


def claim_coords(loc: str, csv_coords):
    if loc in CATALOG:
        return CATALOG[loc]
    if csv_coords is None:
        return None
    lat, lon = round(csv_coords[0], 2), round(csv_coords[1], 2)
    return (lat, lon) if loc == f"grid_{lat:.2f}_{lon:.2f}" else None


def window_values(days: dict, year: int, ms: tuple, me: tuple, field: str):
    """Daily values of `field` for month-day window ms..me in `year`; None if incomplete."""
    try:
        d, end = dt.date(year, *ms), dt.date(year, *me)
    except ValueError:
        return None
    out = []
    while d <= end:
        rec = days.get(d.isoformat())
        if rec is None or rec.get(field) is None:
            return None
        out.append((d, rec[field]))
        d += dt.timedelta(days=1)
    return out


def variable_value(variable, days, year, ms, me):
    field, fn = {
        "temp_mean_c": ("temperature_2m_mean", lambda v: sum(v) / len(v)),
        "temp_mean_max_c": ("temperature_2m_max", lambda v: sum(v) / len(v)),
        "temp_mean_min_c": ("temperature_2m_min", lambda v: sum(v) / len(v)),
        "temp_max_c": ("temperature_2m_max", max),
        "temp_min_c": ("temperature_2m_min", min),
        "precip_sum_mm": ("precipitation_sum", sum),
        "precip_max_day_mm": ("precipitation_sum", max),
        "wind_max_kmh": ("wind_speed_10m_max", max),
        "hot_days_ge_30c": ("temperature_2m_max", lambda v: sum(x >= 30.0 for x in v)),
        "frost_days_lt_0c": ("temperature_2m_min", lambda v: sum(x < 0.0 for x in v)),
        "rainy_days_ge_1mm": ("precipitation_sum", lambda v: sum(x >= 1.0 for x in v)),
    }[variable]
    vals = window_values(days, year, ms, me, field)
    return None if vals is None else fn([v for _, v in vals])


def hottest_month(days, year):
    vals = window_values(days, year, (1, 1), (12, 31), "temperature_2m_mean")
    if vals is None:
        return None
    by = defaultdict(list)
    for d, v in vals:
        by[d.month].append(v)
    means = {m: sum(v) / len(v) for m, v in by.items()}
    return f"{max(means, key=means.get):02d}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def forecast(series: dict, year_x: int, variable: str):
    """series: {year: value}. Returns (mean, sd, n) for year X+1, or None."""
    years = sorted(series)
    n = len(years)
    if n < MIN_YEARS or year_x not in series:
        return None
    ys = [series[y] for y in years]
    tbar, ybar = sum(years) / n, sum(ys) / n
    sxx = sum((t - tbar) ** 2 for t in years)
    b_ols = sum((t - tbar) * (y - ybar) for t, y in zip(years, ys)) / sxx if sxx else 0.0
    prior = PRIOR_TREND.get(variable, 0.0)
    # Precision-weighted shrinkage: OLS slope variance ~ 1/sxx, prior ~ 1/(strength * mean sxx per year).
    w = sxx / (sxx + TREND_PRIOR_STRENGTH * sxx / n)
    b = w * b_ols + (1 - w) * prior
    resid = [y - (ybar + b * (t - tbar)) for t, y in zip(years, ys)]
    sd = statistics.pstdev(resid) * math.sqrt(n / max(n - 2, 1))
    pairs = [(resid[i - 1], resid[i]) for i in range(1, n) if years[i] - years[i - 1] == 1]
    phi = 0.0
    if len(pairs) >= 4 and sd > 0:
        r1 = sum(a * c for a, c in pairs) / sum(a * a for a, _ in pairs) if any(a for a, _ in pairs) else 0.0
        phi = max(0.0, min(AR_CAP, r1 * len(pairs) / (len(pairs) + AR_PRIOR_STRENGTH)))
    mean = ybar + b * (year_x + 1 - tbar) + phi * resid[years.index(year_x)]
    sd = sd * math.sqrt(1 + 1 / n) * SD_INFLATION
    if variable in COUNT_VARS or variable.startswith("precip") or variable == "wind_max_kmh":
        mean = max(mean, 0.0)
    return mean, max(sd, 1e-6), n


def prob_above(mean, sd, t, variable, strict):
    """P(value > t) (or >= t), with a continuity correction for integer counts."""
    if variable in COUNT_VARS:
        t = (math.floor(t) + 0.5) if strict else (math.ceil(t) - 0.5)
    return 1.0 - norm_cdf((t - mean) / sd)


def threshold_claim(variable, mean, sd, persist, clim):
    """Pick (assertion, probability, anchor description) or None."""
    # Anchor: what a naive rival says; the threshold sits halfway, our side toward the model mean.
    if abs(mean - persist) >= 0.3 * sd:
        anchor, label = persist, "persistence of year X"
    elif abs(mean - clim) >= 0.3 * sd:
        anchor, label = clim, "flat climatology"
    else:
        return None
    t = (mean + anchor) / 2
    up = mean > anchor
    if variable in COUNT_VARS:
        t = math.floor(t) if up else math.ceil(t)
        if (up and t >= mean) or (not up and t <= mean):
            return None
        op = "gt" if up else "lt"
        p = prob_above(mean, sd, t, variable, strict=True) if up else 1 - prob_above(mean, sd, t, variable, strict=False)
    else:
        t = round(t, 2)
        op = "gt" if up else "le"
        p = prob_above(mean, sd, t, variable, True) if up else 1 - prob_above(mean, sd, t, variable, True)
    return {"kind": "threshold", "op": op, "value": t}, p, label, anchor


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------

WINDOWS = {
    "year": ((1, 1), (12, 31)),
    "jja": ((6, 1), (8, 31)),
    "jan_feb": ((1, 1), (2, 28)),
}


def build_claims(year_x: int, daily, coords):
    y1 = year_x + 1
    candidates = []

    def add(loc, wname, variable, assertion, p, rationale):
        ms, me = WINDOWS[wname]
        lat, lon = claim_coords(loc, coords.get(loc))
        candidates.append({
            "target": {"location_id": loc, "lat": lat, "lon": lon,
                       "window_start": dt.date(y1, *ms).isoformat(),
                       "window_end": dt.date(y1, *me).isoformat(), "variable": variable},
            "assertion": assertion, "confidence": round(min(max(p, 0.5), 0.90), 3),
            "rationale": rationale,
        })

    for loc in sorted(daily):
        if claim_coords(loc, coords.get(loc)) is None:
            continue
        days = daily[loc]
        years = sorted({int(d[:4]) for d in days})
        cache = {}

        def series(variable, wname):
            if (variable, wname) not in cache:
                ms, me = WINDOWS[wname]
                s = {}
                for y in years:
                    v = variable_value(variable, days, y, ms, me)
                    if v is not None:
                        s[y] = v
                cache[(variable, wname)] = s
            return cache[(variable, wname)]

        jobs = [("temp_mean_c", "year"), ("precip_sum_mm", "year"), ("temp_mean_c", "jja"),
                ("temp_mean_c", "jan_feb"), ("precip_sum_mm", "jja")]
        hot = series("hot_days_ge_30c", "year")
        frost = series("frost_days_lt_0c", "year")
        if hot and statistics.mean(hot.values()) >= 5:
            jobs.append(("hot_days_ge_30c", "year"))
        if frost and statistics.mean(frost.values()) >= 5:
            jobs.append(("frost_days_lt_0c", "year"))
        jobs.append(("rainy_days_ge_1mm", "year"))

        for variable, wname in jobs:
            s = series(variable, wname)
            f = forecast(s, year_x, variable)
            if f is None:
                continue
            mean, sd, n = f
            clim = statistics.mean(s.values())
            tc = threshold_claim(variable, mean, sd, s[year_x], clim)
            if tc is None:
                continue
            assertion, p, label, anchor = tc
            if p < MIN_PROB:
                continue
            add(loc, wname, variable, assertion, p,
                f"{n}-yr shrunk trend+AR(1) forecast {mean:.2f}±{sd:.2f} vs {label} {anchor:.2f}; "
                f"threshold at midpoint, model P={p:.2f}")

        # Booleans: is X+1 warmer / wetter than year X? Uses the same predictive distributions.
        for bvar, base in (("warmer_than_year_x", "temp_mean_c"), ("wetter_than_year_x", "precip_sum_mm")):
            for wname in ("year", "jja"):
                s = series(base, wname)
                f = forecast(s, year_x, base)
                if f is None:
                    continue
                mean, sd, n = f
                p_up = prob_above(mean, sd, s[year_x], base, True)
                val = p_up >= 0.5
                p = p_up if val else 1 - p_up
                if p < MIN_PROB:
                    continue
                add(loc, wname, bvar, {"kind": "boolean", "value": val}, p,
                    f"year X {base}={s[year_x]:.2f} vs forecast {mean:.2f}±{sd:.2f}: "
                    f"P({'higher' if val else 'not higher'})={p:.2f} (anomaly reversion)")

        # Categorical: hottest month of the full year, by Laplace-smoothed base rate.
        hm = [h for h in (hottest_month(days, y) for y in years) if h]
        if len(hm) >= MIN_YEARS:
            month, cnt = Counter(hm).most_common(1)[0]
            p = (cnt + 1) / (len(hm) + 12)
            if p >= MIN_PROB:
                add(loc, "year", "hottest_month", {"kind": "categorical", "value": month}, p,
                    f"hottest month was {month} in {cnt}/{len(hm)} years; Laplace-smoothed P={p:.2f}")

    # Keep the most confident claims within the cap; one claim per target by construction.
    candidates.sort(key=lambda c: -c["confidence"])
    claims = candidates[:MAX_CLAIMS]
    for i, c in enumerate(claims, 1):
        c["claim_id"] = f"p4_{i:03d}"
    return [{"claim_id": c["claim_id"], **{k: c[k] for k in ("target", "assertion", "confidence", "rationale")}}
            for c in claims]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    daily, coords = load_daily(args.data_dir, args.year_x)
    claims = build_claims(args.year_x, daily, coords)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(claims, indent=2))
    print(f"predict4: wrote {len(claims)} claims for {args.year_x + 1} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
