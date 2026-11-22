#!/usr/bin/env python3
"""predict4: the Adversarial Referee. Calibrated forecasts placed on rivals' focal lines.

    python3 predict4/run.py --year-x <X> --data-dir <data_dir> --out <claims_json>

A point is only won in a contradiction, so claims go where rival predictors draw their lines:
  * Seasonal/annual means: threshold predictors cut at the trailing climatology (the 16-year
    mean of max(1980, X-15)..X, or the 15-year mean of X-14..X) and lean warm. predict4 states the
    side of the *lower* (or upper) of those lines that its own model favours, so a disagreement
    is disjoint with every rival on that line.
  * Day counts: rivals cut at floor/ceil of the 16-year mean; predict4 takes the other side when
    its model says so.
  * Booleans (warmer/wetter than year X) on every window rivals use; any disagreement counts.

The model behind every probability (per location, window, variable; all inputs dated <= X-12-31):
  mean = climatology + trend shrunk toward a weak prior
         + AR(1) carry-over of year X's residual, shrunk toward zero
  spread = residual SD inflated for parameter uncertainty.
Calibration was checked by a walk-forward backtest (X=1988..1994 against the real next year):
boolean probabilities match observed frequencies per decile, so they are stated as is; line
probabilities were overconfident (selection on noise), so they are shrunk halfway to 0.5.
A Nino3.4 regression term made forecasts worse at <= 16 years of history and is not used.
Stated confidence is capped at 0.90. Runs offline, stdlib only.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict
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

TEMP_VARS = {"temp_mean_c", "temp_mean_max_c", "temp_mean_min_c"}
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
PRIOR_TREND = {v: 0.025 for v in TEMP_VARS}
TREND_PRIOR_STRENGTH = 20.0   # prior trend counts as this many years' worth of evidence
AR_PRIOR_STRENGTH = 10.0      # shrink lag-1 autocorrelation toward 0
AR_CAP = 0.5
SD_INFLATION = 1.15
LINE_SHRINK = 0.5             # line probabilities move this fraction of the way from 0.5
MIN_PROB = 0.55              # lines (after shrinking)
MIN_BOOL_PROB = 0.60
MAX_CONF = 0.90

# Windows rivals use, as (start month-day, end month-day); day "end" is the month's last day.
WINDOWS = {
    "year": ((1, 1), (12, 31)), "jja": ((6, 1), (8, 31)), "mam": ((3, 1), (5, 31)),
    "son": ((9, 1), (11, 30)), "jf": ((1, 1), (2, "end")), "jfm": ((1, 1), (3, 31)),
    "ond": ((10, 1), (12, 31)),
    **{f"m{m:02d}": ((m, 1), (m, "end")) for m in range(1, 13)},
}
# How many rivals tend to claim each window (from reading their code); used only for ranking.
WINDOW_WEIGHT = {"year": 1.0, "jja": 0.85, "mam": 0.6, "son": 0.6, "jf": 0.55, "jfm": 0.4, "ond": 0.35,
                 **{f"m{m:02d}": 0.3 for m in range(1, 13)}}
WINDOW_WEIGHT["m01"] = WINDOW_WEIGHT["m07"] = 0.45
THRESHOLD_JOBS = [  # (variable, windows)
    ("temp_mean_c", ["year", "jja", "mam", "son", "jf", "m01", "m07"]),
    ("temp_mean_max_c", ["year", "jja"]),
    ("temp_mean_min_c", ["year", "jja"]),
    ("hot_days_ge_30c", ["year", "jja"]),
    ("frost_days_lt_0c", ["year", "jf"]),
]
BOOLEAN_JOBS = [
    ("warmer_than_year_x", "temp_mean_c", list(WINDOWS)),
    ("wetter_than_year_x", "precip_sum_mm", list(WINDOWS)),
]
VAR_WEIGHT = {"temp_mean_c": 1.0, "temp_mean_max_c": 0.7, "temp_mean_min_c": 0.7,
              "hot_days_ge_30c": 0.6, "frost_days_lt_0c": 0.6,
              "warmer_than_year_x": 0.9, "wetter_than_year_x": 0.8}


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


def window_dates(year: int, wname: str):
    (sm, sd), (em, ed) = WINDOWS[wname]
    if ed == "end":
        ed = calendar.monthrange(year, em)[1]
    return dt.date(year, sm, sd), dt.date(year, em, ed)


def window_values(days: dict, year: int, wname: str, field: str):
    """Daily values of `field` over window `wname` in `year`; None if any day is missing."""
    d, end = window_dates(year, wname)
    out = []
    while d <= end:
        rec = days.get(d.isoformat())
        if rec is None or rec.get(field) is None:
            return None
        out.append(rec[field])
        d += dt.timedelta(days=1)
    return out


AGG = {
    "temp_mean_c": ("temperature_2m_mean", lambda v: sum(v) / len(v)),
    "temp_mean_max_c": ("temperature_2m_max", lambda v: sum(v) / len(v)),
    "temp_mean_min_c": ("temperature_2m_min", lambda v: sum(v) / len(v)),
    "precip_sum_mm": ("precipitation_sum", sum),
    "hot_days_ge_30c": ("temperature_2m_max", lambda v: sum(x >= 30.0 for x in v)),
    "frost_days_lt_0c": ("temperature_2m_min", lambda v: sum(x < 0.0 for x in v)),
}


def variable_value(variable, days, year, wname):
    field, fn = AGG[variable]
    vals = window_values(days, year, wname, field)
    return None if vals is None else fn(vals)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def forecast(series: dict, year_x: int, variable: str):
    """series: {year: value}. Returns (mean, sd, n) for X+1, or None."""
    years = sorted(series)
    n = len(years)
    if n < MIN_YEARS or year_x not in series:
        return None
    ys = [series[y] for y in years]
    tbar, ybar = sum(years) / n, sum(ys) / n
    sxx = sum((t - tbar) ** 2 for t in years)
    b_ols = sum((t - tbar) * (y - ybar) for t, y in zip(years, ys)) / sxx if sxx else 0.0
    w = n / (n + TREND_PRIOR_STRENGTH)
    b = w * b_ols + (1 - w) * PRIOR_TREND.get(variable, 0.0)
    resid = [y - (ybar + b * (t - tbar)) for t, y in zip(years, ys)]
    rx = resid[years.index(year_x)]
    mean = ybar + b * (year_x + 1 - tbar)


    sd = statistics.pstdev(resid) * math.sqrt(n / max(n - 2, 1))
    pairs = [(resid[i - 1], resid[i]) for i in range(1, n) if years[i] - years[i - 1] == 1]
    den = sum(a * a for a, _ in pairs)
    if len(pairs) >= 4 and den > 0:
        r1 = sum(a * c for a, c in pairs) / den
        phi = max(0.0, min(AR_CAP, r1 * len(pairs) / (len(pairs) + AR_PRIOR_STRENGTH)))
        mean += phi * rx
    sd = sd * math.sqrt(1 + 1 / n) * SD_INFLATION
    if variable not in TEMP_VARS:
        mean = max(mean, 0.0)
    return mean, max(sd, 1e-6), n


def p_above(mean, sd, t):
    return 1.0 - norm_cdf((t - mean) / sd)


def p_count_ge(mean, sd, k):
    """P(integer count >= k) with a continuity correction."""
    return p_above(mean, sd, k - 0.5)


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------

def focal_lines(series: dict, year_x: int):
    """Trailing means rivals cut at: 16-year (max(1980, X-15)..X) and 15-year (X-14..X)."""
    out = []
    for first in (max(1980, year_x - 15), year_x - 14):
        vals = [v for y, v in series.items() if first <= y <= year_x]
        if len(vals) >= MIN_YEARS:
            out.append(sum(vals) / len(vals))
    return out


def threshold_candidate(variable, mean, sd, lines):
    """The side of the rivals' lines this model favours, as (assertion, p, note) or None."""
    if not lines:
        return None
    if variable in COUNT_VARS:   # rival lines: hot gt floor(mean16), frost lt ceil(mean16)
        k = math.floor(lines[0]) if variable == "hot_days_ge_30c" else math.ceil(lines[0])
        p_hi = p_count_ge(mean, sd, k + 1) if variable == "hot_days_ge_30c" else p_count_ge(mean, sd, k)
        if variable == "hot_days_ge_30c":
            a = ({"kind": "threshold", "op": "gt", "value": k}, p_hi) if p_hi >= 0.5 else \
                ({"kind": "threshold", "op": "le", "value": k}, 1 - p_hi)
        else:
            a = ({"kind": "threshold", "op": "ge", "value": k}, p_hi) if p_hi >= 0.5 else \
                ({"kind": "threshold", "op": "lt", "value": k}, 1 - p_hi)
        return a[0], a[1], f"rival line at {k}"
    # Continuous: "le" at the lowest line is disjoint with every rival "gt" on any line;
    # "gt" at the highest line is disjoint with every rival "lt"/"le" on any line.
    lo, hi = round(min(lines), 2), round(max(lines), 2)
    p_le, p_gt = 1 - p_above(mean, sd, lo), p_above(mean, sd, hi)
    if p_le >= p_gt:
        return {"kind": "threshold", "op": "le", "value": lo}, p_le, f"lowest rival line {lo}"
    return {"kind": "threshold", "op": "gt", "value": hi}, p_gt, f"highest rival line {hi}"


def build_claims(year_x: int, daily, coords):
    y1 = year_x + 1
    candidates = []

    def add(loc, wname, variable, assertion, p, rationale):
        lat, lon = claim_coords(loc, coords.get(loc))
        ws, we = window_dates(y1, wname)
        candidates.append({
            "target": {"location_id": loc, "lat": lat, "lon": lon, "window_start": ws.isoformat(),
                       "window_end": we.isoformat(), "variable": variable},
            "assertion": assertion, "confidence": round(min(max(p, 0.5), MAX_CONF), 3),
            "rationale": rationale,
            "_rank": (p - 0.5) * WINDOW_WEIGHT[wname] * VAR_WEIGHT[variable],
        })

    for loc in sorted(daily):
        if claim_coords(loc, coords.get(loc)) is None:
            continue
        days = daily[loc]
        years = sorted({int(d[:4]) for d in days})
        cache = {}

        def series(variable, wname):
            if (variable, wname) not in cache:
                s = {}
                for y in years:
                    v = variable_value(variable, days, y, wname)
                    if v is not None:
                        s[y] = v
                cache[(variable, wname)] = s
            return cache[(variable, wname)]

        def model(variable, wname):
            return forecast(series(variable, wname), year_x, variable)

        for variable, wnames in THRESHOLD_JOBS:
            for wname in wnames:
                s = series(variable, wname)
                f = model(variable, wname)
                if f is None:
                    continue
                if variable in COUNT_VARS and not (3 <= statistics.mean(s.values()) <= 0.9 * len(
                        window_values(days, year_x, wname, "temperature_2m_mean") or [0])):
                    continue  # rivals skip near-zero and saturated counts
                mean, sd, n = f
                c = threshold_candidate(variable, mean, sd, focal_lines(s, year_x))
                if c is None:
                    continue
                assertion, p_model, note = c
                p = 0.5 + LINE_SHRINK * (p_model - 0.5)
                if p < MIN_PROB:
                    continue
                add(loc, wname, variable, assertion, p,
                    f"forecast {mean:.2f}±{sd:.2f} ({n} yrs, shrunk trend+AR1) vs {note}: "
                    f"model P={p_model:.2f}, backtest-calibrated P={p:.2f}")

        for bvar, base, wnames in BOOLEAN_JOBS:
            for wname in wnames:
                s = series(base, wname)
                f = model(base, wname)
                if f is None:
                    continue
                mean, sd, n = f
                p_up = p_above(mean, sd, s[year_x])
                val = p_up >= 0.5
                p = p_up if val else 1 - p_up
                if p < MIN_BOOL_PROB:
                    continue
                add(loc, wname, bvar, {"kind": "boolean", "value": val}, p,
                    f"year X {base}={s[year_x]:.2f} vs forecast {mean:.2f}±{sd:.2f}: "
                    f"P({'higher' if val else 'not higher'})={p:.2f}")

    candidates.sort(key=lambda c: (-c["_rank"], c["target"]["location_id"], c["target"]["window_start"],
                                   c["target"]["window_end"], c["target"]["variable"]))
    claims = []
    for i, c in enumerate(candidates[:MAX_CLAIMS], 1):
        claims.append({"claim_id": f"p4_{i:03d}", "target": c["target"], "assertion": c["assertion"],
                       "confidence": c["confidence"], "rationale": c["rationale"]})
    return claims


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
