#!/usr/bin/env python3
"""Contradiction detector and ground-truth scorer for the long-horizon weather arena.

    python3 verify/run.py --year-x <X> --claims-dir <claims_dir> --out <report_json>

Pipeline (see CONTRACTS.md):
  1. Load predict1.json .. predict10.json (missing, empty, or malformed files mean no claims).
  2. Validate each claim and enforce anti-spam limits, in file order:
     invalid claims are rejected; a repeated (location_id, window_start, window_end, variable)
     key keeps only its first valid claim; at most 150 claims are accepted per predictor.
  3. For every pair predict{i}, predict{j} (i < j) with claims on the same canonical target
     whose assertions cannot both be true, record a contradiction.
  4. Fetch daily ground truth for year X+1 (and X when needed) from the Open-Meteo archive
     API, compute the target variable, and score: the true side gets +1, the false side -1;
     both false or unresolvable is 0 / 0.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path

PREDICTORS = [f"predict{i}" for i in range(1, 11)]
MAX_CLAIMS_PER_PREDICTOR = 150

CATALOG = {
    "london_uk": (51.5074, -0.1278),
    "new_york_us": (40.7128, -74.0060),
    "tokyo_jp": (35.6762, 139.6503),
    "sydney_au": (-33.8688, 151.2093),
    "cairo_eg": (30.0444, 31.2357),
    "mumbai_in": (19.0760, 72.8777),
    "sao_paulo_br": (-23.5505, -46.6333),
    "reykjavik_is": (64.1466, -21.9426),
    "nairobi_ke": (-1.2921, 36.8219),
    "phoenix_us": (33.4484, -112.0740),
    "singapore_sg": (1.3521, 103.8198),
    "chicago_us": (41.8781, -87.6298),
    "moscow_ru": (55.7558, 37.6173),
    "cape_town_za": (-33.9249, 18.4241),
    "lima_pe": (-12.0464, -77.0428),
    "bangkok_th": (13.7563, 100.5018),
    "anchorage_us": (61.2181, -149.9003),
    "madrid_es": (40.4168, -3.7038),
    "seoul_kr": (37.5665, 126.9780),
    "toronto_ca": (43.6532, -79.3832),
}
# A catalog claim's lat/lon must lie within this many degrees of the catalog coordinates.
CATALOG_COORD_TOLERANCE = 0.05
GRID_RE = re.compile(r"^grid_(-?\d{1,3}\.\d{2})_(-?\d{1,3}\.\d{2})$")

NUMERIC_VARS = {
    "temp_mean_c", "temp_max_c", "temp_min_c", "temp_mean_max_c", "temp_mean_min_c",
    "precip_sum_mm", "precip_max_day_mm", "wind_max_kmh",
    "hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm",
}
COUNT_VARS = {"hot_days_ge_30c", "frost_days_lt_0c", "rainy_days_ge_1mm"}
CATEGORICAL_VARS = {"wettest_month", "hottest_month"}
BOOLEAN_VARS = {"warmer_than_year_x", "wetter_than_year_x"}
ALL_VARS = NUMERIC_VARS | CATEGORICAL_VARS | BOOLEAN_VARS
MONTHS = {f"{m:02d}" for m in range(1, 13)}

DAILY_FIELDS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max",
]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Aggregates are rounded before comparison so float noise such as 17.499999999 vs 17.5
# cannot flip a boundary; Open-Meteo daily values carry at most two decimals.
AGG_DECIMALS = 6


class ClaimError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Claim loading and validation
# ---------------------------------------------------------------------------

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _parse_date(s, field: str) -> dt.date:
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ClaimError(f"{field} must be an ISO date YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise ClaimError(f"{field} is not a valid date: {e}") from None


def canonical_location(location_id, lat, lon) -> tuple[float, float]:
    """Return the coordinates verify uses for a location_id, checking the claim's lat/lon."""
    if not isinstance(location_id, str):
        raise ClaimError("location_id must be a string")
    if not (_is_number(lat) and _is_number(lon)):
        raise ClaimError("lat and lon must be finite numbers")
    if location_id in CATALOG:
        clat, clon = CATALOG[location_id]
        if abs(lat - clat) > CATALOG_COORD_TOLERANCE or abs(lon - clon) > CATALOG_COORD_TOLERANCE:
            raise ClaimError(f"lat/lon do not match catalog coordinates of {location_id}")
        return clat, clon
    m = GRID_RE.match(location_id)
    if not m:
        raise ClaimError("location_id is neither a catalog id nor grid_{lat:.2f}_{lon:.2f}")
    glat, glon = float(m.group(1)), float(m.group(2))
    if not (-90 <= glat <= 90 and -180 <= glon <= 180):
        raise ClaimError("grid coordinates out of range")
    if f"grid_{glat:.2f}_{glon:.2f}" != location_id:
        raise ClaimError("grid location_id is not in canonical grid_{lat:.2f}_{lon:.2f} form")
    if abs(lat - glat) > 0.01 or abs(lon - glon) > 0.01:
        raise ClaimError("lat/lon do not match the grid location_id")
    return glat, glon


def normalize_assertion(variable: str, a) -> dict:
    if not isinstance(a, dict):
        raise ClaimError("assertion must be an object")
    kind = a.get("kind")
    if variable in NUMERIC_VARS:
        if kind == "threshold":
            op, value = a.get("op"), a.get("value")
            if op not in ("gt", "ge", "lt", "le"):
                raise ClaimError("threshold op must be gt, ge, lt or le")
            if not _is_number(value):
                raise ClaimError("threshold value must be a finite number")
            return {"kind": "threshold", "op": op, "value": value}
        if kind == "interval":
            low, high = a.get("low"), a.get("high")
            if not (_is_number(low) and _is_number(high)):
                raise ClaimError("interval low/high must be finite numbers")
            if low > high:
                raise ClaimError("interval low must be <= high")
            return {"kind": "interval", "low": low, "high": high}
        raise ClaimError(f"{variable} needs a threshold or interval assertion")
    if variable in BOOLEAN_VARS:
        if kind != "boolean" or not isinstance(a.get("value"), bool):
            raise ClaimError(f"{variable} needs a boolean assertion with a true/false value")
        return {"kind": "boolean", "value": a["value"]}
    if variable in CATEGORICAL_VARS:
        value = a.get("value")
        if kind != "categorical" or not isinstance(value, str):
            raise ClaimError(f"{variable} needs a categorical assertion with a string value")
        if re.fullmatch(r"\d{1,2}", value) and f"{int(value):02d}" in MONTHS:
            value = f"{int(value):02d}"
        if value not in MONTHS:
            raise ClaimError(f"{variable} value must be a month '01'..'12'")
        return {"kind": "categorical", "value": value}
    raise ClaimError(f"unknown variable {variable!r}")


def validate_claim(raw, year_x: int) -> dict:
    """Return a normalized claim or raise ClaimError."""
    if not isinstance(raw, dict):
        raise ClaimError("claim must be an object")
    t = raw.get("target")
    if not isinstance(t, dict):
        raise ClaimError("target must be an object")
    variable = t.get("variable")
    if variable not in ALL_VARS:
        raise ClaimError(f"unknown variable {variable!r}")
    lat, lon = canonical_location(t.get("location_id"), t.get("lat"), t.get("lon"))
    ws = _parse_date(t.get("window_start"), "window_start")
    we = _parse_date(t.get("window_end"), "window_end")
    y1 = year_x + 1
    if not (dt.date(y1, 1, 1) <= ws <= we <= dt.date(y1, 12, 31)):
        raise ClaimError(f"window must satisfy {y1}-01-01 <= window_start <= window_end <= {y1}-12-31")
    assertion = normalize_assertion(variable, raw.get("assertion"))
    return {
        "claim_id": raw.get("claim_id") if isinstance(raw.get("claim_id"), str) else None,
        "target": {
            "location_id": t["location_id"], "lat": lat, "lon": lon,
            "window_start": ws.isoformat(), "window_end": we.isoformat(), "variable": variable,
        },
        "assertion": assertion,
        "confidence": raw.get("confidence") if _is_number(raw.get("confidence")) else None,
    }


def target_key(target: dict) -> tuple:
    return (target["location_id"], target["window_start"], target["window_end"], target["variable"])


def load_predictor_file(path: Path) -> tuple[list, str | None]:
    """Return (raw claims, load error). Missing or empty files are simply no claims."""
    if not path.exists():
        return [], "missing"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return [], "empty"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return [], f"invalid JSON: {e}"
    if not isinstance(data, list):
        return [], "top-level JSON value is not an array"
    return data, None


def filter_claims(predictor: str, raw_claims: list, year_x: int) -> tuple[list, list]:
    """Validate and apply anti-spam rules in file order. Returns (accepted, rejected)."""
    accepted, rejected, seen = [], [], set()
    for idx, raw in enumerate(raw_claims):
        cid = raw.get("claim_id") if isinstance(raw, dict) and isinstance(raw.get("claim_id"), str) else None
        cid = cid or f"{predictor}#{idx}"
        try:
            claim = validate_claim(raw, year_x)
        except ClaimError as e:
            rejected.append({"claim_id": cid, "index": idx, "reason": f"invalid: {e}"})
            continue
        claim["claim_id"] = cid
        key = target_key(claim["target"])
        if key in seen:
            rejected.append({"claim_id": cid, "index": idx, "reason": "duplicate target key"})
            continue
        if len(accepted) >= MAX_CLAIMS_PER_PREDICTOR:
            rejected.append({"claim_id": cid, "index": idx,
                             "reason": f"exceeds {MAX_CLAIMS_PER_PREDICTOR} claims per predictor"})
            continue
        seen.add(key)
        claim["predictor"] = predictor
        accepted.append(claim)
    return accepted, rejected


# ---------------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------------

def _as_interval(variable: str, a: dict):
    """Convert a numeric assertion to (lo, lo_closed, hi, hi_closed) over the variable's domain.

    Count variables are integers, so their sets are reduced to closed integer ranges;
    e.g. "gt 10" and "lt 11" cannot both hold for a day count.
    Values are converted to exact Fractions so comparisons have no float rounding.
    """
    inf = math.inf
    if a["kind"] == "interval":
        lo, lo_c, hi, hi_c = Fraction(a["low"]), True, Fraction(a["high"]), True
    else:
        v = Fraction(a["value"])
        lo, lo_c, hi, hi_c = {
            "gt": (v, False, inf, False),
            "ge": (v, True, inf, False),
            "lt": (-inf, False, v, False),
            "le": (-inf, False, v, True),
        }[a["op"]]
    if variable in COUNT_VARS:
        if lo != -inf:
            lo = math.ceil(lo) if lo_c else math.floor(lo) + 1
            lo_c = True
        if hi != inf:
            hi = math.floor(hi) if hi_c else math.ceil(hi) - 1
            hi_c = True
    return lo, lo_c, hi, hi_c


def assertions_disjoint(variable: str, a: dict, b: dict) -> bool:
    """True when no value of the variable can satisfy both assertions."""
    if a["kind"] in ("boolean", "categorical") or b["kind"] in ("boolean", "categorical"):
        return a["kind"] == b["kind"] and a["value"] != b["value"]
    alo, alc, ahi, ahc = _as_interval(variable, a)
    blo, blc, bhi, bhc = _as_interval(variable, b)
    for lo, lo_c, hi, hi_c in ((alo, alc, ahi, ahc), (blo, blc, bhi, bhc)):
        if lo > hi or (lo == hi and not (lo_c and hi_c)):
            return True  # an assertion that can never be true shares no value with anything
    if alo > blo or (alo == blo and not alc):
        lo, lo_c = alo, alc
    else:
        lo, lo_c = blo, blc
    if ahi < bhi or (ahi == bhi and not ahc):
        hi, hi_c = ahi, ahc
    else:
        hi, hi_c = bhi, bhc
    return lo > hi or (lo == hi and not (lo_c and hi_c))


def find_contradictions(claims_by_predictor: dict) -> list:
    """All (claim_i, claim_j), i < j, on the same target with disjoint assertions."""
    by_target: dict[tuple, list] = {}
    for p in PREDICTORS:
        for c in claims_by_predictor.get(p, []):
            by_target.setdefault(target_key(c["target"]), []).append(c)
    out = []
    for key in sorted(by_target):
        group = by_target[key]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["predictor"] == b["predictor"]:
                    continue
                if assertions_disjoint(key[3], a["assertion"], b["assertion"]):
                    out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Ground truth variables
# ---------------------------------------------------------------------------

def _window_days(daily: dict, start: dt.date, end: dt.date):
    """Daily records in [start, end]; None if any day is absent."""
    days = []
    d = start
    while d <= end:
        rec = daily.get(d.isoformat())
        if rec is None:
            return None
        days.append((d, rec))
        d += dt.timedelta(days=1)
    return days


def _values(days, field):
    vals = [rec.get(field) for _, rec in days]
    if any(v is None for v in vals):
        return None
    return vals


def _r(x):
    return round(x, AGG_DECIMALS)


def _same_window_in_year(start: dt.date, end: dt.date, year: int) -> tuple[dt.date, dt.date]:
    """Map a window to the same month-days in another year; Feb 29 becomes Mar 1 as a
    start and Feb 28 as an end when that year has no Feb 29."""
    def move(d: dt.date, is_start: bool) -> dt.date:
        if d.month == 2 and d.day == 29 and not calendar.isleap(year):
            return dt.date(year, 3, 1) if is_start else dt.date(year, 2, 28)
        return d.replace(year=year)
    return move(start, True), move(end, False)


def _monthly(days, field, how):
    by_month: dict[str, list] = {}
    for d, rec in days:
        by_month.setdefault(f"{d.month:02d}", []).append(rec.get(field))
    out = {}
    for m, vals in by_month.items():
        if any(v is None for v in vals):
            return None
        out[m] = _r(sum(vals) if how == "sum" else sum(vals) / len(vals))
    return out


def _argmax_unique(d: dict):
    best = max(d.values())
    winners = [k for k, v in d.items() if v == best]
    return winners[0] if len(winners) == 1 else None  # ties are unresolvable


def compute_variable(variable: str, daily: dict, start: dt.date, end: dt.date, year_x: int):
    """Ground-truth value of a canonical variable, or None when it cannot be determined.

    `daily` maps ISO date -> {open-meteo field: value}, covering X+1 (and X if needed).
    """
    days = _window_days(daily, start, end)
    if days is None:
        return None

    def agg(field, fn):
        vals = _values(days, field)
        return None if vals is None else fn(vals)

    mean = lambda v: _r(sum(v) / len(v))
    if variable == "temp_mean_c":
        return agg("temperature_2m_mean", mean)
    if variable == "temp_max_c":
        return agg("temperature_2m_max", max)
    if variable == "temp_min_c":
        return agg("temperature_2m_min", min)
    if variable == "temp_mean_max_c":
        return agg("temperature_2m_max", mean)
    if variable == "temp_mean_min_c":
        return agg("temperature_2m_min", mean)
    if variable == "precip_sum_mm":
        return agg("precipitation_sum", lambda v: _r(sum(v)))
    if variable == "precip_max_day_mm":
        return agg("precipitation_sum", max)
    if variable == "wind_max_kmh":
        return agg("wind_speed_10m_max", max)
    if variable == "hot_days_ge_30c":
        return agg("temperature_2m_max", lambda v: sum(1 for x in v if x >= 30.0))
    if variable == "frost_days_lt_0c":
        return agg("temperature_2m_min", lambda v: sum(1 for x in v if x < 0.0))
    if variable == "rainy_days_ge_1mm":
        return agg("precipitation_sum", lambda v: sum(1 for x in v if x >= 1.0))
    if variable == "wettest_month":
        m = _monthly(days, "precipitation_sum", "sum")
        return None if m is None else _argmax_unique(m)
    if variable == "hottest_month":
        m = _monthly(days, "temperature_2m_mean", "mean")
        return None if m is None else _argmax_unique(m)
    if variable in BOOLEAN_VARS:
        base = "temp_mean_c" if variable == "warmer_than_year_x" else "precip_sum_mm"
        now = compute_variable(base, daily, start, end, year_x)
        ps, pe = _same_window_in_year(start, end, year_x)
        prev = compute_variable(base, daily, ps, pe, year_x)
        if now is None or prev is None:
            return None
        return now > prev
    raise ValueError(f"unknown variable {variable}")


def assertion_holds(variable: str, a: dict, value) -> bool:
    if a["kind"] in ("boolean", "categorical"):
        return a["value"] == value
    if a["kind"] == "interval":
        return a["low"] <= value <= a["high"]
    return {"gt": value > a["value"], "ge": value >= a["value"],
            "lt": value < a["value"], "le": value <= a["value"]}[a["op"]]


# ---------------------------------------------------------------------------
# Open-Meteo archive fetching (batched, cached per (lat, lon, start, end))
# ---------------------------------------------------------------------------

class OpenMeteoArchive:
    def __init__(self, cache_dir: Path | None, batch_size: int = 10, retries: int = 6, timeout: int = 60):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.retries = retries
        self.timeout = timeout
        self.errors: list[str] = []
        self.requests = 0

    def _cache_path(self, lat, lon, start, end) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{lat:.4f}_{lon:.4f}_{start}_{end}.json"

    @staticmethod
    def _to_daily(payload: dict) -> dict:
        d = payload.get("daily") or {}
        times = d.get("time") or []
        return {t: {f: (d.get(f) or [None] * len(times))[i] for f in DAILY_FIELDS}
                for i, t in enumerate(times)}

    def _get(self, params: dict):
        url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
        delay = 10.0
        for attempt in range(self.retries + 1):
            try:
                self.requests += 1
                req = urllib.request.Request(url, headers={"User-Agent": "longhorizonweather-verify/1"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:300]
                quota = e.code == 429 and ("Hourly" in body or "Daily" in body)
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries and not quota:
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                raise RuntimeError(f"network error: {e}") from None

    def fetch(self, coords: list[tuple[float, float]], start: str, end: str) -> dict:
        """Return {(lat, lon): daily dict}; locations that fail are absent (and logged)."""
        out, todo = {}, []
        for lat, lon in dict.fromkeys(coords):
            p = self._cache_path(lat, lon, start, end)
            if p is not None and p.exists():
                out[(lat, lon)] = json.loads(p.read_text())
            else:
                todo.append((lat, lon))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            params = {
                "latitude": ",".join(f"{lat:.4f}" for lat, _ in batch),
                "longitude": ",".join(f"{lon:.4f}" for _, lon in batch),
                "start_date": start, "end_date": end,
                "daily": ",".join(DAILY_FIELDS), "timezone": "GMT",
            }
            try:
                payload = self._get(params)
            except RuntimeError as e:
                self.errors.append(f"{batch}: {e}")
                continue
            payloads = payload if isinstance(payload, list) else [payload]
            if len(payloads) != len(batch):
                self.errors.append(f"{batch}: expected {len(batch)} locations, got {len(payloads)}")
                continue
            for loc, pl in zip(batch, payloads):
                daily = self._to_daily(pl)
                out[loc] = daily
                complete = daily and all(v is not None for rec in daily.values() for v in rec.values())
                p = self._cache_path(*loc, start, end)
                if p is not None and complete:  # never cache gaps that may fill in later
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(daily))
        return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _claim_view(c: dict) -> dict:
    return {"predictor": c["predictor"], "claim_id": c["claim_id"],
            "assertion": c["assertion"], "confidence": c["confidence"]}


def score(year_x: int, claims_by_predictor: dict, fetch_daily) -> dict:
    """Resolve contradictions. `fetch_daily(coords, start, end)` returns {coords: daily}."""
    pairs = find_contradictions(claims_by_predictor)
    y1 = year_x + 1
    needs_x = {(a["target"]["lat"], a["target"]["lon"]) for a, _ in pairs
               if a["target"]["variable"] in BOOLEAN_VARS}
    coords = sorted({(a["target"]["lat"], a["target"]["lon"]) for a, _ in pairs})
    daily = {}
    plain = [c for c in coords if c not in needs_x]
    if plain:
        daily.update(fetch_daily(plain, f"{y1}-01-01", f"{y1}-12-31"))
    if needs_x:
        daily.update(fetch_daily(sorted(needs_x), f"{year_x}-01-01", f"{y1}-12-31"))

    deltas = {p: 0 for p in PREDICTORS}
    details, resolved, truth_cache = [], 0, {}
    for a, b in pairs:
        t = a["target"]
        key = target_key(t)
        if key not in truth_cache:
            loc = daily.get((t["lat"], t["lon"]))
            truth_cache[key] = None if loc is None else compute_variable(
                t["variable"], loc, dt.date.fromisoformat(t["window_start"]),
                dt.date.fromisoformat(t["window_end"]), year_x)
        truth = truth_cache[key]
        entry = {"target": t, "a": _claim_view(a), "b": _claim_view(b), "truth": truth}
        if truth is None:
            entry.update(status="unresolvable", a_correct=None, b_correct=None, delta_a=0, delta_b=0)
        else:
            ac = assertion_holds(t["variable"], a["assertion"], truth)
            bc = assertion_holds(t["variable"], b["assertion"], truth)
            da = (1 if ac else -1) if ac != bc else 0
            db = -da
            deltas[a["predictor"]] += da
            deltas[b["predictor"]] += db
            resolved += 1
            status = "both_wrong" if not (ac or bc) else ("a_wins" if ac else "b_wins")
            entry.update(status=status, a_correct=ac, b_correct=bc, delta_a=da, delta_b=db)
        details.append(entry)
    return {"scores_delta": deltas, "contradictions_count": len(pairs),
            "resolved_count": resolved, "contradictions": details}


def run(year_x: int, claims_dir: Path, fetch_daily) -> dict:
    claims_by_predictor, intake = {}, {}
    for p in PREDICTORS:
        raw, err = load_predictor_file(claims_dir / f"{p}.json")
        accepted, rejected = filter_claims(p, raw, year_x)
        claims_by_predictor[p] = accepted
        intake[p] = {"file_status": err or "ok", "submitted": len(raw),
                     "accepted": len(accepted), "rejected": rejected}
    result = score(year_x, claims_by_predictor, fetch_daily)
    return {"year_x": year_x, "year_x_plus_1": year_x + 1, **result, "intake": intake}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year-x", type=int, required=True)
    ap.add_argument("--claims-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data_cache" / "verify_openmeteo",
                    help="cache for complete Open-Meteo responses (default: <repo>/data_cache/verify_openmeteo)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    archive = OpenMeteoArchive(None if args.no_cache else args.cache_dir)
    report = run(args.year_x, args.claims_dir, archive.fetch)
    report["ground_truth"] = {"source": ARCHIVE_URL, "timezone": "GMT",
                              "http_requests": archive.requests, "errors": archive.errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("year_x", "contradictions_count", "resolved_count", "scores_delta")}))
    for e in archive.errors:
        print(f"error: ground truth fetch failed: {e}", file=sys.stderr)
    # A failed fetch leaves contradictions unresolved; exit nonzero so they are not
    # mistaken for draws. The report is still written for inspection.
    return 2 if archive.errors else 0


if __name__ == "__main__":
    sys.exit(main())
