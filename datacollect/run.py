#!/usr/bin/env python3
"""Collect weather observations strictly dated <= {X}-12-31.

Usage:
    python3 datacollect/run.py --year <X> --out <data_dir> [--cache-dir DIR] [--no-cache] [--no-indices]

Outputs in <data_dir> (see CONTRACTS.md):
  manifest.json        year_x, max_date, locations, files, sources
  daily_year_x.csv     {X}-01-01 .. {X}-12-31 for the 20 benchmark locations
  daily_history.csv    max(1980, X-15)-01-01 .. {X}-12-31, same schema
  indices_monthly.json optional monthly climate indices (ENSO, NAO, AO, PDO), months <= {X}-12

Daily data comes from the Open-Meteo historical archive (ERA5 reanalysis, daily
aggregates in GMT). Each request's end_date is asserted to be <= {X}-12-31 before
it is sent. Full calendar years are cached per (location, year) under --cache-dir
(default: <repo>/data_cache/openmeteo, git-ignored) so repeated runs for nearby
years stay fast and within Open-Meteo's free-tier quota.

Climate index files are published as whole series, so the raw download necessarily
contains months after X; rows dated after {X}-12 are discarded while parsing and
never written. Every output is re-checked against the cutoff before it is written.

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCATIONS = [
    ("london_uk", 51.5074, -0.1278),
    ("new_york_us", 40.7128, -74.0060),
    ("tokyo_jp", 35.6762, 139.6503),
    ("sydney_au", -33.8688, 151.2093),
    ("cairo_eg", 30.0444, 31.2357),
    ("mumbai_in", 19.0760, 72.8777),
    ("sao_paulo_br", -23.5505, -46.6333),
    ("reykjavik_is", 64.1466, -21.9426),
    ("nairobi_ke", -1.2921, 36.8219),
    ("phoenix_us", 33.4484, -112.0740),
    ("singapore_sg", 1.3521, 103.8198),
    ("chicago_us", 41.8781, -87.6298),
    ("moscow_ru", 55.7558, 37.6173),
    ("cape_town_za", -33.9249, 18.4241),
    ("lima_pe", -12.0464, -77.0428),
    ("bangkok_th", 13.7563, 100.5018),
    ("anchorage_us", 61.2181, -149.9003),
    ("madrid_es", 40.4168, -3.7038),
    ("seoul_kr", 37.5665, 126.9780),
    ("toronto_ca", 43.6532, -79.3832),
]

VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
COLUMNS = ["location_id", "lat", "lon", "date"] + VARS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "GMT"
BATCH_LOCATIONS = 20  # locations per HTTP request (Open-Meteo charges quota after a request, so fewer, larger requests finish fastest)

INDEX_SOURCES = {
    # name: (url, parser)
    "nino34_anom": ("https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii", "cpc_nino34"),
    "nao": ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii", "year_month_value"),
    "ao": ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii", "year_month_value"),
    "pdo": ("https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat", "year_12cols"),
}
INDEX_START_YEAR = 1950


def log(msg: str) -> None:
    print(f"[datacollect] {msg}", file=sys.stderr, flush=True)


def http_get(url: str, timeout: float = 60.0, attempts: int = 5) -> bytes:
    """GET with retries. Waits out Open-Meteo per-minute limits; fails fast on hourly/daily limits."""
    delay = 2.0
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "longhorizonweather-datacollect/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last = RuntimeError(f"HTTP {e.code} for {url[:120]}: {body}")
            if e.code == 429:
                low = body.lower()
                if "hourly" in low or "daily" in low:
                    raise RuntimeError(f"Open-Meteo quota exhausted ({body}); rerun later (cached years are kept)") from e
                wait = 61.0
            elif e.code >= 500:
                wait = delay
            else:
                raise last from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            wait = delay
        if i + 1 < attempts:
            log(f"retry {i + 1}/{attempts - 1} in {wait:.0f}s: {last}")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError(f"giving up on {url[:120]}: {last}")


# ---------------------------------------------------------------- daily data


def cache_path(cache_dir: str, loc_id: str, year: int) -> str:
    return os.path.join(cache_dir, f"{loc_id}_{year}.json")


def fetch_range(locs: list[tuple[str, float, float]], start: dt.date, end: dt.date, cutoff: dt.date) -> dict[str, dict]:
    """One archive request for several locations; returns {loc_id: daily dict}."""
    assert end <= cutoff, f"refusing to query beyond cutoff: {end} > {cutoff}"
    params = {
        "latitude": ",".join(f"{lat}" for _, lat, _ in locs),
        "longitude": ",".join(f"{lon}" for _, _, lon in locs),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(VARS),
        "timezone": TIMEZONE,
    }
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    payload = json.loads(http_get(url))
    if isinstance(payload, dict):
        payload = [payload]
    if len(payload) != len(locs):
        raise RuntimeError(f"expected {len(locs)} locations, got {len(payload)}")
    return {loc[0]: p["daily"] for loc, p in zip(locs, payload)}


def split_years(daily: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for i, d in enumerate(daily["time"]):
        y = int(d[:4])
        rec = out.setdefault(y, {"time": [], **{v: [] for v in VARS}})
        rec["time"].append(d)
        for v in VARS:
            rec[v].append(daily[v][i])
    return out


def collect_daily(year_x: int, start_year: int, cache_dir: str | None) -> dict[str, dict[int, dict]]:
    cutoff = dt.date(year_x, 12, 31)
    data: dict[str, dict[int, dict]] = {loc[0]: {} for loc in LOCATIONS}
    missing: dict[str, list[int]] = {}
    for loc_id, _, _ in LOCATIONS:
        for y in range(start_year, year_x + 1):
            p = cache_path(cache_dir, loc_id, y) if cache_dir else None
            if p and os.path.exists(p):
                with open(p) as f:
                    data[loc_id][y] = json.load(f)
            else:
                missing.setdefault(loc_id, []).append(y)
    # Group locations that miss the same contiguous year span into batched requests.
    groups: dict[tuple[int, int], list] = {}
    for loc in LOCATIONS:
        ys = missing.get(loc[0])
        if ys:
            groups.setdefault((min(ys), max(ys)), []).append(loc)
    n_req = 0
    for (y0, y1), locs in sorted(groups.items()):
        for i in range(0, len(locs), BATCH_LOCATIONS):
            batch = locs[i:i + BATCH_LOCATIONS]
            log(f"fetching {[l[0] for l in batch]} {y0}..{y1}")
            got = fetch_range(batch, dt.date(y0, 1, 1), dt.date(y1, 12, 31), cutoff)
            n_req += 1
            for loc_id, daily in got.items():
                for y, rec in split_years(daily).items():
                    if not (y0 <= y <= y1):
                        continue
                    data[loc_id][y] = rec
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                        tmp = cache_path(cache_dir, loc_id, y) + ".tmp"
                        with open(tmp, "w") as f:
                            json.dump(rec, f)
                        os.replace(tmp, cache_path(cache_dir, loc_id, y))
    log(f"daily data: {n_req} archive request(s), {sum(len(v) for v in missing.values())} location-years fetched")
    return data


def fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def build_rows(data, year_from: int, year_to: int, cutoff: dt.date) -> list[list[str]]:
    rows = []
    for loc_id, lat, lon in LOCATIONS:
        for y in range(year_from, year_to + 1):
            rec = data[loc_id][y]
            expected = (dt.date(y, 12, 31) - dt.date(y, 1, 1)).days + 1
            if len(rec["time"]) != expected:
                raise RuntimeError(f"{loc_id} {y}: {len(rec['time'])} days, expected {expected}")
            for i, d in enumerate(rec["time"]):
                if dt.date.fromisoformat(d) > cutoff:
                    raise AssertionError(f"leak: {loc_id} {d} > {cutoff}")
                rows.append([loc_id, str(lat), str(lon), d] + [fmt(rec[v][i]) for v in VARS])
    return rows


def write_csv(path: str, rows: list[list[str]], cutoff: dt.date) -> None:
    assert all(r[3] <= cutoff.isoformat() for r in rows), "date beyond cutoff in " + path
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------- climate indices


def parse_index(text: str, kind: str, year_x: int) -> list[dict]:
    out = []

    def add(y: int, m: int, v: float) -> None:
        if INDEX_START_YEAR <= y and (y, m) <= (year_x, 12) and abs(v) < 90:
            out.append({"date": f"{y:04d}-{m:02d}", "value": round(v, 3)})

    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit() or len(parts[0]) != 4:
            continue
        try:
            y = int(parts[0])
            if kind == "cpc_nino34" and len(parts) >= 10:
                add(y, int(parts[1]), float(parts[9]))
            elif kind == "year_month_value" and len(parts) >= 3:
                add(y, int(parts[1]), float(parts[2]))
            elif kind == "year_12cols" and len(parts) >= 13:
                for m in range(1, 13):
                    add(y, m, float(parts[m]))
        except ValueError:
            continue
    out.sort(key=lambda r: r["date"])
    return out


def collect_indices(year_x: int, cache_dir: str | None) -> tuple[dict, dict]:
    """Returns ({name: [{date, value}]}, {name: status})."""
    series, status = {}, {}
    for name, (url, kind) in INDEX_SOURCES.items():
        raw_path = os.path.join(cache_dir, f"index_{name}.txt") if cache_dir else None
        try:
            if raw_path and os.path.exists(raw_path):
                with open(raw_path) as f:
                    text = f.read()
            else:
                text = http_get(url, timeout=20, attempts=2).decode("utf-8", "replace")
                if raw_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(raw_path, "w") as f:
                        f.write(text)
            rows = parse_index(text, kind, year_x)
            if not rows or rows[-1]["date"] < f"{year_x}-01":
                raise RuntimeError("no rows up to year X")
            series[name] = rows
            status[name] = {"source": url, "last": rows[-1]["date"], "n": len(rows)}
        except Exception as e:  # optional data: fall back to omitting it
            log(f"index {name} unavailable: {e}")
            status[name] = {"source": url, "error": str(e)[:200]}
    return series, status


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "data_cache", "openmeteo"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-indices", action="store_true")
    a = ap.parse_args()

    X = a.year
    if not 1980 <= X <= 2024:
        ap.error("--year must be in 1980..2024")
    cutoff = dt.date(X, 12, 31)
    start_year = max(1980, X - 15)
    cache_dir = None if a.no_cache else a.cache_dir
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    data = collect_daily(X, start_year, cache_dir)
    hist = build_rows(data, start_year, X, cutoff)
    year_rows = [r for r in hist if r[3] >= f"{X}-01-01"]
    max_date = max(r[3] for r in hist)
    assert max_date <= cutoff.isoformat(), f"max_date {max_date} beyond cutoff"
    write_csv(os.path.join(a.out, "daily_history.csv"), hist, cutoff)
    write_csv(os.path.join(a.out, "daily_year_x.csv"), year_rows, cutoff)
    files = ["daily_year_x.csv", "daily_history.csv"]

    index_status = {}
    if not a.no_indices:
        series, index_status = collect_indices(X, cache_dir)
        if series:
            assert all(r["date"] <= f"{X}-12" for s in series.values() for r in s), "index beyond cutoff"
            with open(os.path.join(a.out, "indices_monthly.json"), "w") as f:
                json.dump({"year_x": X, "max_month": f"{X}-12", "series": series}, f)
            files.append("indices_monthly.json")

    manifest = {
        "year_x": X,
        "max_date": max_date,
        "min_date": min(r[3] for r in hist),
        "history_start_year": start_year,
        "locations": [{"location_id": i, "lat": lat, "lon": lon} for i, lat, lon in LOCATIONS],
        "files": files,
        "columns": COLUMNS,
        "rows": {"daily_year_x.csv": len(year_rows), "daily_history.csv": len(hist)},
        "sources": {
            "daily": {"api": ARCHIVE_URL, "timezone": TIMEZONE, "units": {
                "temperature": "degC", "precipitation": "mm", "wind_speed_10m_max": "km/h"}},
            "indices": index_status,
        },
    }
    assert manifest["max_date"] == f"{X}-12-31", manifest["max_date"]
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"wrote {a.out}: {len(hist)} history rows, {len(year_rows)} year-X rows, max_date={max_date}, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        log(f"ERROR: {e}")
        sys.exit(2)
