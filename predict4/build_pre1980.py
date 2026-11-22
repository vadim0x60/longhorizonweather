#!/usr/bin/env python3
"""Build predict4/data/monthly_pre1980.csv: monthly aggregates for 1964..1979 (ERA5 via the
Open-Meteo archive, the same source and GMT days as datacollect) at the 20 catalog locations.

    python3 predict4/build_pre1980.py

Run once, offline from the arena; predict4/run.py only reads the CSV. Every row is dated
before 1980, so it is earlier than any year X the arena uses (1980..2024). predict4 uses it to
extend its 16-year point-forecast window when datacollect's history starts at 1980.
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import CATALOG  # noqa: E402

FIELDS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
URL = "https://archive-api.open-meteo.com/v1/archive"


def main():
    locs = list(CATALOG.items())
    params = {"latitude": ",".join(str(c[0]) for _, c in locs), "longitude": ",".join(str(c[1]) for _, c in locs),
              "start_date": "1964-01-01", "end_date": "1979-12-31", "daily": ",".join(FIELDS), "timezone": "GMT"}
    with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(params), timeout=300) as r:
        payload = json.loads(r.read())
    rows = []
    for (loc, _), p in zip(locs, payload):
        d = p["daily"]
        by = {}
        for i, t in enumerate(d["time"]):
            assert t < "1980-01-01"
            by.setdefault(t[:7], []).append([d[f][i] for f in FIELDS])
        for ym, recs in sorted(by.items()):
            if any(v is None for r in recs for v in r):
                continue
            n = len(recs)
            rows.append([loc, ym, n] + [round(sum(r[j] for r in recs) / n, 4) for j in range(3)]
                        + [round(sum(r[3] for r in recs), 2)])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "monthly_pre1980.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location_id", "month", "days", "temp_mean_c", "temp_mean_max_c", "temp_mean_min_c",
                    "precip_sum_mm"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
