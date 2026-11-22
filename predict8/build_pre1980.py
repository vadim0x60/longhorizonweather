#!/usr/bin/env python3
"""Build predict8/pre1980_monthly.csv: monthly aggregates of ERA5 daily data, 1964..1979.

    python3 predict8/build_pre1980.py <dir with <location_id>.json Open-Meteo archive responses>

Each input is an Open-Meteo archive response (timezone GMT, 1964-01-01..1979-12-31) with daily
temperature_2m_mean/max/min, precipitation_sum and wind_speed_10m_max for a CONTRACTS.md
location. Every date is before 1980, so for any forecast year X >= 1980 the file only holds
observations dated <= X-12-31. run.py uses it to extend short histories (X - 15 < 1980) back to
16 years. Not run at prediction time (needs the network to produce the inputs).
"""
import csv
import json
import os
import sys

FIELDS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
          "precipitation_sum", "wind_speed_10m_max"]
OUT_COLS = ["location_id", "year", "month", "ndays", "tm_mean", "tx_mean", "tn_mean", "p_sum",
            "tx_max", "tn_min", "p_max", "w_max", "hot", "frost", "rainy"]


def main():
    src = sys.argv[1]
    rows = []
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        loc = name[:-5]
        d = json.load(open(os.path.join(src, name)))["daily"]
        by_month = {}
        for i, t in enumerate(d["time"]):
            if t >= "1980-01-01":
                raise SystemExit(f"{name}: date {t} is not before 1980")
            by_month.setdefault((int(t[:4]), int(t[5:7])), []).append([d[f][i] for f in FIELDS])
        for (y, m), recs in sorted(by_month.items()):
            if any(v is None for r in recs for v in r):
                continue
            tm, tx, tn, p, w = zip(*recs)
            n = len(recs)
            rows.append([loc, y, m, n, round(sum(tm) / n, 3), round(sum(tx) / n, 3), round(sum(tn) / n, 3),
                         round(sum(p), 2), max(tx), min(tn), max(p), max(w),
                         sum(1 for v in tx if v >= 30.0), sum(1 for v in tn if v < 0.0),
                         sum(1 for v in p if v >= 1.0)])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre1980_monthly.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLS)
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} location-months")


if __name__ == "__main__":
    main()
