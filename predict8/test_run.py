#!/usr/bin/env python3
"""Smoke test: predict8 emits 120..150 claims that verify accepts, for short and long histories.

    python3 predict8/test_run.py

Builds synthetic data dirs (no network) shaped like datacollect output and runs run.py on them.
"""
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("verify_run", os.path.join(REPO, "verify", "run.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def make_data_dir(path, year_x):
    rng = random.Random(year_x)
    start = max(1980, year_x - 15)
    cols = ["location_id", "lat", "lon", "date", "temperature_2m_mean", "temperature_2m_max",
            "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
    with open(os.path.join(path, "daily_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for loc, (lat, lon) in verify.CATALOG.items():
            base = 25 - abs(lat) * 0.4
            d = dt.date(start, 1, 1)
            while d <= dt.date(year_x, 12, 31):
                season = math.cos(2 * math.pi * (d.timetuple().tm_yday - 200) / 365) * (1 if lat > 0 else -1)
                t = base + 10 * season * min(1, abs(lat) / 40) + rng.gauss(0, 2)
                p = max(0.0, rng.gauss(0, 6)) if rng.random() < 0.4 else 0.0
                w.writerow([loc, lat, lon, d.isoformat(), round(t, 1), round(t + 5, 1), round(t - 5, 1),
                            round(p, 1), round(abs(rng.gauss(20, 6)), 1)])
                d += dt.timedelta(days=1)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump({"year_x": year_x, "max_date": f"{year_x}-12-31",
                   "locations": [{"location_id": k, "lat": a, "lon": b} for k, (a, b) in verify.CATALOG.items()]}, f)


class SmokeTest(unittest.TestCase):
    def check_year(self, year_x):
        with tempfile.TemporaryDirectory() as d:
            make_data_dir(d, year_x)
            out = os.path.join(d, "claims.json")
            subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--quiet", "--year-x", str(year_x),
                            "--data-dir", d, "--out", out], check=True, timeout=180)
            claims = json.load(open(out))
        accepted, rejected = verify.filter_claims("predict8", claims, year_x)
        self.assertEqual(rejected, [])
        self.assertGreaterEqual(len(accepted), 120)
        self.assertGreaterEqual(sum(c["assertion"]["kind"] == "point" for c in accepted), 60)

    def test_first_year(self):  # one year of history
        self.check_year(1980)

    def test_short_history(self):  # six years of history (round 5 emitted no claims here)
        self.check_year(1985)

    def test_full_history(self):
        self.check_year(2001)


if __name__ == "__main__":
    unittest.main()
