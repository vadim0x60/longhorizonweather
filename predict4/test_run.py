"""Tests for predict4/run.py. Run: python3 -m unittest discover -s predict4 -v"""

import csv
import datetime as dt
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P4 = _load("predict4_run", HERE / "run.py")
VERIFY = _load("verify_run", HERE.parent / "verify" / "run.py")  # contract validator only

X = 2020


def write_data(td, hot_last_year=8.0, leak_value=99.0):
    """Synthetic London history 2005..X: 10 °C ± noise, year X much hotter; plus a leaked X+1 row."""
    rng = random.Random(4)
    rows = []
    for year in range(2005, X + 2):
        bump = rng.gauss(0, 0.3) + (hot_last_year if year == X else 0.0)
        d = dt.date(year, 1, 1)
        while d.year == year:
            t = (leak_value if year == X + 1 else 10.0 + bump + rng.gauss(0, 1))
            rows.append(["london_uk", 51.5074, -0.1278, d.isoformat(), t, t + 4, t - 4,
                         max(0.0, rng.gauss(1.5, 2)), 20.0])
            d += dt.timedelta(days=1)
    hdr = ["location_id", "lat", "lon", "date", *P4.FIELDS]
    with open(Path(td) / "daily_history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    (Path(td) / "manifest.json").write_text(json.dumps({"year_x": X, "max_date": f"{X}-12-31"}))


class Predict4Tests(unittest.TestCase):
    def run_p4(self, **kw):
        with tempfile.TemporaryDirectory() as td:
            write_data(td, **kw)
            out = Path(td) / "claims.json"
            P4.main(["--year-x", str(X), "--data-dir", td, "--out", str(out)])
            return json.loads(out.read_text())

    def test_claims_are_valid_and_unique(self):
        claims = self.run_p4()
        self.assertTrue(claims)
        self.assertLessEqual(len(claims), 150)
        acc, rej = VERIFY.filter_claims("predict4", claims, X)
        self.assertEqual(rej, [])
        self.assertEqual(len(acc), len(claims))
        for c in claims:
            self.assertGreaterEqual(c["confidence"], P4.MIN_PROB - 1e-9)

    def test_reverts_after_outlier_year(self):
        claims = {(c["target"]["variable"], c["target"]["window_start"], c["target"]["window_end"]): c
                  for c in self.run_p4()}
        annual = claims[("temp_mean_c", f"{X + 1}-01-01", f"{X + 1}-12-31")]
        # Year X was ~18 °C against a ~10 °C base: expect a claim below persistence, well above 10.
        self.assertEqual(annual["assertion"]["op"], "le")
        self.assertTrue(10 < annual["assertion"]["value"] < 18, annual)
        warmer = claims[("warmer_than_year_x", f"{X + 1}-01-01", f"{X + 1}-12-31")]
        self.assertIs(warmer["assertion"]["value"], False)

    def test_ignores_rows_after_year_x(self):
        a = self.run_p4(leak_value=99.0)
        b = self.run_p4(leak_value=-50.0)
        self.assertEqual(a, b)

    def test_forecast_shrinks_noisy_trend(self):
        # Pure noise series: shrunk slope keeps the mean near the sample mean.
        rng = random.Random(1)
        s = {y: 10 + rng.gauss(0, 1) for y in range(2005, 2021)}
        mean, sd, n = P4.forecast(s, 2020, "precip_sum_mm")
        self.assertEqual(n, 16)
        self.assertLess(abs(mean - sum(s.values()) / 16), 1.0)
        self.assertGreater(sd, 0.8)
        self.assertIsNone(P4.forecast({2019: 1.0, 2020: 2.0}, 2020, "temp_mean_c"))


if __name__ == "__main__":
    unittest.main()
