"""Tests for predict4/run.py. Run: python3 -m unittest discover -s predict4 -v"""

import csv
import datetime as dt
import importlib.util
import json
import random
import statistics
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


def write_data(td, hot_last_year=8.0, leak_value=99.0, trend=0.0):
    """Synthetic London history 2005..X: 10 °C ± noise (+trend/yr), year X hotter; plus a leaked X+1 row."""
    rng = random.Random(4)
    rows = []
    for year in range(2005, X + 2):
        bump = rng.gauss(0, 0.3) + (hot_last_year if year == X else 0.0) + trend * (year - 2005)
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
        return self.run_p4_rows(**kw)[0]

    def run_p4_rows(self, **kw):
        with tempfile.TemporaryDirectory() as td:
            write_data(td, **kw)
            out = Path(td) / "claims.json"
            P4.main(["--year-x", str(X), "--data-dir", td, "--out", str(out)])
            with open(Path(td) / "daily_history.csv") as fh:
                rows = list(csv.DictReader(fh))
            return json.loads(out.read_text()), rows

    def test_claims_are_valid_and_unique(self):
        claims = self.run_p4()
        self.assertTrue(claims)
        self.assertLessEqual(len(claims), 150)
        acc, rej = VERIFY.filter_claims("predict4", claims, X)
        self.assertEqual(rej, [])
        self.assertEqual(len(acc), len(claims))
        for c in claims:
            if c["assertion"]["kind"] != "point":  # points are scored by error, not probability
                self.assertGreaterEqual(c["confidence"], P4.MIN_PROB - 1e-9)

    def test_reverts_after_outlier_year(self):
        claims = {(c["target"]["variable"], c["target"]["window_start"], c["target"]["window_end"]): c
                  for c in self.run_p4()}
        # Year X was ~18 °C against a ~10 °C base: next year is very likely not warmer.
        warmer = claims[("warmer_than_year_x", f"{X + 1}-01-01", f"{X + 1}-12-31")]
        self.assertIs(warmer["assertion"]["value"], False)
        self.assertGreaterEqual(warmer["confidence"], 0.8)

    def test_threshold_sits_on_rival_line(self):
        P4_points = P4.POINTS_PER_LOCATION
        P4.POINTS_PER_LOCATION = 0  # the point claim would otherwise take the annual target
        self.addCleanup(setattr, P4, "POINTS_PER_LOCATION", P4_points)
        # Steady warming: the claim is "gt" the higher trailing mean, which under warming is the
        # 15-year mean of X-14..X (the 16-year mean of X-15..X includes the colder first year).
        claims, rows = self.run_p4_rows(hot_last_year=0.0, trend=0.15)
        annual = [c for c in claims if c["target"]["variable"] == "temp_mean_c"
                  and c["target"]["window_start"] == f"{X + 1}-01-01" and c["target"]["window_end"] == f"{X + 1}-12-31"]
        self.assertEqual(len(annual), 1)
        by_year = {}
        for r in rows:
            by_year.setdefault(int(r["date"][:4]), []).append(float(r["temperature_2m_mean"]))
        yearly = {y: sum(v) / len(v) for y, v in by_year.items() if y <= X}
        mean15 = sum(yearly[y] for y in range(X - 14, X + 1)) / 15
        self.assertEqual(annual[0]["assertion"], {"kind": "threshold", "op": "gt", "value": round(mean15, 2)})
        self.assertLessEqual(annual[0]["confidence"], 0.75)  # line claims are shrunk toward 0.5

    def test_point_claims_and_caps(self):
        claims, rows = self.run_p4_rows(hot_last_year=0.0, trend=0.15)
        points = [c for c in claims if c["assertion"]["kind"] == "point"]
        self.assertEqual(len(points), P4.POINTS_PER_LOCATION)  # one location in the fixture
        self.assertLessEqual(len(claims), P4.MAX_PER_LOCATION)
        acc, rej = VERIFY.filter_claims("predict4", claims, X)
        self.assertEqual(rej, [])
        # Annual temp_mean_c point = shrunk trend line at X+1 + 0.2 * year-X residual
        # + 0.2 * year-X Sep-Dec residual, recomputed here with statistics.linear_regression.
        def daily_mean(pred):
            by = {}
            for r in rows:
                d = dt.date.fromisoformat(r["date"])
                if d.year <= X and pred(d):
                    by.setdefault(d.year, []).append(float(r["temperature_2m_mean"]))
            return {y: sum(v) / len(v) for y, v in by.items() if y >= X - 15}

        def line_and_resid(series):
            ys = sorted(series)
            slope, _ = statistics.linear_regression(ys, [series[y] for y in ys])
            w = len(ys) / (len(ys) + 40)
            b = w * slope + (1 - w) * 0.03
            ybar, tbar = statistics.mean(series.values()), statistics.mean(ys)
            return ybar + b * (X + 1 - tbar), series[X] - (ybar + b * (X - tbar))

        line, r_year = line_and_resid(daily_mean(lambda d: True))
        _, r_sond = line_and_resid(daily_mean(lambda d: d.month >= 9))
        annual = next(c for c in points if c["target"]["variable"] == "temp_mean_c"
                      and c["target"]["window_end"] == f"{X + 1}-12-31")
        self.assertAlmostEqual(annual["assertion"]["value"], line + 0.1 * r_year + 0.2 * r_sond, places=2)
        # The shrunk trend undershoots the true 0.15/yr warming: below naive extrapolation.
        self.assertLess(annual["assertion"]["value"], statistics.mean(daily_mean(lambda d: d.year == X).values()) + 0.15)

    def test_pre1980_windows(self):
        # Two months of JJA-style data: day-weighted temperature means, precip sums; no year > X.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.csv"
            rows = ["location_id,month,days,temp_mean_c,temp_mean_max_c,temp_mean_min_c,precip_sum_mm"]
            for m, days, t, pr in ((6, 30, 10.0, 5.0), (7, 31, 20.0, 7.5), (8, 31, 30.0, 1.0)):
                rows += [f"london_uk,1975-{m:02d},{days},{t},{t + 4},{t - 4},{pr}",
                         f"london_uk,1976-{m:02d},{days},{t},{t + 4},{t - 4},{pr}"]
            p.write_text("\n".join(rows) + "\n")
            old = P4.PRE1980_CSV
            P4.PRE1980_CSV = p
            self.addCleanup(setattr, P4, "PRE1980_CSV", old)
            pre = P4.load_pre1980(1975)
        jja = pre[("london_uk", "temp_mean_c", "jja")]
        self.assertEqual(list(jja), [1975])  # 1976 is after year_x
        self.assertAlmostEqual(jja[1975], (30 * 10 + 31 * 20 + 31 * 30) / 92)
        self.assertAlmostEqual(pre[("london_uk", "precip_sum_mm", "jja")][1975], 13.5)
        self.assertNotIn(("london_uk", "temp_mean_c", "year"), pre)  # incomplete year

    def test_long_mean_blend_needs_contiguous_history(self):
        rng = random.Random(7)
        full = {y: 10 + 0.02 * (y - 1964) + rng.gauss(0, 0.4) for y in range(1964, 1991)}
        recent = {y: v for y, v in full.items() if y >= 1975}
        sond = {y: v - 3 for y, v in full.items()}
        blended = P4.point_forecast(full, 1990, "temp_mean_c", "year", sond)[0]
        plain = P4.point_forecast(recent, 1990, "temp_mean_c", "year", sond)[0]  # 16-yr formula only
        self.assertAlmostEqual(blended, 0.4 * plain + 0.6 * statistics.mean(full.values()), places=2)
        gappy = {y: v for y, v in full.items() if y != 1970}  # gap: no long-mean blend
        self.assertEqual(P4.point_forecast(gappy, 1990, "temp_mean_c", "year", sond)[0], plain)

    def test_enso_term_tracks_lagged_nino(self):
        # Annual temperature = 10 + 0.5 x the previous year's Oct-Dec Nino3.4: the term should
        # recover 0.5, damped by the ridge factor 1 / (1 + 6/16), times (Nino[X] - its mean).
        rng = random.Random(3)
        enso = {y: rng.gauss(0, 1) for y in range(2004, 2020)}
        enso[2020] = 2.0
        s = {y: 10 + 0.5 * enso[y - 1] for y in range(2005, 2021)}
        nb = statistics.mean(enso[y - 1] for y in s)
        saved = dict(P4.ENSO)
        try:
            P4.ENSO.clear()
            without = P4.point_forecast(s, 2020, "temp_mean_c", "year")[0]
            P4.ENSO.update(enso)
            shift = P4.enso_shift(s, 2020, 0.03, 40.0)
            self.assertAlmostEqual(shift, 0.5 / (1 + 6 / 16) * (2.0 - nb), delta=0.1)
            with_enso = P4.point_forecast(s, 2020, "temp_mean_c", "year")[0]
            self.assertAlmostEqual(with_enso - without, P4.ENSO_WEIGHT * shift, places=1)
            # Seasonal windows and precipitation do not use it.
            self.assertEqual(P4.point_forecast(s, 2020, "temp_mean_c", "jja")[0],
                             (P4.ENSO.clear(), P4.point_forecast(s, 2020, "temp_mean_c", "jja")[0])[1])
        finally:
            P4.ENSO.clear()
            P4.ENSO.update(saved)

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
