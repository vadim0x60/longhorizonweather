"""Unit tests for verify/run.py. Run: python3 -m unittest discover -s verify -v"""

import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as V  # noqa: E402

X = 2022
Y = X + 1


def claim(cid, var="temp_mean_c", assertion=None, loc="london_uk", ws=f"{Y}-06-01", we=f"{Y}-06-30",
          lat=None, lon=None):
    clat, clon = V.CATALOG.get(loc, (lat, lon))
    return {"claim_id": cid,
            "target": {"location_id": loc, "lat": lat if lat is not None else clat,
                       "lon": lon if lon is not None else clon,
                       "window_start": ws, "window_end": we, "variable": var},
            "assertion": assertion or {"kind": "threshold", "op": "gt", "value": 10},
            "confidence": 0.7}


def th(op, v):
    return {"kind": "threshold", "op": op, "value": v}


def iv(lo, hi):
    return {"kind": "interval", "low": lo, "high": hi}


def synthetic_daily(start, end, fn):
    """fn(date) -> dict of open-meteo fields."""
    out, d = {}, start
    while d <= end:
        out[d.isoformat()] = fn(d)
        d += dt.timedelta(days=1)
    return out


class DisjointTests(unittest.TestCase):
    def d(self, a, b, var="temp_mean_c"):
        r1 = V.assertions_disjoint(var, a, b)
        r2 = V.assertions_disjoint(var, b, a)
        self.assertEqual(r1, r2, "disjointness must be symmetric")
        return r1

    def test_boundary_gt_vs_le(self):
        self.assertTrue(self.d(th("gt", 10), th("le", 10)))

    def test_boundary_ge_vs_le_share_point(self):
        self.assertFalse(self.d(th("ge", 10), th("le", 10)))

    def test_boundary_gt_vs_lt_same_value(self):
        self.assertTrue(self.d(th("gt", 10), th("lt", 10)))

    def test_ge_vs_lt(self):
        self.assertTrue(self.d(th("ge", 10), th("lt", 10)))

    def test_same_direction_never_disjoint(self):
        self.assertFalse(self.d(th("gt", 10), th("gt", 30)))
        self.assertFalse(self.d(th("lt", 10), th("le", -5)))

    def test_overlapping_opposites(self):
        self.assertFalse(self.d(th("gt", 10), th("lt", 10.5)))

    def test_float_precision_boundary(self):
        # 0.1 + 0.2 != 0.3 in floats; exact comparison on the values as given.
        self.assertTrue(self.d(th("gt", 0.30000000000000004), th("le", 0.3)))
        self.assertTrue(self.d(th("gt", 17.5), th("le", 17.5)))

    def test_intervals(self):
        self.assertTrue(self.d(iv(1, 2), iv(3, 4)))
        self.assertFalse(self.d(iv(1, 3), iv(3, 4)))  # inclusive endpoints touch
        self.assertFalse(self.d(iv(1, 5), iv(2, 3)))  # nested

    def test_threshold_vs_interval(self):
        self.assertTrue(self.d(th("gt", 5), iv(1, 5)))
        self.assertFalse(self.d(th("ge", 5), iv(1, 5)))
        self.assertTrue(self.d(th("lt", 1), iv(1, 5)))
        self.assertFalse(self.d(th("le", 1), iv(1, 5)))
        self.assertFalse(self.d(th("gt", 3), iv(1, 5)))

    def test_count_variables_are_integers(self):
        v = "hot_days_ge_30c"
        self.assertTrue(self.d(th("gt", 10), th("lt", 11), v))   # no integer in (10, 11)
        self.assertFalse(self.d(th("gt", 10), th("lt", 11), "temp_mean_c"))
        self.assertTrue(self.d(th("gt", 10.2), th("le", 10.9), v))
        self.assertFalse(self.d(th("ge", 10.2), th("le", 11), v))  # 11 satisfies both
        self.assertTrue(self.d(iv(1.2, 1.8), th("ge", 0), v))  # first can never hold

    def test_boolean_and_categorical(self):
        b = lambda x: {"kind": "boolean", "value": x}
        c = lambda x: {"kind": "categorical", "value": x}
        self.assertTrue(self.d(b(True), b(False), "warmer_than_year_x"))
        self.assertFalse(self.d(b(True), b(True), "warmer_than_year_x"))
        self.assertTrue(self.d(c("07"), c("08"), "hottest_month"))
        self.assertFalse(self.d(c("07"), c("07"), "hottest_month"))


def pt(v):
    return {"kind": "point", "value": v}


class PointDisagreementTests(unittest.TestCase):
    def test_min_gap_boundaries(self):
        # Exactly the gap counts, even with float noise (15.30 - 15.05 = 0.2500000000000018 / 0.24999...).
        self.assertTrue(V.points_disagree("temp_mean_c", pt(15.30), pt(15.05)))
        self.assertTrue(V.points_disagree("temp_mean_c", pt(0.1 + 0.2), pt(0.05)))
        self.assertFalse(V.points_disagree("temp_mean_c", pt(15.29), pt(15.05)))
        self.assertTrue(V.points_disagree("temp_max_c", pt(30.0), pt(30.5)))
        self.assertFalse(V.points_disagree("temp_max_c", pt(30.0), pt(30.4)))
        self.assertTrue(V.points_disagree("precip_sum_mm", pt(500), pt(520)))
        self.assertFalse(V.points_disagree("precip_sum_mm", pt(500), pt(519.9)))
        self.assertTrue(V.points_disagree("precip_max_day_mm", pt(40), pt(37)))
        self.assertFalse(V.points_disagree("wind_max_kmh", pt(60), pt(61.9)))
        self.assertTrue(V.points_disagree("hot_days_ge_30c", pt(10), pt(12)))
        self.assertFalse(V.points_disagree("frost_days_lt_0c", pt(10), pt(11)))
        self.assertTrue(V.points_disagree("rainy_days_ge_1mm", pt(150), pt(148)))

    def test_point_never_set_disjoint(self):
        # A point is only compared with another point, never with a threshold or interval.
        self.assertFalse(V.assertions_disjoint("temp_mean_c", pt(5), th("gt", 10)))
        self.assertFalse(V.assertions_disjoint("temp_mean_c", pt(5), iv(10, 20)))
        self.assertFalse(V.points_disagree("temp_mean_c", pt(5), th("gt", 10)))


class ValidationTests(unittest.TestCase):
    def test_window_must_be_in_year_x_plus_1(self):
        for ws, we in [(f"{X}-12-31", f"{Y}-01-31"), (f"{Y}-06-01", f"{Y + 1}-01-01"),
                       (f"{Y}-07-01", f"{Y}-06-01"), (f"{Y}-02-30", f"{Y}-03-01")]:
            with self.assertRaises(V.ClaimError):
                V.validate_claim(claim("x", ws=ws, we=we), X)
        V.validate_claim(claim("x", ws=f"{Y}-01-01", we=f"{Y}-12-31"), X)

    def test_assertion_kind_must_match_variable(self):
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", var="warmer_than_year_x"), X)
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", assertion={"kind": "boolean", "value": True}), X)
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", var="hottest_month", assertion={"kind": "categorical", "value": "13"}), X)
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", assertion=iv(5, 1)), X)
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", assertion=th("gt", True)), X)
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", assertion=th("eq", 1)), X)

    def test_month_normalized(self):
        c = V.validate_claim(claim("x", var="hottest_month", assertion={"kind": "categorical", "value": "7"}), X)
        self.assertEqual(c["assertion"]["value"], "07")

    def test_locations(self):
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", lat=40.0), X)  # london with wrong lat
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", loc="atlantis"), X)
        g = V.validate_claim(claim("x", loc="grid_10.50_-20.25", lat=10.5, lon=-20.25), X)
        self.assertEqual((g["target"]["lat"], g["target"]["lon"]), (10.5, -20.25))
        with self.assertRaises(V.ClaimError):
            V.validate_claim(claim("x", loc="grid_10.5_-20.25", lat=10.5, lon=-20.25), X)

    def test_antispam_duplicates_and_total_cap(self):
        raw = [claim("a"), claim("dup", assertion=th("lt", 0)), claim("bad", var="nope")]
        # 20 locations x 11 windows = 220 more claims, at most 12 per location including "a".
        raw += [claim(f"{loc}{d}", loc=loc, ws=f"{Y}-01-01",
                      we=(dt.date(Y, 1, 2) + dt.timedelta(days=d)).isoformat())
                for loc in V.CATALOG for d in range(11)]
        acc, rej = V.filter_claims("predict3", raw, X)
        self.assertEqual(len(acc), 150)
        self.assertEqual(acc[0]["claim_id"], "a")  # first claim per key wins
        reasons = [r["reason"] for r in rej]
        self.assertTrue(reasons[0].startswith("duplicate"))
        self.assertTrue(reasons[1].startswith("invalid"))
        self.assertEqual(sum("per predictor" in r for r in reasons), 221 - 150)
        self.assertFalse(any("claims for" in r for r in reasons))  # no location exceeded 12

    def test_per_location_cap(self):
        # 15 London claims then 1 Tokyo claim: claims 13-15 for London are rejected, Tokyo kept.
        raw = [claim(f"l{d}", ws=f"{Y}-01-01", we=(dt.date(Y, 1, 1) + dt.timedelta(days=d)).isoformat())
               for d in range(15)]
        raw.append(claim("t0", loc="tokyo_jp"))
        acc, rej = V.filter_claims("predict3", raw, X)
        self.assertEqual([c["claim_id"] for c in acc], [f"l{d}" for d in range(12)] + ["t0"])
        self.assertEqual([r["claim_id"] for r in rej], ["l12", "l13", "l14"])
        self.assertTrue(all(r["reason"] == "exceeds 12 claims for london_uk" for r in rej))
        # An invalid or duplicate claim does not use up a location's quota.
        raw = [claim("bad", assertion=th("eq", 1)), claim("d0"), claim("d0dup")]
        raw += [claim(f"x{d}", we=(dt.date(Y, 6, 1) + dt.timedelta(days=d)).isoformat()) for d in range(11)]
        acc, _ = V.filter_claims("predict3", raw, X)
        self.assertEqual(len(acc), 12)

    def test_point_validation(self):
        c = V.validate_claim(claim("x", assertion={"kind": "point", "value": 15.3}), X)
        self.assertEqual(c["assertion"], {"kind": "point", "value": 15.3})
        V.validate_claim(claim("x", var="hot_days_ge_30c", assertion={"kind": "point", "value": 4}), X)
        for bad in ({"kind": "point", "value": "15"}, {"kind": "point", "value": True},
                    {"kind": "point", "value": float("nan")}, {"kind": "point"}):
            with self.assertRaises(V.ClaimError):
                V.validate_claim(claim("x", assertion=bad), X)
        with self.assertRaises(V.ClaimError):  # not for booleans / categoricals
            V.validate_claim(claim("x", var="warmer_than_year_x", assertion={"kind": "point", "value": 1}), X)

    def test_load_missing_empty_bad(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "e.json").write_text("  \n")
            (p / "b.json").write_text("{not json")
            (p / "o.json").write_text('{"a": 1}')
            self.assertEqual(V.load_predictor_file(p / "m.json"), ([], "missing"))
            self.assertEqual(V.load_predictor_file(p / "e.json"), ([], "empty"))
            self.assertEqual(V.load_predictor_file(p / "b.json")[0], [])
            self.assertEqual(V.load_predictor_file(p / "o.json")[0], [])


class ComputeTests(unittest.TestCase):
    def test_numeric_variables_hand_computed(self):
        vals = [  # mean, max, min, precip, wind
            (10.0, 31.0, -1.0, 0.5, 20.0),
            (12.0, 30.0, 0.0, 1.0, 35.5),
            (14.0, 29.9, 2.0, 12.25, 10.0),
            (16.0, 25.0, -0.1, 0.99, 40.0),
        ]
        start = dt.date(Y, 3, 1)
        daily = {(start + dt.timedelta(days=i)).isoformat(): dict(zip(V.DAILY_FIELDS, v))
                 for i, v in enumerate(vals)}
        end = start + dt.timedelta(days=3)
        c = lambda var: V.compute_variable(var, daily, start, end, X)
        self.assertEqual(c("temp_mean_c"), 13.0)
        self.assertEqual(c("temp_max_c"), 31.0)
        self.assertEqual(c("temp_min_c"), -1.0)
        self.assertEqual(c("temp_mean_max_c"), 28.975)
        self.assertEqual(c("temp_mean_min_c"), 0.225)
        self.assertEqual(c("precip_sum_mm"), 14.74)
        self.assertEqual(c("precip_max_day_mm"), 12.25)
        self.assertEqual(c("wind_max_kmh"), 40.0)
        self.assertEqual(c("hot_days_ge_30c"), 2)    # 31.0 and 30.0; 29.9 excluded
        self.assertEqual(c("frost_days_lt_0c"), 2)   # -1.0 and -0.1; 0.0 excluded
        self.assertEqual(c("rainy_days_ge_1mm"), 2)  # 1.0 and 12.25; 0.99 excluded
        # Window extending past data, or a null value, is unresolvable.
        self.assertIsNone(V.compute_variable("temp_mean_c", daily, start, end + dt.timedelta(days=1), X))
        daily[start.isoformat()]["temperature_2m_mean"] = None
        self.assertIsNone(c("temp_mean_c"))
        self.assertEqual(c("temp_max_c"), 31.0)  # other fields still fine

    def test_monthly_and_year_comparison(self):
        def fn(d):
            # July hottest in Y; August wettest; X is uniformly cooler and wetter.
            base = 20.0 if d.month == 7 else 10.0
            rain = 5.0 if d.month == 8 else 1.0
            if d.year == X:
                return dict(zip(V.DAILY_FIELDS, (base - 1, base, base - 2, rain * 2, 10.0)))
            return dict(zip(V.DAILY_FIELDS, (base, base + 1, base - 1, rain, 10.0)))
        daily = synthetic_daily(dt.date(X, 1, 1), dt.date(Y, 12, 31), fn)
        s, e = dt.date(Y, 6, 1), dt.date(Y, 9, 30)
        self.assertEqual(V.compute_variable("hottest_month", daily, s, e, X), "07")
        self.assertEqual(V.compute_variable("wettest_month", daily, s, e, X), "08")
        self.assertIs(V.compute_variable("warmer_than_year_x", daily, s, e, X), True)
        self.assertIs(V.compute_variable("wetter_than_year_x", daily, s, e, X), False)
        # Ties in argmax are unresolvable.
        self.assertIsNone(V.compute_variable("hottest_month", daily, dt.date(Y, 1, 1), dt.date(Y, 2, 28), X))

    def test_equal_years_is_not_warmer(self):
        daily = synthetic_daily(dt.date(X, 1, 1), dt.date(Y, 12, 31),
                                lambda d: dict(zip(V.DAILY_FIELDS, (5.0, 6.0, 4.0, 1.0, 10.0))))
        self.assertIs(V.compute_variable("warmer_than_year_x", daily, dt.date(Y, 1, 1), dt.date(Y, 1, 31), X), False)

    def test_leap_day_mapping(self):
        self.assertEqual(V._same_window_in_year(dt.date(2024, 2, 29), dt.date(2024, 2, 29), 2023),
                         (dt.date(2023, 3, 1), dt.date(2023, 2, 28)))
        self.assertEqual(V._same_window_in_year(dt.date(2024, 2, 1), dt.date(2024, 2, 29), 2023),
                         (dt.date(2023, 2, 1), dt.date(2023, 2, 28)))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        # Ground truth: London June mean temp exactly 10.0; 20 days with >= 1 mm rain.
        def fn(d):
            return dict(zip(V.DAILY_FIELDS, (10.0, 12.0, 8.0, 2.0 if d.day <= 20 else 0.0, 15.0)))
        self.daily = synthetic_daily(dt.date(Y, 1, 1), dt.date(Y, 12, 31), fn)
        self.calls = []

    def fetch(self, coords, start, end):
        self.calls.append((tuple(coords), start, end))
        return {c: self.daily for c in coords if c != V.CATALOG["tokyo_jp"]}

    def write(self, td, files):
        for name, obj in files.items():
            (Path(td) / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            self.write(td, {
                "predict1.json": [claim("p1a", assertion=th("gt", 10)),          # wrong (10 is not > 10)
                                  claim("p1b", var="rainy_days_ge_1mm", assertion=th("ge", 20))],
                "predict2.json": [claim("p2a", assertion=th("le", 10)),          # right
                                  claim("p2b", var="rainy_days_ge_1mm", assertion=th("lt", 20)),
                                  claim("p2c", loc="tokyo_jp", assertion=th("gt", 15))],
                "predict3.json": [claim("p3a", assertion=iv(11, 12)),           # wrong; overlaps p1a (no contradiction)
                                  claim("p3c", loc="tokyo_jp", assertion=th("lt", 15))],
                "predict4.json": [claim("p4a", assertion=th("le", 10.5))],       # right; overlaps p1a and p2a
                "predict5.json": "",
                "predict6.json": "garbage",
            })
            r = V.run(X, Path(td), self.fetch)
        self.assertEqual(r["year_x_plus_1"], Y)
        pairs = {(c["a"]["claim_id"], c["b"]["claim_id"]): c for c in r["contradictions"]}
        self.assertEqual(set(pairs), {("p1a", "p2a"), ("p2a", "p3a"), ("p3a", "p4a"),
                                      ("p1b", "p2b"), ("p2c", "p3c")})
        self.assertEqual(pairs[("p1a", "p2a")]["status"], "b_wins")
        self.assertEqual(pairs[("p2a", "p3a")]["status"], "a_wins")
        self.assertEqual(pairs[("p3a", "p4a")]["status"], "b_wins")
        self.assertEqual(pairs[("p1b", "p2b")]["status"], "a_wins")
        self.assertEqual(pairs[("p2c", "p3c")]["status"], "unresolvable")
        self.assertEqual(r["contradictions_count"], 5)
        self.assertEqual(r["resolved_count"], 4)
        self.assertEqual(r["scores_delta"], {"predict1": 0, "predict2": 1, "predict3": -2, "predict4": 1,
                                             **{f"predict{i}": 0 for i in range(5, 11)}})
        self.assertEqual(sum(r["scores_delta"].values()), 0)
        self.assertEqual(r["intake"]["predict5"]["file_status"], "empty")
        self.assertEqual(r["intake"]["predict7"]["file_status"], "missing")
        self.assertTrue(r["intake"]["predict6"]["file_status"].startswith("invalid JSON"))
        # One batched year X+1 fetch, no year X fetch (no boolean targets).
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1:], (f"{Y}-01-01", f"{Y}-12-31"))

    def test_point_duels(self):
        # Truth: June mean temperature 10.0, June precip 20 * 2.0 = 40 mm.
        with tempfile.TemporaryDirectory() as td:
            self.write(td, {
                "predict1.json": [claim("a1", assertion=pt(10.2)),                       # error 0.2
                                  claim("a2", var="precip_sum_mm", assertion=pt(30.0)),  # error 10
                                  claim("a3", var="temp_max_c", assertion=pt(11.5))],    # error 0.5
                "predict2.json": [claim("b1", assertion=pt(9.7)),                        # error 0.3
                                  claim("b2", var="precip_sum_mm", assertion=pt(50.0)),  # error 10 -> tie
                                  claim("b3", var="temp_max_c", assertion=pt(12.2))],    # gap 0.7
                "predict3.json": [claim("c1", assertion=pt(10.05)),                      # 0.15 from p1: no duel
                                  claim("c2", var="precip_sum_mm", assertion=pt(45.0)),  # < 20 from both
                                  claim("c3", var="temp_max_c", assertion=th("gt", 12))],
            })
            r = V.run(X, Path(td), self.fetch)
        pairs = {(c["a"]["claim_id"], c["b"]["claim_id"]): c for c in r["contradictions"]}
        # c1 vs b1: gap 0.35 -> duel, c1 error 0.05 beats b1 error 0.3. a1 vs c1: gap 0.15 -> none.
        self.assertEqual(set(pairs), {("a1", "b1"), ("b1", "c1"), ("a2", "b2"), ("a3", "b3")})
        self.assertEqual(pairs[("a1", "b1")]["status"], "a_wins")
        self.assertEqual(pairs[("b1", "c1")]["status"], "b_wins")
        self.assertEqual(pairs[("a2", "b2")]["status"], "tie")
        self.assertEqual((pairs[("a2", "b2")]["delta_a"], pairs[("a2", "b2")]["delta_b"]), (0, 0))
        self.assertEqual(pairs[("a3", "b3")]["status"], "b_wins")  # truth 12.0: errors 0.5 vs 0.2
        self.assertEqual(r["scores_delta"]["predict1"], 0)    # +1 (a1), -1 (a3)
        self.assertEqual(r["scores_delta"]["predict2"], -1)   # -1 (b1 vs a1), -1 (b1 vs c1), +1 (b3)
        self.assertEqual(pairs[("a1", "b1")]["kind"], "point")
        self.assertEqual(r["resolved_count"], 4)
        self.assertEqual(r["scores_delta"]["predict3"], 1)
        self.assertEqual(sum(r["scores_delta"].values()), 0)

    def test_point_duel_unresolvable(self):
        with tempfile.TemporaryDirectory() as td:
            self.write(td, {"predict1.json": [claim("a", loc="tokyo_jp", assertion=pt(20.0))],
                            "predict2.json": [claim("b", loc="tokyo_jp", assertion=pt(21.0))]})
            r = V.run(X, Path(td), self.fetch)
        self.assertEqual(r["contradictions"][0]["status"], "unresolvable")
        self.assertEqual(r["resolved_count"], 0)

    def test_both_wrong_is_draw(self):
        with tempfile.TemporaryDirectory() as td:
            self.write(td, {"predict1.json": [claim("a", assertion=iv(0, 5))],
                            "predict9.json": [claim("b", assertion=iv(11, 20))]})
            r = V.run(X, Path(td), self.fetch)
        self.assertEqual(r["contradictions"][0]["status"], "both_wrong")
        self.assertEqual(r["resolved_count"], 1)
        self.assertTrue(all(v == 0 for v in r["scores_delta"].values()))

    def test_different_windows_do_not_contradict(self):
        with tempfile.TemporaryDirectory() as td:
            self.write(td, {"predict1.json": [claim("a", assertion=th("gt", 10))],
                            "predict2.json": [claim("b", assertion=th("le", 10), we=f"{Y}-06-29")]})
            r = V.run(X, Path(td), self.fetch)
        self.assertEqual(r["contradictions_count"], 0)
        self.assertEqual(self.calls, [])

    def test_boolean_fetches_year_x(self):
        with tempfile.TemporaryDirectory() as td:
            b = lambda v: {"kind": "boolean", "value": v}
            self.write(td, {"predict1.json": [claim("a", var="warmer_than_year_x", assertion=b(True))],
                            "predict2.json": [claim("b", var="warmer_than_year_x", assertion=b(False))]})
            r = V.run(X, Path(td), self.fetch)
        self.assertEqual(self.calls[0][1:], (f"{X}-01-01", f"{Y}-12-31"))
        # Synthetic daily has no year X data -> unresolvable, not a guess.
        self.assertEqual(r["contradictions"][0]["status"], "unresolvable")


class FetchTests(unittest.TestCase):
    def test_batching_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            a = V.OpenMeteoArchive(Path(td), batch_size=2)
            sent = []

            def fake_get(params):
                lats = params["latitude"].split(",")
                sent.append(len(lats))
                pl = {"daily": {"time": ["2023-01-01"], **{f: [1.0] for f in V.DAILY_FIELDS}}}
                return [pl] * len(lats) if len(lats) > 1 else pl
            a._get = fake_get
            coords = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]
            out = a.fetch(coords, "2023-01-01", "2023-01-01")
            self.assertEqual(sent, [2, 1])
            self.assertEqual(len(out), 3)
            out2 = a.fetch(coords, "2023-01-01", "2023-01-01")
            self.assertEqual(sent, [2, 1])  # served from cache
            self.assertEqual(out, out2)


if __name__ == "__main__":
    unittest.main()
