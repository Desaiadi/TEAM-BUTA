"""Study boundaries and housing feature/target separation."""
import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from src import config
from src.features_housing import build_snapshot, feature_columns
from src.study import in_window
from src.resolution import prepare, CATEGORIES


class StudyTests(unittest.TestCase):
    def test_inclusive_october_and_exclusive_november(self):
        dates = pd.Series(["2024-12-31", "2025-01-01", "2026-10-31T23:59:59", "2026-11-01"])
        self.assertEqual(in_window(dates).tolist(), [False, True, True, False])

    def test_closure_after_study_cutoff_is_unobserved(self):
        data = pd.DataFrame({"unique_key": ["old", "recent", "outside"],
                             "created_date": ["2026-07-01", "2026-10-01", "2024-12-01"],
                             "closed_date": ["2026-11-02", None, "2024-12-02"],
                             **{c: [None] * 3 for c in CATEGORIES}})
        clean, audit = prepare(data, "2026-12-31")
        self.assertEqual(clean.unique_key.tolist(), ["old"])
        self.assertEqual(clean.hours_to_close_capped.tolist(), [2160])
        self.assertEqual(audit["insufficient_followup_rows"], 1)
        self.assertEqual(audit["as_of"], pd.Timestamp(config.STUDY_AS_OF).isoformat())

    def test_housing_history_does_not_include_targets(self):
        b = pd.DataFrame({"bbl": ["1000010001"], "yearbuilt": [1900], "unitsres": [10],
                          "landlord_id": ["L1"], "portfolio_buildings": [1]})
        dates = pd.to_datetime(["2024-12-31", "2025-01-01", "2025-06-30", "2025-07-01",
                                "2025-10-31", "2025-11-01"])
        comp = pd.DataFrame({"bbl": b.bbl.iloc[0], "complaint_id": range(6), "year": dates.year,
                             "received_date": dates, "problem_type": "heat/hot water", "is_emergency": False})
        inp = {"complaints": comp,
               "violations": pd.DataFrame({"bbl": b.bbl.iloc[0], "inspectiondate": dates,
                                            "year": dates.year, "class": "C", "problem_type": "heat/hot water"}),
               "bedbugs": pd.DataFrame({"bbl": b.bbl.iloc[0], "period_end": dates, "filing_date": dates,
                                        "year": dates.year, "has_infestation": True}),
               "evictions": pd.DataFrame({"bbl": b.bbl.iloc[0], "executed_date": dates, "year": dates.year}),
               "crime": None}
        snap = build_snapshot("train", inp, b, sparse.csr_matrix((1, 1)))
        self.assertEqual(snap.c_heat_hot_water.iloc[0], 2)
        self.assertEqual(snap.v_class_C.iloc[0], 2)
        self.assertEqual(snap.evictions.iloc[0], 2)
        self.assertEqual(snap.y_heat_hot_water.iloc[0], 1)
        self.assertFalse(set(["history_start", "target_end", "window"]) & set(feature_columns(snap)))
        forecast = build_snapshot("forecast", inp, b, sparse.csr_matrix((1, 1)))
        self.assertFalse(any(c.startswith("y_") for c in forecast))
        crime_dates = pd.to_datetime(["2026-05-01", "2026-09-30"])
        inp["crime"] = pd.DataFrame({"rpt_dt": crime_dates, "year": 2026,
                                     "latitude": 40.7, "longitude": -73.99,
                                     "severity": "FELONY", "crime_group": "violent", "in_residence": True})
        b["latitude"], b["longitude"] = 40.7, -73.99
        forecast = build_snapshot("forecast", inp, b, sparse.csr_matrix((1, 1)))
        self.assertAlmostEqual(forecast.crime_total.iloc[0], 2 * 365.25 / 61)
        self.assertLess(forecast.crime_exposure_fraction.iloc[0], 1)
        self.assertNotIn("crime_observed_end", feature_columns(forecast))
        with self.assertRaises(ValueError):
            build_snapshot(2024, inp, b, sparse.csr_matrix((1, 1)))


if __name__ == "__main__":
    unittest.main()
