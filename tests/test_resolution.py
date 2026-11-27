"""Behavior checks for observation cutoffs, temporal evaluation and inference."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.resolution import CAP, CATEGORIES, MODEL_NAMES, prepare, predict, train


class ResolutionTests(unittest.TestCase):
    def test_cutoff_and_invalid_records(self):
        frame = pd.DataFrame({
            "unique_key": ["closed", "open", "future", "recent", "negative", "bad"],
            "created_date": ["2025-01-01"] * 3 + ["2025-05-01", "2025-01-01", "bad"],
            "closed_date": ["2025-01-02", None, "2025-07-01", None, "2024-12-31", None],
            **{col: [None] * 6 for col in CATEGORIES}})
        clean, audit = prepare(frame, "2025-06-01")
        self.assertEqual(clean.unique_key.tolist(), ["closed", "open", "future"])
        self.assertEqual(clean.hours_to_close_capped.tolist(), [24, CAP, CAP])
        self.assertEqual(audit["invalid_rows"], 1)
        self.assertEqual(audit["outside_study_rows"], 1)
        self.assertEqual(audit["insufficient_followup_rows"], 1)

    def test_train_save_load_unseen_categories(self):
        rng = np.random.default_rng(42)
        dates = pd.DatetimeIndex(list(pd.date_range("2025-01-05", periods=150, freq="h"))
                                 + list(pd.date_range("2026-01-05", periods=80, freq="h")))
        frame = pd.DataFrame({"unique_key": np.arange(len(dates)).astype(str),
                              "created_date": dates.astype(str),
                              "closed_date": (dates + pd.to_timedelta(rng.uniform(1, 80, len(dates)), unit="h")).astype(str),
                              **{col: ["example"] * len(dates) for col in CATEGORIES}})
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.joblib"
            scores = train(frame, "2026-10-31T23:59:59-04:00", model_path=model, report_dir=Path(directory))
            self.assertEqual(len(scores), 4)
            bundle = joblib.load(model)
            result = predict(bundle, "2026-01-01 12:00", agency="unseen", incident_zip="99999")
            self.assertTrue(0 <= result["estimated_hours"] <= CAP)
            self.assertEqual(bundle["metadata"]["train_rows"], 150)
            stale = {**bundle, "metadata": {**bundle["metadata"], "study_id": "previous-study"}}
            with self.assertRaises(ValueError):
                predict(stale, "2026-01-01")
            self.assertEqual(set(bundle["models"]), set(MODEL_NAMES))
            for name in MODEL_NAMES:
                result = predict(bundle, "2026-01-01 12:00", estimator=name, agency="unseen")
                self.assertTrue(0 <= result["estimated_hours"] <= CAP)
                self.assertEqual(result["model"], name)
            with self.assertRaises(ValueError):
                predict(bundle, "2026-01-01", estimator="unavailable")
            with self.assertRaises(ValueError):
                predict(bundle, "2025-01-01", closed_date="2025-01-02")
            with self.assertRaises(ValueError):
                predict(bundle, "2025-03-09 02:30")
        with self.assertRaises(ValueError):
            train(frame, "2025-12-01")
        with self.assertRaises(ValueError):
            train(frame, "2026-05-01", test_start="2025-07")


if __name__ == "__main__":
    unittest.main()
