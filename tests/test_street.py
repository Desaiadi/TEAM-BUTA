"""Behavior checks for the noise & sanitation analysis (src/street.py), on small made-up data."""
import unittest

import numpy as np
import pandas as pd

from src import street

# Two apartment buildings about 150 m apart on the Upper East Side, one district.
LAT, LON = 40.7820, -73.9500
BUILDINGS = pd.DataFrame({
    "bbl": ["1000000001", "1000000002"], "address": ["1 EAST 1 STREET", "2 EAST 2 STREET"],
    "borough": ["MANHATTAN"] * 2, "zipcode": ["10128"] * 2, "unitsres": [10, 20], "cd": [108, 108],
    "district": ["Manhattan Community District 8"] * 2,
    "latitude": [LAT, LAT + 0.00135], "longitude": [LON, LON]})


def raw_complaints() -> pd.DataFrame:
    """30 noise complaints next to building 1 (mostly at night), 3 next to building 2, plus bad rows."""
    rows = []
    for i in range(30):
        hour = 23 if i < 24 else 14
        rows.append((f"n{i}", f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}T{hour:02d}:00:00.000",
                     "Noise - Residential", "Loud Music/Party", LAT, LON))
    for i in range(3):
        rows.append((f"b{i}", f"2026-03-1{i}T15:00:00.000", "Rodent", "Rat Sighting", LAT + 0.00135, LON))
    rows += [("n0", "2025-01-01T23:00:00.000", "Noise - Residential", "Loud Music/Party", LAT, LON),  # duplicate key
             ("again", "2025-01-01T23:30:00.000", "Noise - Residential", "Loud Music/Party", LAT, LON),  # same-day repeat
             ("old", "2024-06-01T12:00:00.000", "Rodent", "Rat Sighting", LAT, LON),                 # before the study
             ("nowhere", "2025-05-01T12:00:00.000", "Rodent", "Rat Sighting", None, None),           # no location
             ("heli", "2025-05-01T12:00:00.000", "Noise - Helicopter", "Other", LAT, LON)]           # not a street type
    frame = pd.DataFrame(rows, columns=["unique_key", "created_date", "complaint_type", "descriptor",
                                        "latitude", "longitude"])
    frame["incident_address"] = "SOMEWHERE"
    return frame.astype({"latitude": "object", "longitude": "object"})


class StreetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complaints, cls.report = street.clean(raw_complaints())
        cls.buildings = street.building_counts(BUILDINGS, cls.complaints)
        cls.index = street.StreetIndex(cls.complaints, cls.buildings)

    def test_cleaning_drops_bad_rows(self):
        removed = {s["step"]: s["rows_removed"] for s in self.report.steps}
        self.assertEqual(removed["drop_duplicate_keys"], 1)
        self.assertEqual(removed["study_window"], 1)
        self.assertEqual(removed["known_types"], 1)
        self.assertEqual(removed["valid_location"], 1)
        self.assertEqual(removed["same_day_repeats"], 1)
        self.assertEqual(len(self.complaints), 33)
        self.assertEqual(set(self.complaints["group"]), {"noise", "sanitation"})

    def test_labels_are_tidy(self):
        self.assertEqual(street.tidy_label("Noise: Construction Before/After Hours (NM1)"),
                         "Construction Before/After Hours")
        self.assertEqual(street.tidy_label("Loud Music/Party"), "Loud Music/Party")

    def test_counts_within_100m(self):
        b = self.buildings.set_index("bbl")
        # building 1 sees its own 30 noise complaints; building 2 (150 m away) sees none of them
        self.assertAlmostEqual(b.loc["1000000001", "noise_per_year"], 30 / street.study_years())
        self.assertEqual(b.loc["1000000002", "noise_per_year"], 0)
        self.assertEqual(b.loc["1000000002", "sanitation_2026"], 3)

    def test_midrank_percentile(self):
        ref = np.array([0, 0, 0, 10])
        self.assertAlmostEqual(street.midrank_percentile(ref, 0), 0.375)    # ties count half
        self.assertAlmostEqual(street.midrank_percentile(ref, 20), 1.0)
        self.assertTrue(np.isnan(street.midrank_percentile(np.array([]), 1)))

    def test_profile(self):
        p = self.index.profile(LAT, LON)
        noise = p["groups"]["noise"]
        self.assertEqual(noise["count"], 30)
        self.assertAlmostEqual(noise["night_share"], 24 / 30)
        self.assertEqual(noise["top"][0]["label"], "Loud Music/Party")
        self.assertEqual(sum(noise["by_hour"]), 30)
        self.assertEqual(p["district"], "Manhattan Community District 8")

    def test_wider_radius_sees_more(self):
        # at 200 m building 2 (149 m away) now sees building 1's noise too
        b = self.buildings.set_index("bbl")
        self.assertAlmostEqual(b.loc["1000000002", "noise_per_year_200m"], 30 / street.study_years())
        near, wide = self.index.profile(LAT, LON, 100), self.index.profile(LAT, LON, 500)
        self.assertGreaterEqual(wide["groups"]["sanitation"]["count"], near["groups"]["sanitation"]["count"])
        self.assertIn("within 500 m", street.profile_report(wide))
        with self.assertRaises(ValueError):
            self.index.profile(LAT, LON, 300)

    def test_compare_report_names_the_quieter_building(self):
        pa, pb = self.index.profile(LAT, LON), self.index.profile(LAT + 0.00135, LON)
        text = street.compare_report(pa, pb, "Building A", "Building B")
        self.assertTrue(text.startswith("These two buildings, 149 meters apart"))
        self.assertIn("Building B looks like the better choice", text)
        self.assertIn("more than ten times as many noise complaints as Building B", text)
        self.assertIn("mostly loud music/party around Building A", text)
        self.assertNotIn("nan", text.lower())
        self.assertLessEqual(text.count(". "), 5)      # short: the verdict and a few reasons

    def test_within_between_shares_add_up(self):
        cells = street.cell_table(self.buildings, self.complaints)
        q1 = street.within_between(cells, "noise")
        self.assertAlmostEqual(q1["within_share"] + q1["between_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
