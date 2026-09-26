"""Regression tests for elapsed-time matching of detected solar-wind peaks."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from utils.skill_scores import compute_peaks


def profile(centres, size=400, radius=30):
    """Create separated peaks broad enough for the production peak detector."""
    values = np.full(size, 300.0)
    positions = np.arange(size)
    for centre in centres:
        values += 400.0 * np.maximum(1.0 - np.abs(positions - centre) / radius, 0.0)
    return pd.Series(values)


class TestPeakMatching(unittest.TestCase):
    def run_peaks(self, observed, predicted, tolerance, frequency="h"):
        dates = pd.Series(pd.date_range("2026-01-01", periods=400, freq=frequency))
        obs = profile(observed)
        pred = profile(predicted)
        before = [series.copy(deep=True) for series in (dates, obs, pred)]
        result = compute_peaks(dates, obs, pred, 400, tolerance)
        np.testing.assert_array_equal(result[0], observed)
        np.testing.assert_array_equal(result[1], predicted)
        for series, original in zip((dates, obs, pred), before):
            assert_series_equal(series, original)
        return result

    def test_equal_early_and_late_offsets_match_within_fractional_day(self):
        for hours in (-18, -12, -6, 6, 12, 18):
            with self.subTest(hours=hours):
                result = self.run_peaks([180], [180 + hours], 0.75)
                self.assertEqual(list(result[2]), [180 + hours])
                self.assertEqual(list(result[3]), [])
                self.assertEqual(list(result[4]), [])
                self.assertEqual(result[5], [float(hours)])
                self.assertEqual(result[6], [0.0])

    def test_offsets_beyond_tolerance_are_rejected_in_both_directions(self):
        for hours, tolerance in ((-20, 0.5), (20, 0.5), (-25, 1), (25, 1), (47, 1)):
            with self.subTest(hours=hours, tolerance=tolerance):
                result = self.run_peaks([180], [180 + hours], tolerance)
                self.assertEqual(list(result[2]), [])
                self.assertEqual(list(result[3]), [180])
                self.assertEqual(list(result[4]), [180 + hours])
                self.assertEqual(result[5], [])
                self.assertEqual(result[6], [])

    def test_nearest_candidate_wins_within_the_same_rounded_day(self):
        result = self.run_peaks([180], [100, 160], 1, frequency="15min")
        self.assertEqual(list(result[2]), [160])
        self.assertEqual(list(result[3]), [])
        self.assertEqual(list(result[4]), [100])
        self.assertEqual(result[5], [-5.0])

    def test_whole_day_and_zero_offset_boundaries_keep_existing_matches(self):
        for hours, tolerance in ((-48, 2), (-24, 1), (0, 0), (24, 1), (48, 2)):
            with self.subTest(hours=hours, tolerance=tolerance):
                result = self.run_peaks([180], [180 + hours], tolerance)
                self.assertEqual(list(result[2]), [180 + hours])
                self.assertEqual(result[5], [float(hours)])
                self.assertEqual(list(result[3]), [])
                self.assertEqual(list(result[4]), [])

    def test_zero_tolerance_rejects_noncoincident_peaks(self):
        for hours in (-1, 1):
            with self.subTest(hours=hours):
                result = self.run_peaks([180], [180 + hours], 0)
                self.assertEqual(list(result[2]), [])
                self.assertEqual(list(result[3]), [180])
                self.assertEqual(list(result[4]), [180 + hours])

    def test_empty_peak_sets_keep_output_structure(self):
        for observed, predicted in (([], []), ([180], []), ([], [180])):
            with self.subTest(observed=observed, predicted=predicted):
                result = self.run_peaks(observed, predicted, 1)
                self.assertEqual(len(result), 7)
                self.assertEqual(list(result[2]), [])
                self.assertEqual(list(result[3]), observed)
                self.assertEqual(list(result[4]), predicted)
                self.assertEqual(result[5:], ([], []))

    def test_timezone_aware_dates_preserve_labels_and_signed_errors(self):
        labels = pd.Index([f"sample-{i}" for i in range(400)])
        dates = pd.Series(
            pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC"), index=labels
        )
        obs = profile([180])
        pred = profile([168]) + 10.0
        result = compute_peaks(dates, obs, pred, 400, 0.5)
        self.assertEqual(list(result[2]), ["sample-168"])
        self.assertEqual(result[5], [-12.0])
        self.assertEqual(result[6], [10.0])


if __name__ == "__main__":
    unittest.main()
