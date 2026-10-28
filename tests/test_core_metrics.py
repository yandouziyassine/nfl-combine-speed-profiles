"""Controlled unit fixtures for core transformations.

These small hand-built traces validate code behavior only. They are never
written to project outputs or used as NFL/Combine observations.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_dl_game_burst import summarize_player_play
from analyze_40_repeatability import icc_absolute_single
from run_analysis import calculate_attempt, interpolate_at


class CoreMetricTests(unittest.TestCase):
    def test_interpolate_at_known_value_and_out_of_range(self) -> None:
        self.assertAlmostEqual(interpolate_at(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 4.0]), 0.5), 1.0)
        self.assertTrue(np.isnan(interpolate_at(np.array([0.0, 1.0]), np.array([2.0, 3.0]), 1.1)))

    def test_icc_perfect_absolute_agreement(self) -> None:
        samples = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [7.0, 7.0]])
        self.assertAlmostEqual(icc_absolute_single(samples), 1.0)

    def test_combine_attempt_detects_sampled_movement_and_10yd_crossing(self) -> None:
        t = np.arange(0.0, 6.1, 0.1)
        speed = np.maximum(0.0, 1.8 * t)
        x = np.cumsum(speed * 0.1)
        trace = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=len(t), freq="100ms").astype(str),
            "s": speed, "x": x, "y": np.zeros_like(x),
        })
        result = calculate_attempt(trace)
        self.assertGreaterEqual(result["movement_onset_from_recording_s"], 0.0)
        self.assertGreater(result["speed_at_10yd_ydps"], 0.0)
        self.assertGreater(result["time_to_10yd_s"], 0.0)
        self.assertTrue(np.isfinite(result["early_accel_0_1s_ydps2"]))

    def test_game_snap_timestamp_keeps_tenth_second_scale(self) -> None:
        times = pd.date_range("2025-09-01T12:00:00", periods=21, freq="100ms")
        rows = []
        for i, stamp in enumerate(times):
            event = "ball_snap" if i == 10 else ""
            speed = 0.0 if i <= 10 else min(4.0, (i - 10) * 0.6)
            # Match the source's consistent fractional-second timestamp form.
            rows.append((stamp.strftime("%Y-%m-%dT%H:%M:%S.%f"), speed, float(i) * 0.05, 0.0, event))
        result = summarize_player_play(rows)
        self.assertEqual(result["game_tracking_status"], "usable")
        self.assertEqual(result["snap_time_resolution_s"], 0.1)
        self.assertTrue(np.isfinite(result["speed_0.5s_after_snap_ydps"]))
        self.assertLess(result["speed_1s_after_snap_ydps"], 10.0)


if __name__ == "__main__":
    unittest.main()
