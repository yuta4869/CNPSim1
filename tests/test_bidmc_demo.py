"""Tests for leakage-safe BIDMC GUI inference helpers."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from cnp import CNP
from cnp.bidmc_demo import (
    BIDMCDemoRuntime,
    WindowEntry,
    centered_gap_indices,
    linear_interpolate,
    predict_window,
)
from cnp.datasets import BIDMCWindow, HeartRateStandardizer


class GapIndicesTests(unittest.TestCase):
    def test_centered_gap_has_exact_width_and_contains_query(self) -> None:
        context, gap, start, stop = centered_gap_indices(120, 60, 30)
        self.assertEqual((start, stop), (46, 76))
        self.assertEqual(gap.tolist(), list(range(46, 76)))
        self.assertIn(60, gap.tolist())
        self.assertEqual(context.numel(), 90)
        self.assertIn(0, context.tolist())
        self.assertIn(119, context.tolist())

    def test_gap_shifts_at_query_boundaries_without_shrinking(self) -> None:
        for query, expected in ((5, (5, 65)), (114, (55, 115))):
            context, gap, start, stop = centered_gap_indices(120, query, 60)
            self.assertEqual((start, stop), expected)
            self.assertEqual(gap.numel(), 60)
            self.assertIn(query, gap.tolist())
            self.assertTrue(torch.equal(context[:5], torch.arange(5)))
            self.assertTrue(torch.equal(context[-5:], torch.arange(115, 120)))

    def test_invalid_query_and_width_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_index"):
            centered_gap_indices(120, 4, 30)
        with self.assertRaisesRegex(ValueError, "gap_width"):
            centered_gap_indices(120, 60, 111)
        with self.assertRaises(TypeError):
            centered_gap_indices(120, 60.0, 30)  # type: ignore[arg-type]


class PredictionHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = CNP(x_dim=1, y_dim=1, hidden_dim=16, representation_dim=8)
        self.standardizer = HeartRateStandardizer(mean=80.0, std=10.0)
        time = torch.linspace(-1.0, 1.0, 21).unsqueeze(-1)
        heart_rate = (0.25 + 0.5 * time).contiguous()
        window = BIDMCWindow(
            record_id="bidmc_test",
            patient_id="patient_test",
            window_index=0,
            time=time,
            heart_rate=heart_rate,
        )
        self.entry = WindowEntry(
            key="test/bidmc_test/0",
            split="test",
            label="synthetic",
            window=window,
        )

    def test_linear_interpolation_is_exact_for_a_line(self) -> None:
        target_x = self.entry.window.time
        keep = torch.tensor([0, 1, 2, 17, 18, 19, 20])
        result = linear_interpolate(
            target_x[keep], self.entry.window.heart_rate[keep], target_x
        )
        torch.testing.assert_close(result, self.entry.window.heart_rate)

    def test_prediction_is_finite_and_does_not_use_gap_truth(self) -> None:
        first = predict_window(
            self.model,
            self.entry,
            self.standardizer,
            query_index=10,
            gap_width=5,
        )
        changed_y = self.entry.window.heart_rate.clone()
        changed_y[first.gap_start : first.gap_stop] += 1000.0
        changed_window = BIDMCWindow(
            record_id=self.entry.window.record_id,
            patient_id=self.entry.window.patient_id,
            window_index=0,
            time=self.entry.window.time,
            heart_rate=changed_y,
        )
        changed_entry = WindowEntry(
            key=self.entry.key,
            split=self.entry.split,
            label=self.entry.label,
            window=changed_window,
        )
        second = predict_window(
            self.model,
            changed_entry,
            self.standardizer,
            query_index=10,
            gap_width=5,
        )

        self.assertEqual(first.cnp_mean_bpm.shape, (21,))
        self.assertTrue(torch.isfinite(first.cnp_mean_bpm).all())
        self.assertTrue(torch.all(first.cnp_scale_bpm > 0))
        torch.testing.assert_close(first.cnp_mean_bpm, second.cnp_mean_bpm)
        torch.testing.assert_close(first.cnp_scale_bpm, second.cnp_scale_bpm)
        torch.testing.assert_close(first.linear_bpm, second.linear_bpm)
        self.assertNotEqual(first.truth_query_bpm, second.truth_query_bpm)


class RealRuntimeSmokeTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "bidmc"
    raw_data_ready = (
        (data_dir / "raw" / "bidmc_01_Numerics.csv").is_file()
        and (data_dir / "raw" / "bidmc_01_Fix.txt").is_file()
    )
    checkpoint = project_root / "outputs" / "bidmc" / "best_model.pt"

    @unittest.skipUnless(
        raw_data_ready and checkpoint.is_file(),
        "downloaded BIDMC data and trained checkpoint are optional",
    )
    def test_saved_checkpoint_catalog_and_prediction(self) -> None:
        runtime = BIDMCDemoRuntime.load(self.data_dir, self.checkpoint)
        self.assertEqual(
            {split: len(runtime.entries_for_split(split)) for split in ("train", "validation", "test")},
            {"train": 144, "validation": 40, "test": 27},
        )
        first_test = runtime.entries_for_split("test")[0]
        result = runtime.predict(first_test.key, query_index=60, gap_width=30)
        self.assertEqual(result.context_indices.numel(), 90)
        self.assertEqual(result.cnp_mean_bpm.shape, (120,))
        self.assertTrue(torch.isfinite(result.cnp_mean_bpm).all())
        self.assertGreater(result.cnp_query_scale_bpm, 0.0)

    @unittest.skipUnless(
        raw_data_ready and checkpoint.is_file(),
        "downloaded BIDMC data and trained checkpoint are optional",
    )
    def test_runtime_rejects_patient_overlap_even_when_records_are_disjoint(self) -> None:
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        modified = copy.deepcopy(checkpoint)
        modified["split"]["train"]["record_ids"].remove("bidmc_20")
        modified["split"]["test"]["record_ids"].append("bidmc_20")
        modified["split"]["test"]["record_ids"].sort()
        modified["split"]["test"]["patient_ids"].append("s11342")
        modified["split"]["test"]["patient_ids"].sort()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlapping_patients.pt"
            torch.save(modified, path)
            with self.assertRaisesRegex(ValueError, "patients occur in both train and test"):
                BIDMCDemoRuntime.load(self.data_dir, path)


if __name__ == "__main__":
    unittest.main()
