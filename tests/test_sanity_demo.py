from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import torch

from cnp import CNP
from cnp.sanity_demo import SUITE_KEYS, SanityDemoRuntime


def write_checkpoint(path: Path, *, benchmark: str = "canonical_sanity") -> None:
    config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": 16,
        "representation_dim": 8,
    }
    model = CNP(**config)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    functions = {
        "single_sin": ["sin"],
        "single_cos": ["cos"],
        "sin_cos_pair": ["sin", "cos"],
    }
    torch.save(
        {
            "format_version": 1,
            "benchmark": benchmark,
            "model_config": config,
            "x_range": [-math.pi, math.pi],
            "models": {
                key: {
                    "step": 1,
                    "function_names": functions[key],
                    "model_state_dict": state,
                }
                for key in SUITE_KEYS
            },
        },
        path,
    )


class SanityDemoRuntimeTests(unittest.TestCase):
    def test_loads_three_suites_and_predicts_without_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_models.pt"
            write_checkpoint(checkpoint)
            runtime = SanityDemoRuntime.load(checkpoint)

            self.assertEqual(tuple(suite.key for suite in runtime.suites), SUITE_KEYS)
            self.assertEqual(runtime.x_range, (-math.pi, math.pi))
            result = runtime.predict(
                "sin_cos_pair",
                "cos",
                context_count=5,
                query_x=0.5,
                context_seed=23,
                grid_points=33,
            )
            self.assertEqual(result.context_x.shape, (5,))
            self.assertEqual(result.target_x.shape, (32,))
            self.assertTrue(torch.isfinite(result.cnp_mean).all())
            self.assertTrue(torch.all(result.cnp_scale > 0))
            self.assertAlmostEqual(
                result.truth_query_y,
                math.cos(result.query_x),
                places=6,
            )
            self.assertTrue(
                set(result.context_x.tolist()).isdisjoint(result.target_x.tolist())
            )

    def test_function_must_belong_to_selected_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_models.pt"
            write_checkpoint(checkpoint)
            runtime = SanityDemoRuntime.load(checkpoint)
            with self.assertRaisesRegex(ValueError, "not available"):
                runtime.predict(
                    "single_sin",
                    "cos",
                    context_count=5,
                    query_x=0.5,
                    context_seed=1,
                )

    def test_missing_and_wrong_benchmark_checkpoints_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                SanityDemoRuntime.load(root / "missing.pt")

            checkpoint = root / "wrong.pt"
            write_checkpoint(checkpoint, benchmark="sincos")
            with self.assertRaisesRegex(ValueError, "not a canonical_sanity"):
                SanityDemoRuntime.load(checkpoint)

    def test_query_validation_and_exact_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_models.pt"
            write_checkpoint(checkpoint)
            runtime = SanityDemoRuntime.load(checkpoint)
            with self.assertRaisesRegex(ValueError, "inside"):
                runtime.predict(
                    "single_sin",
                    "sin",
                    context_count=3,
                    query_x=math.pi + 0.1,
                    context_seed=1,
                )
            result = runtime.predict(
                "single_sin",
                "sin",
                context_count=3,
                query_x=0.503,
                context_seed=1,
                grid_points=33,
            )
            self.assertEqual(result.query_x, 0.503)
            self.assertAlmostEqual(result.truth_query_y, math.sin(0.503), places=6)
            self.assertNotIn(result.query_x, result.context_x.tolist())


if __name__ == "__main__":
    unittest.main()
