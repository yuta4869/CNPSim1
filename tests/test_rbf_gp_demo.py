from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import Tensor, nn

from cnp import CNP
from cnp.rbf_gp import RBFKernelConfig
from cnp.rbf_gp_demo import RBFGPDemoRuntime


def write_checkpoint(
    path: Path,
    *,
    benchmark: str = "rbf_gp",
    kernel: dict[str, float] | None = None,
    max_context: int = 10,
) -> None:
    model_config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": 16,
        "representation_dim": 8,
    }
    model = CNP(**model_config)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format_version": 1,
            "benchmark": benchmark,
            "step": 250,
            "model_config": model_config,
            "model_state_dict": state,
            "x_range": [-2.0, 2.0],
            "grid_points": 129,
            "kernel": kernel
            or RBFKernelConfig(
                length_scale=0.4,
                signal_scale=1.0,
                observation_noise=0.02,
                jitter=1e-6,
            ).as_dict(),
            "training_config": {"min_context": 3, "max_context": max_context},
            "gate_passed": True,
            "test_summary": {"10": {"cnp": {"function_macro_rmse": 0.5}}},
        },
        path,
    )


class RecordingModel(nn.Module):
    """A three-input model that records the runtime's complete model contract."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[Tensor, Tensor, Tensor] | None = None

    def forward(
        self, context_x: Tensor, context_y: Tensor, target_x: Tensor
    ) -> tuple[Tensor, Tensor]:
        self.inputs = (
            context_x.detach().cpu().clone(),
            context_y.detach().cpu().clone(),
            target_x.detach().cpu().clone(),
        )
        shape = (*target_x.shape[:-1], 1)
        return target_x.new_zeros(shape), target_x.new_ones(shape)


class RBFGPDemoRuntimeTests(unittest.TestCase):
    def test_loads_checkpoint_and_returns_cnp_and_gp_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint)
            runtime = RBFGPDemoRuntime.load(checkpoint)

            self.assertEqual(runtime.step, 250)
            self.assertEqual(runtime.x_range, (-2.0, 2.0))
            self.assertEqual(runtime.grid_points, 129)
            self.assertEqual(runtime.training_max_context, 10)
            self.assertTrue(runtime.gate_passed)
            self.assertAlmostEqual(runtime.kernel.observation_noise, 0.02)

            result = runtime.predict(
                7001,
                context_count=7,
                context_seed=91,
                query_x=0.317,
                display_grid_points=65,
            )
            self.assertEqual(result.context_x.shape, (7,))
            self.assertEqual(result.target_x.shape, (32,))
            self.assertEqual(result.truth_y.shape, result.cnp_mean.shape)
            self.assertEqual(result.truth_y.shape, result.gp_mean.shape)
            self.assertTrue(torch.isfinite(result.cnp_mean).all())
            self.assertTrue(torch.isfinite(result.gp_mean).all())
            self.assertTrue(torch.all(result.cnp_scale > 0))
            self.assertTrue(torch.all(result.gp_scale > 0))
            self.assertEqual(result.query_x, 0.317)
            self.assertGreaterEqual(result.cnp_rmse, 0.0)
            self.assertGreaterEqual(result.gp_rmse, 0.0)
            self.assertFalse(result.context_is_ood)

    def test_moving_exact_query_keeps_display_curve_and_context_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint)
            runtime = RBFGPDemoRuntime.load(checkpoint)

            left = runtime.predict(
                42,
                context_count=8,
                context_seed=17,
                query_x=-0.731,
                display_grid_points=65,
            )
            right = runtime.predict(
                42,
                context_count=8,
                context_seed=17,
                query_x=0.619,
                display_grid_points=65,
            )

            self.assertTrue(torch.equal(left.target_x, right.target_x))
            self.assertTrue(torch.allclose(left.truth_y, right.truth_y, atol=1e-6, rtol=0.0))
            self.assertTrue(torch.equal(left.context_x, right.context_x))
            self.assertTrue(torch.allclose(left.context_y, right.context_y, atol=1e-6, rtol=0.0))
            self.assertEqual(left.query_x, -0.731)
            self.assertEqual(right.query_x, 0.619)

    def test_function_and_context_seeds_have_separate_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint)
            runtime = RBFGPDemoRuntime.load(checkpoint)

            first = runtime.predict(
                11,
                context_count=6,
                context_seed=9,
                query_x=0.2,
                display_grid_points=65,
            )
            new_function = runtime.predict(
                12,
                context_count=6,
                context_seed=9,
                query_x=0.2,
                display_grid_points=65,
            )
            new_context = runtime.predict(
                11,
                context_count=6,
                context_seed=10,
                query_x=0.2,
                display_grid_points=65,
            )

            self.assertFalse(torch.allclose(first.truth_y, new_function.truth_y))
            self.assertFalse(torch.equal(first.context_x, new_context.context_x))
            self.assertTrue(torch.allclose(first.truth_y, new_context.truth_y))

    def test_model_receives_only_context_pairs_and_query_x(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint)
            runtime = RBFGPDemoRuntime.load(checkpoint)
            recorder = RecordingModel()
            runtime.model = recorder  # type: ignore[assignment]

            result = runtime.predict(
                8,
                context_count=5,
                context_seed=3,
                query_x=0.123,
                display_grid_points=33,
            )
            self.assertIsNotNone(recorder.inputs)
            context_x, context_y, target_x = recorder.inputs or ()
            self.assertEqual(context_x.shape, (1, 5, 1))
            self.assertEqual(context_y.shape, (1, 5, 1))
            self.assertEqual(target_x.shape, (1, 17, 1))
            self.assertAlmostEqual(float(target_x[0, -1, 0]), 0.123, places=6)
            self.assertEqual(result.truth_y.shape, (16,))

    def test_context_above_training_maximum_is_labelled_ood(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint, max_context=5)
            runtime = RBFGPDemoRuntime.load(checkpoint)
            result = runtime.predict(
                2,
                context_count=6,
                context_seed=1,
                query_x=0.0,
                display_grid_points=33,
            )
            self.assertTrue(result.context_is_ood)
            self.assertTrue(result.context_out_of_training_range)
            self.assertEqual(result.training_max_context, 5)

    def test_missing_wrong_and_malformed_checkpoints_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                RBFGPDemoRuntime.load(root / "missing.pt")

            wrong = root / "wrong.pt"
            write_checkpoint(wrong, benchmark="canonical_sanity")
            with self.assertRaisesRegex(ValueError, "not an rbf_gp"):
                RBFGPDemoRuntime.load(wrong)

            malformed = root / "malformed.pt"
            write_checkpoint(malformed)
            payload = torch.load(malformed, map_location="cpu", weights_only=True)
            payload["kernel"].pop("observation_noise")
            torch.save(payload, malformed)
            with self.assertRaisesRegex(ValueError, "kernel field mismatch"):
                RBFGPDemoRuntime.load(malformed)

    def test_predict_validates_query_counts_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            write_checkpoint(checkpoint)
            runtime = RBFGPDemoRuntime.load(checkpoint)
            with self.assertRaisesRegex(ValueError, "inside"):
                runtime.predict(
                    1,
                    context_count=3,
                    context_seed=2,
                    query_x=2.1,
                )
            with self.assertRaisesRegex(ValueError, "Context candidates"):
                runtime.predict(
                    1,
                    context_count=18,
                    context_seed=2,
                    query_x=0.0,
                    display_grid_points=33,
                )
            with self.assertRaisesRegex(ValueError, "function_seed"):
                runtime.predict(
                    -1,
                    context_count=3,
                    context_seed=2,
                    query_x=0.0,
                )


if __name__ == "__main__":
    unittest.main()
