from __future__ import annotations

import argparse
import math
import unittest

import torch

from cnp import CNP, gaussian_nll
from cnp.synthetic import (
    SinusoidParameterRange,
    deterministic_regression_metrics,
    evaluate_sinusoids,
    gaussian_regression_metrics,
    interpolation_baselines,
    make_synthetic_evaluation_task,
    sample_sinusoid_parameters,
    sample_synthetic_training_batch,
    sinusoid_regression_baseline,
)
from examples.train_sincos import make_parameter_banks


class SinusoidFunctionTests(unittest.TestCase):
    def test_phase_zero_is_sine_and_phase_pi_over_two_is_cosine(self) -> None:
        x = torch.linspace(-2.0, 2.0, 31).unsqueeze(-1)
        parameters = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, math.pi / 2.0, 0.0]]
        )

        values = evaluate_sinusoids(parameters, x)

        self.assertTrue(torch.allclose(values[0], torch.sin(x), atol=1e-6))
        self.assertTrue(torch.allclose(values[1], torch.cos(x), atol=1e-6))

    def test_parameter_sampling_is_seeded_and_respects_half_open_ranges(self) -> None:
        parameter_range = SinusoidParameterRange(
            amplitude=(0.5, 1.5),
            angular_frequency=(0.8, 2.0),
            phase=(0.0, 2.0 * math.pi),
            offset=(-0.5, 0.5),
        )
        first = sample_sinusoid_parameters(
            128, parameter_range, generator=torch.Generator().manual_seed(17)
        )
        second = sample_sinusoid_parameters(
            128, parameter_range, generator=torch.Generator().manual_seed(17)
        )
        different = sample_sinusoid_parameters(
            128, parameter_range, generator=torch.Generator().manual_seed(18)
        )

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        for column, (lower, upper) in enumerate(
            (
                parameter_range.amplitude,
                parameter_range.angular_frequency,
                parameter_range.phase,
                parameter_range.offset,
            )
        ):
            self.assertTrue(torch.all(first[:, column] >= lower))
            self.assertTrue(torch.all(first[:, column] < upper))

    def test_fixed_banks_are_disjoint_and_include_pure_sin_cos(self) -> None:
        args = argparse.Namespace(
            train_functions=32,
            validation_functions=8,
            test_functions=8,
            anchor_functions=6,
            seed=23,
        )
        banks, manifest = make_parameter_banks(args)

        self.assertEqual(manifest["overlap_check"], "all exact parameter tuples are disjoint")
        self.assertTrue(torch.equal(banks["pure_sin_test"][:, 2], torch.zeros(6)))
        self.assertTrue(
            torch.allclose(
                banks["pure_cos_test"][:, 2],
                torch.full((6,), math.pi / 2.0),
            )
        )
        self.assertTrue(torch.equal(banks["pure_sin_test"][:, 3], torch.zeros(6)))
        self.assertTrue(torch.equal(banks["pure_cos_test"][:, 3], torch.zeros(6)))
        train_rows = {tuple(row) for row in banks["train"].tolist()}
        for name, bank in banks.items():
            if name != "train":
                self.assertTrue(train_rows.isdisjoint({tuple(row) for row in bank.tolist()}))


class SinusoidTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = sample_sinusoid_parameters(
            5,
            SinusoidParameterRange(),
            generator=torch.Generator().manual_seed(29),
        )

    def test_training_batch_uses_variable_context_mask_and_completes_step(self) -> None:
        batch = sample_synthetic_training_batch(
            self.parameters,
            batch_size=7,
            min_context=3,
            max_context=8,
            target_count=6,
            x_range=(-2.0, 2.0),
            generator=torch.Generator().manual_seed(31),
            device=torch.device("cpu"),
        )
        self.assertEqual(batch.context_x.shape, (7, 8, 1))
        self.assertEqual(batch.context_y.shape, (7, 8, 1))
        self.assertEqual(batch.context_mask.shape, (7, 8))
        self.assertEqual(batch.target_x.shape, (7, 6, 1))
        self.assertTrue(torch.equal(batch.context_mask.sum(dim=1), batch.context_counts))
        self.assertTrue(torch.all(batch.context_counts >= 3))
        self.assertTrue(torch.all(batch.context_counts <= 8))
        expected_target = evaluate_sinusoids(
            self.parameters[batch.function_indices], batch.target_x
        )
        self.assertTrue(torch.allclose(batch.target_y, expected_target))

        model = CNP(x_dim=1, y_dim=1, hidden_dim=16, representation_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        mean, scale = model(
            batch.context_x,
            batch.context_y,
            batch.target_x,
            batch.context_mask,
        )
        loss = gaussian_nll(mean, scale, batch.target_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_evaluation_contexts_are_nested_disjoint_and_have_expected_geometry(self) -> None:
        for regime in (
            "interpolation",
            "gap_interpolation",
            "extrapolation_stress",
        ):
            small = make_synthetic_evaluation_task(
                self.parameters,
                regime=regime,
                context_count=3,
                grid_points=129,
                x_range=(-2.0, 2.0),
                seed=37,
            )
            large = make_synthetic_evaluation_task(
                self.parameters,
                regime=regime,
                context_count=10,
                grid_points=129,
                x_range=(-2.0, 2.0),
                seed=37,
            )
            for function_index in range(self.parameters.shape[0]):
                small_context = set(small.context_indices[function_index].tolist())
                large_context = set(large.context_indices[function_index].tolist())
                targets = set(small.target_indices[function_index].tolist())
                self.assertTrue(small_context.issubset(large_context))
                self.assertTrue(small_context.isdisjoint(targets))
                self.assertEqual(
                    targets,
                    set(large.target_indices[function_index].tolist()),
                )

            if regime == "interpolation":
                minimum = small.context_x.min(dim=1).values.unsqueeze(1)
                maximum = small.context_x.max(dim=1).values.unsqueeze(1)
                self.assertTrue(torch.all(small.target_x > minimum))
                self.assertTrue(torch.all(small.target_x < maximum))
            elif regime == "gap_interpolation":
                self.assertTrue(torch.all(small.context_x.abs() >= 0.6))
                self.assertTrue(torch.all(small.target_x.abs() < 0.6))
            else:
                self.assertTrue(torch.all(small.context_x.abs() <= 1.0))
                self.assertTrue(torch.all(small.target_x.abs() > 1.0))

    def test_zero_centered_regimes_reject_incompatible_domains(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-centered gap"):
            make_synthetic_evaluation_task(
                self.parameters,
                regime="gap_interpolation",
                context_count=3,
                grid_points=129,
                x_range=(1.0, 3.0),
                seed=41,
            )
        with self.assertRaisesRegex(ValueError, "zero-centered Context interval"):
            make_synthetic_evaluation_task(
                self.parameters,
                regime="extrapolation_stress",
                context_count=3,
                grid_points=129,
                x_range=(-0.5, 2.0),
                seed=41,
            )
        with self.assertRaisesRegex(ValueError, "two Context candidates"):
            make_synthetic_evaluation_task(
                self.parameters,
                regime="extrapolation_stress",
                context_count=3,
                grid_points=9,
                x_range=(-2.0, 2.0),
                extrapolation_context_half_width=0.1,
                seed=41,
            )


class SinusoidMetricAndBaselineTests(unittest.TestCase):
    def test_metric_helpers_match_a_hand_computed_fixture(self) -> None:
        target = torch.zeros(2, 2, 1)
        prediction = torch.tensor([[[1.0], [-1.0]], [[2.0], [-2.0]]])
        deterministic = deterministic_regression_metrics(prediction, target)
        self.assertAlmostEqual(deterministic["function_macro_mae"], 1.5)
        self.assertAlmostEqual(deterministic["point_micro_rmse"], math.sqrt(2.5))

        gaussian = gaussian_regression_metrics(
            torch.zeros_like(target), torch.ones_like(target), target
        )
        self.assertAlmostEqual(
            gaussian["function_macro_nll"], 0.5 * math.log(2.0 * math.pi), places=6
        )
        self.assertEqual(gaussian["coverage_68"], 1.0)
        self.assertEqual(gaussian["coverage_95"], 1.0)
        self.assertAlmostEqual(gaussian["mean_interval_width_95"], 3.92, places=6)

    def test_linear_and_sincos_baselines_recover_their_function_families(self) -> None:
        context_x = torch.tensor([[[-2.0], [-1.0], [0.0], [1.0], [2.0]]])
        linear_y = 2.0 * context_x + 1.0
        target_x = torch.tensor([[[-1.5], [-0.5], [0.5], [1.5], [2.5]]])
        linear = interpolation_baselines(context_x, linear_y, target_x)["linear"]
        self.assertTrue(torch.allclose(linear, 2.0 * target_x + 1.0, atol=1e-6))

        parameters = torch.tensor([[1.3, 1.4, 0.7, -0.2]])
        sinusoid_context_y = evaluate_sinusoids(parameters, context_x)
        truth = evaluate_sinusoids(parameters, target_x)
        fitted = sinusoid_regression_baseline(
            context_x,
            sinusoid_context_y,
            target_x,
            frequency_min=1.0,
            frequency_max=1.8,
            frequency_steps=81,
        )
        self.assertLess(torch.sqrt(torch.mean((fitted - truth).square())).item(), 2e-3)

        with self.assertRaisesRegex(ValueError, "scalar"):
            sinusoid_regression_baseline(
                context_x.expand(-1, -1, 2),
                linear_y.expand(-1, -1, 2),
                target_x.expand(-1, -1, 2),
            )


if __name__ == "__main__":
    unittest.main()
