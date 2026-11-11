from __future__ import annotations

import unittest

import torch

from cnp import CNP, gaussian_nll
from cnp.rbf_gp import (
    RBFKernelConfig,
    evaluate_rbf_kernel,
    gp_posterior,
    make_gp_cholesky,
    make_gp_grid,
    make_rbf_evaluation_task,
    rbf_reference_predictions,
    rbf_regression_metrics,
    sample_gp_functions,
    sample_rbf_training_batch,
)


class RBFKernelAndSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = RBFKernelConfig()
        self.grid = make_gp_grid(33)

    def test_kernel_has_expected_diagonal_symmetry_and_decay(self) -> None:
        covariance = evaluate_rbf_kernel(self.grid, self.grid, self.kernel)

        self.assertEqual(covariance.shape, (33, 33))
        self.assertTrue(torch.allclose(covariance, covariance.T))
        self.assertTrue(
            torch.allclose(
                covariance.diag(),
                torch.full((33,), self.kernel.signal_scale**2),
            )
        )
        self.assertLess(covariance[0, -1].item(), covariance[0, 1].item())

    def test_config_validation_and_serialization(self) -> None:
        self.assertEqual(self.kernel.as_dict()["length_scale"], 0.4)
        self.assertEqual(self.kernel.as_dict()["observation_noise"], 0.02)
        with self.assertRaisesRegex(ValueError, "length_scale"):
            RBFKernelConfig(length_scale=0.0)
        with self.assertRaisesRegex(ValueError, "observation_noise"):
            RBFKernelConfig(observation_noise=-1.0)
        with self.assertRaisesRegex(ValueError, "jitter"):
            RBFKernelConfig(jitter=0.0)

    def test_seeded_samples_are_repeatable_and_accept_precomputed_factor(self) -> None:
        factor = make_gp_cholesky(self.grid, self.kernel)
        reconstructed = factor @ factor.T
        latent = evaluate_rbf_kernel(self.grid, self.grid, self.kernel)
        diagonal_addition = (reconstructed - latent).diag()
        expected_addition = self.kernel.observation_noise**2 + self.kernel.jitter
        self.assertTrue(
            torch.allclose(
                diagonal_addition,
                torch.full_like(diagonal_addition, expected_addition),
                atol=2e-6,
            )
        )
        first = sample_gp_functions(
            5,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(17),
            cholesky=factor,
        )
        second = sample_gp_functions(
            5,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(17),
        )
        different = sample_gp_functions(
            5,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(18),
            cholesky=factor,
        )

        self.assertEqual(first.shape, (5, 33, 1))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_appending_exact_query_preserves_the_display_curve_for_same_seed(self) -> None:
        query = torch.tensor([[0.12345]], dtype=self.grid.dtype)
        extended_grid = torch.cat((self.grid, query), dim=0)
        base = sample_gp_functions(
            2,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(19),
        )
        extended = sample_gp_functions(
            2,
            extended_grid,
            self.kernel,
            generator=torch.Generator().manual_seed(19),
        )

        self.assertTrue(torch.allclose(base, extended[:, : self.grid.shape[0]], atol=1e-6))


class RBFTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = RBFKernelConfig()
        self.grid = make_gp_grid(65)
        self.factor = make_gp_cholesky(self.grid, self.kernel)

    def test_training_batch_is_fresh_masked_disjoint_and_trainable(self) -> None:
        generator = torch.Generator().manual_seed(23)
        batch = sample_rbf_training_batch(
            grid=self.grid,
            kernel=self.kernel,
            batch_size=6,
            min_context=3,
            max_context=8,
            target_count=9,
            generator=generator,
            device=torch.device("cpu"),
            cholesky=self.factor,
        )

        self.assertEqual(batch.context_x.shape, (6, 8, 1))
        self.assertEqual(batch.context_y.shape, (6, 8, 1))
        self.assertEqual(batch.context_mask.shape, (6, 8))
        self.assertEqual(batch.target_x.shape, (6, 9, 1))
        self.assertEqual(batch.function_values.shape, (6, 65, 1))
        self.assertTrue(torch.equal(batch.context_mask.sum(dim=1), batch.context_counts))
        for row in range(6):
            self.assertTrue(
                set(batch.context_indices[row].tolist()).isdisjoint(
                    batch.target_indices[row].tolist()
                )
            )

        model = CNP(x_dim=1, y_dim=1, hidden_dim=16, representation_dim=8)
        mean, scale = model(
            batch.context_x,
            batch.context_y,
            batch.target_x,
            batch.context_mask,
        )
        loss = gaussian_nll(mean, scale, batch.target_y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

        later = sample_rbf_training_batch(
            grid=self.grid,
            kernel=self.kernel,
            batch_size=6,
            min_context=3,
            max_context=8,
            target_count=9,
            generator=generator,
            device=torch.device("cpu"),
            cholesky=self.factor,
        )
        self.assertFalse(torch.equal(batch.function_values, later.function_values))

    def test_evaluation_contexts_are_nested_and_queries_are_fixed(self) -> None:
        functions = sample_gp_functions(
            4,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(29),
            cholesky=self.factor,
        )
        small = make_rbf_evaluation_task(
            functions,
            self.grid,
            context_count=3,
            seed=31,
        )
        large = make_rbf_evaluation_task(
            functions,
            self.grid,
            context_count=10,
            seed=31,
        )

        self.assertTrue(torch.equal(small.target_indices, large.target_indices))
        self.assertTrue(torch.equal(small.target_x, large.target_x))
        self.assertTrue(torch.equal(small.target_y, large.target_y))
        for row in range(functions.shape[0]):
            small_context = set(small.context_indices[row].tolist())
            large_context = set(large.context_indices[row].tolist())
            targets = set(small.target_indices[row].tolist())
            self.assertTrue(small_context.issubset(large_context))
            self.assertTrue(small_context.isdisjoint(targets))


class RBFPosteriorAndMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = RBFKernelConfig(observation_noise=0.02)
        self.grid = make_gp_grid(65)
        self.functions = sample_gp_functions(
            3,
            self.grid,
            self.kernel,
            generator=torch.Generator().manual_seed(37),
        )
        self.task = make_rbf_evaluation_task(
            self.functions,
            self.grid,
            context_count=12,
            seed=41,
        )

    def test_posterior_batched_and_unbatched_contracts_match(self) -> None:
        batched_mean, batched_scale = gp_posterior(
            self.task.context_x,
            self.task.context_y,
            self.task.target_x,
            self.kernel,
        )
        mean, scale = gp_posterior(
            self.task.context_x[0],
            self.task.context_y[0],
            self.task.target_x[0],
            self.kernel,
        )

        self.assertEqual(batched_mean.shape, self.task.target_y.shape)
        self.assertEqual(batched_scale.shape, self.task.target_y.shape)
        self.assertTrue(torch.all(batched_scale > 0))
        self.assertTrue(torch.all(batched_scale >= self.kernel.observation_noise))
        self.assertTrue(torch.allclose(mean, batched_mean[0], atol=1e-6))
        self.assertTrue(torch.allclose(scale, batched_scale[0], atol=1e-6))

    def test_posterior_recovers_observed_context_and_reduces_uncertainty(self) -> None:
        observed_mean, observed_scale = gp_posterior(
            self.task.context_x,
            self.task.context_y,
            self.task.context_x,
            self.kernel,
        )
        prior_scale = self.kernel.signal_scale

        self.assertLess(
            torch.max((observed_mean - self.task.context_y).abs()).item(),
            5e-2,
        )
        self.assertLess(observed_scale.max().item(), 0.05)
        self.assertLess(observed_scale.mean().item(), prior_scale)

    def test_reference_predictions_and_metric_schema(self) -> None:
        references = rbf_reference_predictions(self.task, self.kernel)
        metrics = rbf_regression_metrics(
            references.gp_mean,
            torch.ones_like(references.gp_scale),
            self.task,
            self.kernel,
        )

        self.assertEqual(references.context_mean.shape, self.task.target_y.shape)
        self.assertEqual(set(metrics), {"cnp", "gp_posterior", "context_mean"})
        self.assertIn("function_macro_rmse", metrics["cnp"])
        self.assertIn("coverage_95", metrics["gp_posterior"])
        self.assertIn("point_micro_mae", metrics["context_mean"])


if __name__ == "__main__":
    unittest.main()
