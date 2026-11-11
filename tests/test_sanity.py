from __future__ import annotations

import math
import unittest

import torch

from cnp import CNP, gaussian_nll
from cnp.sanity import (
    evaluate_canonical_functions,
    make_canonical_evaluation_task,
    sample_canonical_training_batch,
)


class CanonicalFunctionTests(unittest.TestCase):
    def test_definitions_are_exact_sin_and_cos(self) -> None:
        x = torch.linspace(-math.pi, math.pi, 65).unsqueeze(-1)
        values = evaluate_canonical_functions(torch.tensor([0, 1]), x)

        self.assertTrue(torch.allclose(values[0], torch.sin(x), atol=1e-6))
        self.assertTrue(torch.allclose(values[1], torch.cos(x), atol=1e-6))

    def test_invalid_function_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            evaluate_canonical_functions(
                torch.tensor([2]), torch.tensor([[0.25]], dtype=torch.float32)
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            evaluate_canonical_functions(
                torch.tensor([0.0]), torch.tensor([[0.25]], dtype=torch.float32)
            )


class CanonicalTaskTests(unittest.TestCase):
    def test_training_batch_masks_variable_contexts_and_completes_step(self) -> None:
        batch = sample_canonical_training_batch(
            torch.tensor([0, 1]),
            batch_size=9,
            min_context=3,
            max_context=8,
            target_count=7,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
        )
        self.assertEqual(batch.context_x.shape, (9, 8, 1))
        self.assertEqual(batch.target_x.shape, (9, 7, 1))
        self.assertTrue(torch.equal(batch.context_mask.sum(1), batch.context_counts))
        self.assertTrue(
            torch.allclose(
                batch.target_y,
                evaluate_canonical_functions(batch.function_indices, batch.target_x),
            )
        )

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
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_evaluation_contexts_are_nested_and_disjoint_from_queries(self) -> None:
        functions = torch.tensor([0, 1])
        small = make_canonical_evaluation_task(
            functions, context_count=3, grid_points=129, seed=17
        )
        large = make_canonical_evaluation_task(
            functions, context_count=10, grid_points=129, seed=17
        )
        for function_index in range(2):
            small_context = set(small.context_indices[function_index].tolist())
            large_context = set(large.context_indices[function_index].tolist())
            queries = set(small.target_indices[function_index].tolist())
            self.assertTrue(small_context.issubset(large_context))
            self.assertTrue(small_context.isdisjoint(queries))
            self.assertEqual(
                queries, set(large.target_indices[function_index].tolist())
            )
        self.assertTrue(torch.all(small.context_x >= -math.pi))
        self.assertTrue(torch.all(small.context_x <= math.pi))
        self.assertTrue(torch.all(small.target_x > -math.pi))
        self.assertTrue(torch.all(small.target_x < math.pi))

    def test_context_truth_belongs_to_selected_function(self) -> None:
        task = make_canonical_evaluation_task(
            torch.tensor([0, 1]), context_count=5, grid_points=33, seed=23
        )
        expected = evaluate_canonical_functions(
            task.function_indices, task.context_x
        )
        self.assertTrue(torch.equal(task.context_y, expected))


if __name__ == "__main__":
    unittest.main()
