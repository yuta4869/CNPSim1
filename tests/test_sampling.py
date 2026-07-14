from __future__ import annotations

import unittest

import torch

from cnp import sample_context_target


class SamplingTests(unittest.TestCase):
    def test_shapes_and_disjoint_points(self) -> None:
        x = torch.arange(20, dtype=torch.float32).view(2, 10, 1)
        y = 2.0 * x + 1.0
        batch = sample_context_target(x, y, num_context=3, num_target=4)

        self.assertEqual(batch.context_x.shape, (2, 3, 1))
        self.assertEqual(batch.target_x.shape, (2, 4, 1))
        self.assertTrue(torch.equal(batch.context_y, 2.0 * batch.context_x + 1.0))
        self.assertTrue(torch.equal(batch.target_y, 2.0 * batch.target_x + 1.0))
        for batch_id in range(2):
            context_values = set(batch.context_x[batch_id, :, 0].tolist())
            target_values = set(batch.target_x[batch_id, :, 0].tolist())
            self.assertTrue(context_values.isdisjoint(target_values))

    def test_seeded_sampling_is_reproducible(self) -> None:
        x = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
        y = x.square()
        generator_a = torch.Generator().manual_seed(19)
        generator_b = torch.Generator().manual_seed(19)
        batch_a = sample_context_target(x, y, 4, 5, generator=generator_a)
        batch_b = sample_context_target(x, y, 4, 5, generator=generator_b)
        self.assertTrue(torch.equal(batch_a.context_x, batch_b.context_x))
        self.assertTrue(torch.equal(batch_a.target_x, batch_b.target_x))

    def test_invalid_counts_are_rejected(self) -> None:
        x = torch.randn(2, 5, 1)
        y = torch.randn(2, 5, 1)
        with self.assertRaises(ValueError):
            sample_context_target(x, y, 0, 1)
        with self.assertRaises(ValueError):
            sample_context_target(x, y, 3, 3)


if __name__ == "__main__":
    unittest.main()
