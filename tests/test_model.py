from __future__ import annotations

import unittest

import torch

from cnp import CNP, gaussian_nll


class ConditionalNeuralProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = CNP(x_dim=2, y_dim=3, hidden_dim=32, representation_dim=16)

    def test_unbatched_and_batched_shapes(self) -> None:
        mean, scale = self.model(
            torch.randn(4, 2),
            torch.randn(4, 3),
            torch.randn(7, 2),
        )
        self.assertEqual(mean.shape, (7, 3))
        self.assertEqual(scale.shape, (7, 3))
        self.assertTrue(torch.all(torch.isfinite(scale)))
        self.assertTrue(torch.all(scale > 0))

        mean, scale = self.model(
            torch.randn(5, 4, 2),
            torch.randn(5, 4, 3),
            torch.randn(5, 7, 2),
        )
        self.assertEqual(mean.shape, (5, 7, 3))
        self.assertEqual(scale.shape, (5, 7, 3))

    def test_context_order_is_invariant(self) -> None:
        context_x = torch.randn(2, 6, 2)
        context_y = torch.randn(2, 6, 3)
        target_x = torch.randn(2, 5, 2)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])

        original = self.model(context_x, context_y, target_x)
        reordered = self.model(
            context_x[:, permutation],
            context_y[:, permutation],
            target_x,
        )
        self.assertTrue(torch.allclose(original[0], reordered[0], atol=1e-6))
        self.assertTrue(torch.allclose(original[1], reordered[1], atol=1e-6))

    def test_target_order_is_equivariant(self) -> None:
        context_x = torch.randn(6, 2)
        context_y = torch.randn(6, 3)
        target_x = torch.randn(5, 2)
        permutation = torch.tensor([3, 1, 4, 0, 2])

        original = self.model(context_x, context_y, target_x)
        reordered = self.model(context_x, context_y, target_x[permutation])
        self.assertTrue(torch.allclose(original[0][permutation], reordered[0], atol=1e-6))
        self.assertTrue(torch.allclose(original[1][permutation], reordered[1], atol=1e-6))

    def test_mask_ignores_padding(self) -> None:
        valid_x = torch.randn(2, 3, 2)
        valid_y = torch.randn(2, 3, 3)
        padding_x_a = torch.randn(2, 2, 2)
        padding_y_a = torch.randn(2, 2, 3)
        padding_x_b = torch.randn(2, 2, 2)
        padding_y_b = torch.randn(2, 2, 3)
        target_x = torch.randn(2, 4, 2)
        mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)

        output_a = self.model(
            torch.cat((valid_x, padding_x_a), dim=1),
            torch.cat((valid_y, padding_y_a), dim=1),
            target_x,
            mask,
        )
        output_b = self.model(
            torch.cat((valid_x, padding_x_b), dim=1),
            torch.cat((valid_y, padding_y_b), dim=1),
            target_x,
            mask,
        )
        self.assertTrue(torch.allclose(output_a[0], output_b[0], atol=1e-6))
        self.assertTrue(torch.allclose(output_a[1], output_b[1], atol=1e-6))

    def test_masked_nan_padding_does_not_contaminate_gradients(self) -> None:
        context_x = torch.randn(2, 5, 2)
        context_y = torch.randn(2, 5, 3)
        context_x[:, 3:] = torch.nan
        context_y[:, 3:] = torch.nan
        mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
        target_x = torch.randn(2, 4, 2)
        target_y = torch.randn(2, 4, 3)

        mean, scale = self.model(context_x, context_y, target_x, mask)
        loss = gaussian_nll(mean, scale, target_y)
        loss.backward()

        self.assertTrue(torch.all(torch.isfinite(mean)))
        self.assertTrue(torch.all(torch.isfinite(scale)))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
                for parameter in self.model.parameters()
            )
        )

    def test_masked_batch_matches_each_unpadded_item(self) -> None:
        context_x = torch.randn(2, 5, 2)
        context_y = torch.randn(2, 5, 3)
        target_x = torch.randn(2, 4, 2)
        mask = torch.tensor(
            [[True, True, False, False, False], [True, True, True, True, False]]
        )

        batched = self.model(context_x, context_y, target_x, mask)
        first = self.model(context_x[0, :2], context_y[0, :2], target_x[0])
        second = self.model(context_x[1, :4], context_y[1, :4], target_x[1])

        self.assertTrue(torch.allclose(batched[0][0], first[0], atol=1e-6))
        self.assertTrue(torch.allclose(batched[1][0], first[1], atol=1e-6))
        self.assertTrue(torch.allclose(batched[0][1], second[0], atol=1e-6))
        self.assertTrue(torch.allclose(batched[1][1], second[1], atol=1e-6))

    def test_nll_backpropagates(self) -> None:
        context_x = torch.randn(3, 5, 2)
        context_y = torch.randn(3, 5, 3)
        target_x = torch.randn(3, 4, 2)
        target_y = torch.randn(3, 4, 3)
        mean, scale = self.model(context_x, context_y, target_x)
        loss = gaussian_nll(mean, scale, target_y)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
                for parameter in self.model.parameters()
            )
        )

    def test_distribution_and_nll_agree(self) -> None:
        context_x = torch.randn(2, 5, 2)
        context_y = torch.randn(2, 5, 3)
        target_x = torch.randn(2, 4, 2)
        target_y = torch.randn(2, 4, 3)
        mean, scale = self.model(context_x, context_y, target_x)

        expected = -self.model.distribution(context_x, context_y, target_x).log_prob(target_y)
        actual = gaussian_nll(mean, scale, target_y, reduction="none")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_empty_or_fully_masked_context_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one context"):
            self.model(torch.empty(0, 2), torch.empty(0, 3), torch.randn(2, 2))

        with self.assertRaisesRegex(ValueError, "at least one valid context"):
            self.model(
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 3),
                torch.randn(2, 4, 2),
                torch.zeros(2, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
