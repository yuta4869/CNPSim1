from __future__ import annotations

import unittest

import torch

from examples.train_rbf_gp import final_test_gate, tensor_fingerprint


def _condition(cnp_rmse: float, context_mean_rmse: float) -> dict[str, object]:
    return {
        "cnp": {"function_macro_rmse": cnp_rmse},
        "context_mean": {"function_macro_rmse": context_mean_rmse},
    }


class RBFTrainingDriverTests(unittest.TestCase):
    def test_final_test_gate_uses_predeclared_relative_criteria(self) -> None:
        evaluation = {
            "3": _condition(1.0, 1.1),
            "5": _condition(0.85, 1.0),
            "10": _condition(0.60, 0.8),
            "20": _condition(0.79, 0.9),
        }

        gate = final_test_gate(evaluation)

        self.assertIs(gate["criteria_declared_before_final_evaluation"], True)
        self.assertIs(gate["passed"], True)
        self.assertTrue(all(gate["checks"].values()))
        self.assertAlmostEqual(
            gate["observed"]["c3_to_c20_relative_improvement"], 0.21
        )

    def test_final_test_gate_reports_each_failed_condition(self) -> None:
        evaluation = {
            "3": _condition(1.0, 1.0),
            "5": _condition(1.0, 1.0),
            "10": _condition(0.80, 1.0),
            "20": _condition(0.81, 1.0),
        }

        gate = final_test_gate(evaluation)

        self.assertIs(gate["passed"], False)
        self.assertFalse(
            gate["checks"]["c20_rmse_improves_at_least_20_percent_over_c3"]
        )
        self.assertFalse(
            gate["checks"][
                "c10_rmse_beats_context_mean_by_at_least_25_percent"
            ]
        )

    def test_tensor_fingerprint_is_stable_and_value_sensitive_without_numpy(
        self,
    ) -> None:
        value = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

        first = tensor_fingerprint(value)
        second = tensor_fingerprint(value.clone())
        changed = tensor_fingerprint(value + 0.25)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
