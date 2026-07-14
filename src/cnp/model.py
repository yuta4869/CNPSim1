"""Deterministic Conditional Neural Process with a Gaussian output head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Independent, Normal


class ConditionalNeuralProcess(nn.Module):
    """A deterministic Conditional Neural Process (CNP).

    The model encodes each ``(context_x, context_y)`` pair independently,
    averages the resulting representations, and decodes every ``target_x``
    into the mean and scale of a diagonal Gaussian.

    Inputs may be unbatched or batched:

    - ``context_x``: ``[C, x_dim]`` or ``[B, C, x_dim]``
    - ``context_y``: ``[C, y_dim]`` or ``[B, C, y_dim]``
    - ``target_x``: ``[T, x_dim]`` or ``[B, T, x_dim]``
    - ``context_mask``: optional ``[C]`` or ``[B, C]`` boolean mask

    At least one valid context point is required for every batch item. This
    matches the upstream PyTorch implementation and avoids assigning an
    unspecified meaning to an empty-context prediction.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 128,
        representation_dim: int = 128,
        min_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if x_dim < 1 or y_dim < 1:
            raise ValueError("x_dim and y_dim must both be positive")
        if hidden_dim < 1 or representation_dim < 1:
            raise ValueError("hidden_dim and representation_dim must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.representation_dim = representation_dim
        self.min_scale = min_scale

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * y_dim),
        )

    @staticmethod
    def _with_batch(tensor: Tensor, name: str) -> tuple[Tensor, bool]:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0), True
        if tensor.ndim == 3:
            return tensor, False
        raise ValueError(f"{name} must have rank 2 or 3, got shape {tuple(tensor.shape)}")

    def _prepare_context(
        self,
        context_x: Tensor,
        context_y: Tensor,
        context_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, bool]:
        context_x, x_was_unbatched = self._with_batch(context_x, "context_x")
        context_y, y_was_unbatched = self._with_batch(context_y, "context_y")
        if x_was_unbatched != y_was_unbatched:
            raise ValueError("context_x and context_y must use the same batching convention")
        if context_x.shape[:-1] != context_y.shape[:-1]:
            raise ValueError("context_x and context_y must have matching batch and point axes")
        if context_x.shape[-1] != self.x_dim:
            raise ValueError(
                f"context_x last dimension must be {self.x_dim}, got {context_x.shape[-1]}"
            )
        if context_y.shape[-1] != self.y_dim:
            raise ValueError(
                f"context_y last dimension must be {self.y_dim}, got {context_y.shape[-1]}"
            )
        if context_x.shape[-2] == 0:
            raise ValueError("at least one context point is required")
        if context_x.device != context_y.device:
            raise ValueError("context_x and context_y must be on the same device")
        if context_x.dtype != context_y.dtype:
            raise ValueError("context_x and context_y must have the same dtype")

        prepared_mask: Tensor | None = None
        if context_mask is not None:
            expected_shape = context_x.shape[:-1]
            if x_was_unbatched and context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if tuple(context_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    "context_mask must match the context batch and point axes; "
                    f"expected {tuple(expected_shape)}, got {tuple(context_mask.shape)}"
                )
            prepared_mask = context_mask.to(device=context_x.device, dtype=torch.bool)
            valid_counts = prepared_mask.sum(dim=-1)
            if torch.any(valid_counts == 0):
                raise ValueError("every batch item must contain at least one valid context point")

        return context_x, context_y, prepared_mask, x_was_unbatched

    def encode_context(
        self,
        context_x: Tensor,
        context_y: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode and permutation-invariantly aggregate a context set."""

        context_x, context_y, context_mask, was_unbatched = self._prepare_context(
            context_x, context_y, context_mask
        )
        if context_mask is not None:
            # Sanitize padding before the encoder. Masking only the encoded
            # values would let NaN/Inf padding contaminate Linear gradients.
            valid = context_mask.unsqueeze(-1)
            context_x = torch.where(valid, context_x, torch.zeros_like(context_x))
            context_y = torch.where(valid, context_y, torch.zeros_like(context_y))
        encoded = self.encoder(torch.cat((context_x, context_y), dim=-1))

        if context_mask is None:
            representation = encoded.mean(dim=-2)
        else:
            encoded = torch.where(valid, encoded, torch.zeros_like(encoded))
            counts = valid.sum(dim=-2).to(encoded.dtype)
            representation = encoded.sum(dim=-2) / counts

        return representation.squeeze(0) if was_unbatched else representation

    def decode(self, representation: Tensor, target_x: Tensor) -> tuple[Tensor, Tensor]:
        """Decode target inputs from an already aggregated representation."""

        target_x, target_was_unbatched = self._with_batch(target_x, "target_x")
        if representation.ndim == 1:
            representation = representation.unsqueeze(0)
            representation_was_unbatched = True
        elif representation.ndim == 2:
            representation_was_unbatched = False
        else:
            raise ValueError(
                "representation must have rank 1 or 2, "
                f"got shape {tuple(representation.shape)}"
            )

        if target_was_unbatched != representation_was_unbatched:
            raise ValueError("representation and target_x must use the same batching convention")
        if representation.shape[0] != target_x.shape[0]:
            raise ValueError("representation and target_x batch sizes must match")
        if representation.shape[-1] != self.representation_dim:
            raise ValueError(
                "representation last dimension must be "
                f"{self.representation_dim}, got {representation.shape[-1]}"
            )
        if target_x.shape[-1] != self.x_dim:
            raise ValueError(
                f"target_x last dimension must be {self.x_dim}, got {target_x.shape[-1]}"
            )
        if representation.device != target_x.device:
            raise ValueError("representation and target_x must be on the same device")
        if representation.dtype != target_x.dtype:
            raise ValueError("representation and target_x must have the same dtype")

        target_count = target_x.shape[-2]
        repeated = representation.unsqueeze(-2).expand(-1, target_count, -1)
        raw_output = self.decoder(torch.cat((repeated, target_x), dim=-1))
        mean, raw_scale = raw_output.split(self.y_dim, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale

        if target_was_unbatched:
            return mean.squeeze(0), scale.squeeze(0)
        return mean, scale

    def forward(
        self,
        context_x: Tensor,
        context_y: Tensor,
        target_x: Tensor,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return Gaussian ``(mean, scale)`` at each target input."""

        representation = self.encode_context(context_x, context_y, context_mask)
        return self.decode(representation, target_x)

    def distribution(
        self,
        context_x: Tensor,
        context_y: Tensor,
        target_x: Tensor,
        context_mask: Tensor | None = None,
    ) -> Independent:
        """Return the factorized Gaussian predicted at the target inputs."""

        mean, scale = self(context_x, context_y, target_x, context_mask)
        return Independent(Normal(mean, scale), 1)


CNP = ConditionalNeuralProcess
