"""Train the standalone CNP on a family of sine functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cnp import CNP, gaussian_nll, sample_context_target


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def function_batch(
    batch_size: int,
    point_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a batch of related but distinct sine functions."""

    x = torch.linspace(-2.0, 2.0, point_count, device=device)
    x = x.view(1, point_count, 1).expand(batch_size, -1, -1)
    amplitude = 0.5 + 1.5 * torch.rand(batch_size, 1, 1, device=device)
    frequency = 0.8 + 1.2 * torch.rand(batch_size, 1, 1, device=device)
    phase = 2.0 * torch.pi * torch.rand(batch_size, 1, 1, device=device)
    y = amplitude * torch.sin(frequency * x + phase)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--min-context", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/toy_cnp.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.min_context <= args.max_context:
        raise ValueError("expected 1 <= min-context <= max-context")
    if args.max_context + args.targets > args.points:
        raise ValueError("max-context + targets cannot exceed points")

    torch.manual_seed(args.seed)
    device = default_device() if args.device == "auto" else torch.device(args.device)
    model = CNP(x_dim=1, y_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for step in range(1, args.steps + 1):
        x, y = function_batch(args.batch_size, args.points, device)
        num_context = int(
            torch.randint(args.min_context, args.max_context + 1, size=()).item()
        )
        batch = sample_context_target(x, y, num_context, args.targets)
        mean, scale = model(batch.context_x, batch.context_y, batch.target_x)
        loss = gaussian_nll(mean, scale, batch.target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step={step:04d} context={num_context:02d} nll={loss.item():.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"x_dim": 1, "y_dim": 1, "hidden_dim": 128},
        },
        args.output,
    )
    print(f"saved={args.output} device={device}")


if __name__ == "__main__":
    main()
