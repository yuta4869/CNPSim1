from __future__ import annotations

import hashlib
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

from cnp import CNP, gaussian_nll
from cnp.datasets import (
    BIDMCRecord,
    BIDMCWindow,
    fit_heart_rate_standardizer,
    load_bidmc_records,
    make_bidmc_windows,
    split_records_by_subject,
)
from examples.train_bidmc import (
    MaskRegime,
    group_windows_by_subject,
    mask_indices,
    sample_training_batch,
)
from scripts.download_bidmc import parse_manifest, requested_files, sha256_file


def _record(
    record_id: str,
    patient_id: str,
    heart_rate: list[float],
    *,
    start_time: float = 0.0,
) -> BIDMCRecord:
    count = len(heart_rate)
    return BIDMCRecord(
        record_id=record_id,
        patient_id=patient_id,
        time=(torch.arange(count, dtype=torch.float32) + start_time).unsqueeze(-1),
        heart_rate=torch.tensor(heart_rate, dtype=torch.float32).unsqueeze(-1),
        metadata={"fixture": "true"},
    )


def _write_record(
    raw: Path,
    record_id: str,
    patient_id: str,
    rows: str,
) -> None:
    (raw / f"{record_id}_Numerics.csv").write_text(
        "Time [s], HR, PULSE\n" + rows,
        encoding="utf-8",
    )
    (raw / f"{record_id}_Fix.txt").write_text(
        f"{record_id}\n"
        f"MIMIC II matched wdb ID: {patient_id}\n"
        "Comment: fixture: value containing a colon\n",
        encoding="utf-8",
    )


class BIDMCParsingTests(unittest.TestCase):
    def test_loads_csv_and_fix_metadata_in_record_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            _write_record(raw, "bidmc_02", "patient:two", "0,80,79\n1,81,80\n")
            _write_record(raw, "bidmc_01", "patient-one", "0,72,71\n1,,72\n2,75,74\n")

            records = load_bidmc_records(root)

        self.assertEqual([record.record_id for record in records], ["bidmc_01", "bidmc_02"])
        first = records[0]
        self.assertEqual(first.patient_id, "patient-one")
        self.assertEqual(first.time.shape, (3, 1))
        self.assertEqual(first.heart_rate.shape, (3, 1))
        self.assertEqual(first.time.dtype, torch.float32)
        self.assertTrue(torch.equal(first.time[:, 0], torch.tensor([0.0, 1.0, 2.0])))
        self.assertEqual(first.heart_rate[0, 0].item(), 72.0)
        self.assertTrue(math.isnan(first.heart_rate[1, 0].item()))
        self.assertEqual(first.metadata["Comment"], "fixture: value containing a colon")
        self.assertEqual(records[1].patient_id, "patient:two")


class BIDMCSplitAndPreprocessingTests(unittest.TestCase):
    def test_subject_split_is_disjoint_and_deterministic(self) -> None:
        records = [
            _record("bidmc_01", "p1", [60.0, 61.0]),
            _record("bidmc_02", "p2", [62.0, 63.0]),
            _record("bidmc_03", "p3", [64.0, 65.0]),
            _record("bidmc_04", "p4", [66.0, 67.0]),
            _record("bidmc_05", "p5", [68.0, 69.0]),
            _record("bidmc_06", "p6", [70.0, 71.0]),
            _record("bidmc_07", "p2", [72.0, 73.0]),
        ]
        kwargs = {
            "seed": 19,
            "train_subjects": 3,
            "validation_subjects": 1,
            "test_subjects": 2,
        }

        first = split_records_by_subject(records, **kwargs)
        second = split_records_by_subject(tuple(reversed(records)), **kwargs)

        def signature(split: object) -> tuple[tuple[str, ...], ...]:
            return tuple(
                tuple(record.record_id for record in getattr(split, name))
                for name in ("train", "validation", "test")
            )

        self.assertEqual(signature(first), signature(second))
        train_ids = set(first.train_patient_ids)
        validation_ids = set(first.validation_patient_ids)
        test_ids = set(first.test_patient_ids)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual(train_ids | validation_ids | test_ids, {f"p{i}" for i in range(1, 7)})

        buckets: dict[str, set[str]] = {}
        for name in ("train", "validation", "test"):
            for record in getattr(first, name):
                buckets.setdefault(record.patient_id, set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in buckets.values()))

    def test_standardizer_uses_only_the_records_passed_as_training_data(self) -> None:
        records = (
            _record("bidmc_01", "low", [10.0, 11.0, 12.0, 13.0, 14.0]),
            _record("bidmc_02", "middle", [100.0, 102.0, 104.0, 106.0, 108.0]),
            _record("bidmc_03", "high", [1000.0, 1003.0, 1006.0, 1009.0, 1012.0]),
        )
        split = split_records_by_subject(
            records,
            seed=5,
            train_subjects=1,
            validation_subjects=1,
            test_subjects=1,
        )

        standardizer = fit_heart_rate_standardizer(split.train, max_points=4)
        expected = torch.cat(
            [record.heart_rate[:4].reshape(-1).to(torch.float64) for record in split.train]
        )
        all_values = torch.cat(
            [record.heart_rate[:4].reshape(-1).to(torch.float64) for record in records]
        )

        self.assertAlmostEqual(standardizer.mean, expected.mean().item())
        self.assertAlmostEqual(standardizer.std, expected.std(correction=0).item())
        self.assertNotAlmostEqual(standardizer.mean, all_values.mean().item())

    def test_windowing_normalizes_locally_and_skips_a_nan_window(self) -> None:
        training = _record("bidmc_01", "train", [60.0, 62.0, 64.0, 66.0])
        standardizer = fit_heart_rate_standardizer((training,), max_points=4)
        observed = _record(
            "bidmc_02",
            "held-out",
            [60.0, 62.0, 64.0, 66.0, 70.0, math.nan, 74.0, 76.0],
            start_time=100.0,
        )

        windows = make_bidmc_windows(
            (observed,),
            standardizer,
            window_size=4,
            max_points=8,
        )

        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window.record_id, "bidmc_02")
        self.assertEqual(window.patient_id, "held-out")
        self.assertEqual(window.window_index, 0)
        self.assertTrue(
            torch.allclose(
                window.time[:, 0],
                torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]),
            )
        )
        expected_hr = standardizer.normalize(observed.heart_rate[:4])
        self.assertTrue(torch.allclose(window.heart_rate, expected_hr))
        self.assertTrue(torch.isfinite(window.heart_rate).all())


class BIDMCMaskingAndTrainingTests(unittest.TestCase):
    def test_random_and_block_masks_are_deterministic_interpolation_masks(self) -> None:
        random_regime = MaskRegime("fixture_random", "random", 0.30)
        first = mask_indices(20, random_regime, torch.Generator().manual_seed(23))
        second = mask_indices(20, random_regime, torch.Generator().manual_seed(23))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

        context_indices, target_indices = first
        self.assertEqual(target_indices.numel(), 6)
        self.assertNotIn(0, target_indices.tolist())
        self.assertNotIn(19, target_indices.tolist())
        self.assertIn(0, context_indices.tolist())
        self.assertIn(19, context_indices.tolist())
        self.assertEqual(
            sorted(context_indices.tolist() + target_indices.tolist()),
            list(range(20)),
        )
        self.assertTrue(torch.all(target_indices[1:] > target_indices[:-1]))

        block_regime = MaskRegime("fixture_block", "block", 5)
        block_context, block_target = mask_indices(
            20, block_regime, torch.Generator().manual_seed(29)
        )
        self.assertEqual(block_target.numel(), 5)
        self.assertTrue(torch.equal(block_target, torch.arange(block_target[0], block_target[0] + 5)))
        self.assertGreaterEqual(block_target[0].item(), 5)
        self.assertLessEqual(block_target[-1].item(), 14)
        self.assertEqual(
            sorted(block_context.tolist() + block_target.tolist()),
            list(range(20)),
        )

    def test_sampled_batch_completes_one_finite_model_step(self) -> None:
        torch.manual_seed(31)
        windows = tuple(
            BIDMCWindow(
                record_id=f"bidmc_{index + 1:02d}",
                patient_id=f"patient-{index + 1}",
                window_index=0,
                time=torch.linspace(-1.0, 1.0, 20).unsqueeze(-1),
                heart_rate=(
                    torch.sin(torch.linspace(-1.0, 1.0, 20) * math.pi)
                    + index * 0.1
                ).unsqueeze(-1),
            )
            for index in range(3)
        )
        batch = sample_training_batch(
            group_windows_by_subject(windows),
            MaskRegime("fixture_random", "random", 0.30),
            batch_size=4,
            python_rng=random.Random(37),
            torch_generator=torch.Generator().manual_seed(41),
            device=torch.device("cpu"),
        )
        context_x, context_y, target_x, target_y = batch
        self.assertEqual(context_x.shape, (4, 14, 1))
        self.assertEqual(context_y.shape, (4, 14, 1))
        self.assertEqual(target_x.shape, (4, 6, 1))
        self.assertEqual(target_y.shape, (4, 6, 1))
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in batch))

        model = CNP(x_dim=1, y_dim=1, hidden_dim=16, representation_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        mean, scale = model(context_x, context_y, target_x)
        loss = gaussian_nll(mean, scale, target_y)
        optimizer.zero_grad()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(torch.all(scale > 0))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))


class BIDMCDownloaderHelperTests(unittest.TestCase):
    def test_manifest_and_checksum_helpers_without_network(self) -> None:
        payload = b"offline BIDMC fixture\n"
        digest = hashlib.sha256(payload).hexdigest()
        second_digest = hashlib.sha256(b"second fixture\n").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "payload.bin"
            payload_path.write_bytes(payload)
            manifest_path = root / "SHA256SUMS.txt"
            manifest_path.write_text(
                f"{digest}  bidmc_csv/bidmc_01_Numerics.csv\n"
                f"{second_digest} *bidmc_csv/bidmc_01_Fix.txt\n",
                encoding="utf-8",
            )

            checksums = parse_manifest(manifest_path)
            self.assertEqual(sha256_file(payload_path), digest)

        self.assertEqual(
            checksums,
            {
                "bidmc_csv/bidmc_01_Numerics.csv": digest,
                "bidmc_csv/bidmc_01_Fix.txt": second_digest,
            },
        )
        files = requested_files()
        self.assertEqual(len(files), 108)
        self.assertIn(
            ("bidmc_csv/bidmc_53_Numerics.csv", Path("raw/bidmc_53_Numerics.csv")),
            files,
        )


if __name__ == "__main__":
    unittest.main()
