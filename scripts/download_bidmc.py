#!/usr/bin/env python3
"""Download the small BIDMC subset used by the heart-rate experiments.

Only the 1 Hz numerics, fixed metadata, and dataset documentation are needed.
Every payload is checked against PhysioNet's pinned v1.0.0 SHA256 manifest
before an atomic rename makes it visible at its final path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VERSION = "1.0.0"
BASE_URL = f"https://physionet.org/files/bidmc/{VERSION}"
MANIFEST_NAME = "SHA256SUMS.txt"
# Pinning the manifest closes the otherwise circular gap where a downloaded
# checksum file would be trusted to authenticate itself.
MANIFEST_SHA256 = "e28756681318a18d9c682a68cdff65efcaccb3ece40cfccd380540b77a3fdd6d"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_atomic(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    timeout: float,
) -> None:
    """Stream *url* to ``destination.part``, verify it, then rename it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    part.unlink(missing_ok=True)
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "standalone-cnp-bidmc/0.1"})

    try:
        with urlopen(request, timeout=timeout) as response, part.open("wb") as output:
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
        os.replace(part, destination)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def parse_manifest(path: Path) -> dict[str, str]:
    """Parse the two-column PhysioNet SHA256SUMS format."""

    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed manifest line {line_number}: {raw_line!r}")
            checksum, filename = parts
            filename = filename.lstrip("*")
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError(f"Invalid SHA256 on manifest line {line_number}")
            if filename in checksums:
                raise ValueError(f"Duplicate manifest entry: {filename}")
            checksums[filename] = checksum
    return checksums


def requested_files() -> tuple[tuple[str, Path], ...]:
    """Return ``(upstream path, local relative path)`` pairs."""

    files: list[tuple[str, Path]] = [
        ("LICENSE", Path("LICENSE")),
        ("README", Path("PHYSIONET_README.txt")),
    ]
    for number in range(1, 54):
        stem = f"bidmc_{number:02d}"
        files.extend(
            (
                (f"bidmc_csv/{stem}_Numerics.csv", Path("raw") / f"{stem}_Numerics.csv"),
                (f"bidmc_csv/{stem}_Fix.txt", Path("raw") / f"{stem}_Fix.txt"),
            )
        )
    return tuple(files)


def ensure_file(
    upstream_path: str,
    destination: Path,
    expected_sha256: str,
    *,
    timeout: float,
    force: bool,
) -> str:
    if not force and destination.is_file() and sha256_file(destination) == expected_sha256:
        return "verified"
    download_atomic(
        f"{BASE_URL}/{upstream_path}",
        destination,
        expected_sha256,
        timeout=timeout,
    )
    return "downloaded"


def download_dataset(output: Path, *, timeout: float = 60.0, force: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME

    if force or not manifest_path.is_file() or sha256_file(manifest_path) != MANIFEST_SHA256:
        download_atomic(
            f"{BASE_URL}/{MANIFEST_NAME}",
            manifest_path,
            MANIFEST_SHA256,
            timeout=timeout,
        )
        print(f"downloaded {manifest_path}")
    else:
        print(f"verified   {manifest_path}")

    checksums = parse_manifest(manifest_path)
    files = requested_files()
    missing_entries = [upstream for upstream, _ in files if upstream not in checksums]
    if missing_entries:
        raise ValueError(
            "Required files are absent from the official manifest: "
            + ", ".join(missing_entries)
        )

    downloaded = 0
    for upstream_path, relative_destination in files:
        destination = output / relative_destination
        status = ensure_file(
            upstream_path,
            destination,
            checksums[upstream_path],
            timeout=timeout,
            force=force,
        )
        downloaded += status == "downloaded"
        print(f"{status:<10} {destination}")

    print(
        f"BIDMC v{VERSION}: verified {len(files)} files "
        f"({downloaded} downloaded, {len(files) - downloaded} already present)."
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "bidmc",
        help="destination directory (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--force", action="store_true", help="download even verified files again")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    try:
        download_dataset(args.output.expanduser().resolve(), timeout=args.timeout, force=args.force)
    except (HTTPError, URLError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
