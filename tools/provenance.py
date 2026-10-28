from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

GIT_DIRTY_EXCLUDED_PATHS = (
    "examples/figs",
    "benchmarks/figs",
)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_is_dirty() -> bool | None:

    excluded_pathspecs = [f":(exclude){path}" for path in GIT_DIRTY_EXCLUDED_PATHS]

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                *excluded_pathspecs,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()
