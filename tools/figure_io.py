import json
from datetime import UTC, datetime
from pathlib import Path

from .provenance import (
    get_git_commit,
    get_git_is_dirty,
    sha256_file,
)


def save_figure_with_metadata(
    figure,
    output_path: str | Path,
    metadata: dict,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    metadata_path = output_path.with_suffix(".json")

    metadata = {
        **metadata,
        "figure": output_path.name,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_is_dirty(),
        "figure_sha256": sha256_file(output_path),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            sort_keys=True,
        )
