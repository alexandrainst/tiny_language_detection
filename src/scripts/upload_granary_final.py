#!/usr/bin/env python3
"""Upload train/test parquet files to HF using hf CLI."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_id = "saattrupdan/yodas-granary-language-detection"
    data_dir = Path("data/granary_final")

    if not data_dir.exists():
        print(f"Error: {data_dir} doesn't exist")
        sys.exit(1)

    train_files = list(data_dir.glob("train-*.parquet"))
    test_files = list(data_dir.glob("test-*.parquet"))

    print(
        f"Uploading {len(train_files)} train shards and {len(test_files)} test shards..."
    )

    result = subprocess.run(
        ["uv", "run", "hf", "upload", repo_id, str(data_dir / "*.parquet"), "."],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Upload failed:\n{result.stderr}")
        sys.exit(1)

    print(f"Upload completed:\n{result.stdout}")


if __name__ == "__main__":
    main()
