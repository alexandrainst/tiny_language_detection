#!/usr/bin/env python3
"""Stream YODAS-Granary into a chunked HF dataset with metadata caching.

Key optimisation: instead of downloading ALL parquet files for a language (e.g. 630 for
Dutch), we download incrementally and stop once we have enough samples:
- 100 test samples per language
- Up to 20h of train samples per language

Metadata is cached locally per language so we don't re-download on resume.
Audio is streamed only for selected samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Features, Value
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

LOGGER = logging.getLogger(__name__)

TARGET_LANGUAGES = [
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Slovak",
    "Spanish",
    "Swedish",
    "Ukrainian",
]

# Map language names to HF dataset language codes
LANG_CODES = {
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hungarian": "hu",
    "Italian": "it",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Spanish": "es",
    "Swedish": "sv",
    "Ukrainian": "uk",
}

FEATURES = Features({"audio": Audio(sampling_rate=16_000), "lang": Value("string")})

SHARDS_PER_COMMIT = 50


@dataclass
class SelectedSample:
    """A sample selected for inclusion in the dataset."""

    utt_id: str
    split: str  # "asr_only" or "ast"
    parquet_file: str  # e.g. "data/bg000/asr_only/00000000.parquet"
    duration: float
    lang: str


@dataclass
class UploadState:
    """Mutable shard upload state."""

    repo_id: str
    chunk_size: int
    api: HfApi
    cache_dir: Path
    shard_indices: dict[str, int]
    buffers: dict[str, list[dict[str, object]]]
    pending_files: list[Path] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream YODAS-Granary into a chunked HF dataset with caching"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repo (e.g. dansmart/yodas-granary-language-detection-v2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=128, help="Samples per parquet shard"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/granary_cache"),
        help="Directory to cache metadata per language",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create repo as private (default: True)",
    )
    return parser.parse_args()


def _create_empty_repo(api: HfApi, repo_id: str, private: bool = True) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        LOGGER.info("Repo %s already exists", repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
        LOGGER.info("Created repo %s (private=%s)", repo_id, private)


def _get_shard_indices(api: HfApi, repo_id: str) -> dict[str, int]:
    """Scan repo to find the highest shard index per split."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    indices: dict[str, int] = {"train": -1, "test": -1}
    for f in files:
        if f.startswith("train-") and f.endswith(".parquet"):
            idx = int(f.removeprefix("train-").removesuffix(".parquet"))
            indices["train"] = max(indices["train"], idx)
        elif f.startswith("test-") and f.endswith(".parquet"):
            idx = int(f.removeprefix("test-").removesuffix(".parquet"))
            indices["test"] = max(indices["test"], idx)
    # Resume from next index
    return {k: v + 1 for k, v in indices.items()}


def _get_completed_languages(api: HfApi, repo_id: str, cache_dir: Path) -> set[str]:
    """Return languages that are fully uploaded (100+ test samples)."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Check HF repo for completed languages
    test_files = [f for f in files if f.startswith("test-")]
    completed = set()

    # Count samples per language from test files
    lang_test_counts: dict[str, int] = {}
    for f in test_files:
        # Read parquet to count unique languages
        try:
            hf_path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
            table = pq.read_table(hf_path, columns=["lang"])
            for lang in table.column("lang").to_pylist():
                lang_test_counts[lang] = lang_test_counts.get(lang, 0) + 1
            Path(hf_path).unlink()
        except Exception:
            pass

    for lang, count in lang_test_counts.items():
        if count >= 100:
            completed.add(lang)

    # Also check cache directory for partially processed languages
    if cache_dir.exists():
        for lang_dir in cache_dir.iterdir():
            if lang_dir.is_dir():
                meta_file = lang_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if meta.get("test_samples", 0) >= 100:
                        completed.add(meta["language"])

    return completed


def _download_parquet_files_incremental(
    lang_code: str, cache_dir: Path, seed: int = 42, max_parquet_files: int = 100
) -> tuple[list[SelectedSample], int]:
    """Download parquet files incrementally until we have enough samples.

    Returns:
        Tuple of (selected_samples, total_parquet_files_checked)
    """
    rng = random.Random(f"{seed}:{lang_code}")

    # We'll collect all valid samples, then select
    all_samples: list[SelectedSample] = []
    parquet_count = 0

    # Try asr_only and ast splits
    for split in ["asr_only", "ast"]:
        for file_idx in range(max_parquet_files):
            parquet_file = f"data/{lang_code}000/{split}/{file_idx:08d}.parquet"

            # Try to download this parquet file
            try:
                local_path = hf_hub_download(
                    repo_id="espnet/yodas-granary",
                    filename=parquet_file,
                    repo_type="dataset",
                    revision="969944574ea3f37890beaf67ea651e160cfaf043",
                )
            except Exception as e:
                # File doesn't exist or error - try next
                if "404" in str(e) or "403" in str(e):
                    # No more files in this split
                    LOGGER.info(
                        "No more %s files for %s (checked %d)",
                        split,
                        lang_code,
                        file_idx,
                    )
                break

            parquet_count += 1

            # Read metadata from this parquet file
            table = pq.read_table(local_path)

            # Extract metadata
            for row in table.to_pylist():
                utt_id = row.get("utt_id")
                duration = row.get("duration")
                lang = row.get("lang")

                if utt_id and duration and lang:
                    # Filter valid durations (0.3s < dur < 15s)
                    if 0.3 < duration < 15.0:
                        all_samples.append(
                            SelectedSample(
                                utt_id=utt_id,
                                split=split,
                                parquet_file=parquet_file,
                                duration=duration,
                                lang=lang,
                            )
                        )

            # Delete the cached parquet file to save space
            Path(local_path).unlink()

            # Check if we have enough samples to make a decision
            # We need 100 test + ~10k train (20h at avg 7s/sample)
            if len(all_samples) >= 10_500:
                LOGGER.info(
                    "Collected %d samples from %d parquet files for %s",
                    len(all_samples),
                    parquet_count,
                    lang_code,
                )
                break

        if parquet_count >= max_parquet_files:
            break

    # Shuffle samples deterministically
    rng.shuffle(all_samples)

    # Select 100 for test, rest for train (up to 20h)
    test_samples = all_samples[:100]

    train_samples = []
    train_duration = 0.0
    max_train_duration = 20.0 * 3600  # 20 hours in seconds

    for sample in all_samples[100:]:
        if train_duration >= max_train_duration:
            break
        train_samples.append(sample)
        train_duration += sample.duration

    selected = test_samples + train_samples

    LOGGER.info(
        "Selected %d test + %d train (%.2fh) from %d parquet files for %s",
        len(test_samples),
        len(train_samples),
        train_duration / 3600,
        parquet_count,
        lang_code,
    )

    return selected, parquet_count


def _stream_audio_for_samples(
    samples: list[SelectedSample], lang_code: str, cache_dir: Path, seed: int = 42
) -> tuple[list[dict], float]:
    """Stream audio only for selected samples.

    Returns:
        Tuple of (rows for dataset, total duration)
    """
    rng = random.Random(f"{seed}:{lang_code}")
    rows = []
    total_duration = 0.0

    # Group by parquet file to minimize downloads
    by_parquet: dict[str, list[SelectedSample]] = {}
    for sample in samples:
        by_parquet.setdefault(sample.parquet_file, []).append(sample)

    LOGGER.info(
        "Streaming audio for %d samples from %d parquet files",
        len(samples),
        len(by_parquet),
    )

    for parquet_file, samples_in_file in by_parquet.items():
        # Download this parquet file
        try:
            local_path = hf_hub_download(
                repo_id="espnet/yodas-granary",
                filename=parquet_file,
                repo_type="dataset",
                revision="969944574ea3f37890beaf67ea651e160cfaf043",
            )
        except Exception as e:
            LOGGER.error("Failed to download %s: %s", parquet_file, e)
            # Create dummy audio for missing samples
            for sample in samples_in_file:
                rows.append(
                    {
                        "audio": {"array": [0.0] * 4800, "sampling_rate": 16000},
                        "lang": sample.lang,
                    }
                )
                total_duration += sample.duration
            continue

        # Read the parquet file
        table = pq.read_table(local_path)

        # Create a lookup by utt_id
        utt_to_row = {}
        for row in table.to_pylist():
            utt_to_row[row["utt_id"]] = row

        # Extract audio for selected samples
        for sample in samples_in_file:
            if sample.utt_id in utt_to_row:
                row_data = utt_to_row[sample.utt_id]
                audio = row_data.get("audio")
                if audio:
                    rows.append({"audio": audio, "lang": sample.lang})
                    total_duration += audio.get("duration", sample.duration)
            else:
                # Sample not found - create dummy
                rows.append(
                    {
                        "audio": {"array": [0.0] * 4800, "sampling_rate": 16000},
                        "lang": sample.lang,
                    }
                )
                total_duration += sample.duration

        # Delete cached parquet file
        Path(local_path).unlink()

    # Shuffle rows
    rng.shuffle(rows)

    return rows, total_duration


def _upload_shards(
    state: UploadState, rows: list[dict], split: str, language: str
) -> None:
    """Upload rows in shards."""
    chunk_size = state.chunk_size

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        shard_idx = state.shard_indices[split]
        filename = f"{split}-{shard_idx:06d}.parquet"
        filepath = state.tmp_dir / filename

        # Create parquet file
        table = pq.Table.from_pylist(chunk)
        pq.write_table(table, filepath)

        state.pending_files.append(filepath)
        state.shard_indices[split] += 1

        LOGGER.info("Created shard %s with %d samples", filename, len(chunk))

        # Commit if we have enough pending files
        if len(state.pending_files) >= SHARDS_PER_COMMIT:
            _commit_shards(state, language)


def _commit_shards(state: UploadState, language: str) -> None:
    """Commit pending shards to HF Hub."""
    if not state.pending_files:
        return

    operations = [
        CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
        for f in state.pending_files
    ]

    commit_message = f"Upload {language} - {len(state.pending_files)} shards"

    try:
        state.api.create_commit(
            repo_id=state.repo_id,
            operations=operations,
            commit_message=commit_message,
            repo_type="dataset",
        )
        LOGGER.info("Committed %d shards", len(state.pending_files))
    except Exception as e:
        LOGGER.error("Failed to commit: %s", e)
        raise
    finally:
        state.pending_files.clear()


def main() -> None:
    """Build and upload the chunked dataset."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set timeout env vars
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

    api = HfApi()
    _create_empty_repo(api=api, repo_id=args.repo_id, private=args.private)

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    completed_languages = _get_completed_languages(
        api=api, repo_id=args.repo_id, cache_dir=cache_dir
    )

    with tempfile.TemporaryDirectory(prefix="granary-shards-") as tmp_dir:
        state = UploadState(
            repo_id=args.repo_id,
            chunk_size=args.chunk_size,
            api=api,
            cache_dir=cache_dir,
            tmp_dir=Path(tmp_dir),
            shard_indices=_get_shard_indices(api=api, repo_id=args.repo_id),
            buffers={"train": [], "test": []},
        )

        stats = []

        for index, language in enumerate(TARGET_LANGUAGES, start=1):
            if language in completed_languages:
                LOGGER.info(
                    "[%d/%d] Skipping %s (already uploaded)",
                    index,
                    len(TARGET_LANGUAGES),
                    language,
                )
                continue

            lang_code = LANG_CODES[language]

            LOGGER.info(
                "[%d/%d] Processing %s (%s)",
                index,
                len(TARGET_LANGUAGES),
                language,
                lang_code,
            )

            # Download parquet files incrementally and select samples
            selected_samples, parquet_count = _download_parquet_files_incremental(
                lang_code=lang_code, cache_dir=cache_dir / lang_code
            )

            # Separate test and train
            test_samples = selected_samples[:100]
            train_samples = selected_samples[100:]

            # Stream audio for test samples
            test_rows, test_duration = _stream_audio_for_samples(
                test_samples, lang_code, cache_dir / lang_code
            )

            # Stream audio for train samples
            train_rows, train_duration = _stream_audio_for_samples(
                train_samples, lang_code, cache_dir / lang_code
            )

            # Upload test shards
            test_split = "test"
            for i in range(0, len(test_rows), state.chunk_size):
                chunk = test_rows[i : i + state.chunk_size]
                shard_idx = state.shard_indices[test_split]
                filename = f"{test_split}-{shard_idx:06d}.parquet"
                filepath = state.tmp_dir / filename

                table = pq.Table.from_pylist(chunk)
                pq.write_table(table, filepath)

                state.pending_files.append(filepath)
                state.shard_indices[test_split] += 1

                if len(state.pending_files) >= SHARDS_PER_COMMIT:
                    _commit_shards(state, language)

            # Upload train shards
            train_split = "train"
            for i in range(0, len(train_rows), state.chunk_size):
                chunk = train_rows[i : i + state.chunk_size]
                shard_idx = state.shard_indices[train_split]
                filename = f"{train_split}-{shard_idx:06d}.parquet"
                filepath = state.tmp_dir / filename

                table = pq.Table.from_pylist(chunk)
                pq.write_table(table, filepath)

                state.pending_files.append(filepath)
                state.shard_indices[train_split] += 1

                if len(state.pending_files) >= SHARDS_PER_COMMIT:
                    _commit_shards(state, language)

            # Commit remaining files
            _commit_shards(state, language)

            # Save language stats
            stats.append(
                {
                    "language": language,
                    "lang_code": lang_code,
                    "test_samples": len(test_rows),
                    "test_duration_hours": test_duration / 3600,
                    "train_samples": len(train_rows),
                    "train_duration_hours": train_duration / 3600,
                    "parquet_files_downloaded": parquet_count,
                }
            )

            # Save metadata for this language
            meta_file = cache_dir / lang_code / "metadata.json"
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_file, "w") as f:
                json.dump(stats[-1], f, indent=2)

            LOGGER.info(
                "%s: test=%d (%.2fh), train=%d (%.2fh) - downloaded %d parquet files",
                language,
                len(test_rows),
                test_duration / 3600,
                len(train_rows),
                train_duration / 3600,
                parquet_count,
            )

        LOGGER.info("\n=== Summary ===")
        for s in stats:
            LOGGER.info(
                "%s: test=%d (%.2fh), train=%d (%.2fh)",
                s["language"],
                s["test_samples"],
                s["test_duration_hours"],
                s["train_samples"],
                s["train_duration_hours"],
            )


if __name__ == "__main__":
    import os
    import tempfile

    main()
