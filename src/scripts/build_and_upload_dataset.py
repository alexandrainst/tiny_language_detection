#!/usr/bin/env python3
"""Build and upload balanced YODAS-Granary language detection dataset.

Combines dataset building and upload into single pipeline:
1. Stream through source dataset to collect metadata
2. Split into train/test (100 samples test per language, rest train)
3. Stream audio for selected samples
4. Upload to Hugging Face Hub as parquet shards

Supports incremental upload with metadata caching to avoid re-download on resume.
"""

from __future__ import annotations

import argparse
import logging
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
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
TEST_SAMPLES_PER_LANG = 100
MAX_TRAIN_HOURS_PER_LANG = 20


@dataclass
class Sample:
    """Metadata for a single audio sample."""

    utt_id: str
    lang: str
    duration: float
    audio_file: str | None = field(default=None)


def collect_metadata(lang_name: str, seed: int = 42) -> list[Sample]:
    """Stream through dataset splits to collect metadata.

    Args:
        lang_name:
            Full language name (e.g., "Danish").
        seed:
            Random seed for shuffling.

    Returns:
        List of Sample metadata objects.
    """
    lang_code = LANG_CODES.get(lang_name, lang_name[:2].lower())
    LOGGER.info(f"Collecting metadata for {lang_name} ({lang_code})...")

    ds = load_dataset("espnet/yodas-granary", lang_code, streaming=True)
    all_samples = []

    for split in ds.keys():
        for sample in ds[split]:
            dur = float(sample.get("duration", 0))
            if 0.3 < dur < 15:
                all_samples.append(
                    Sample(utt_id=sample.get("utt_id"), lang=lang_name, duration=dur)
                )

    random.seed(seed)
    random.shuffle(all_samples)
    LOGGER.info(f"  Found {len(all_samples)} valid samples")
    return all_samples


def split_samples(
    samples: list[Sample], test_count: int = TEST_SAMPLES_PER_LANG
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into train and test sets.

    Args:
        samples:
            List of all samples.
        test_count:
            Number of samples for test set.

    Returns:
        Tuple of (test_samples, train_samples).
    """
    test = samples[:test_count]
    train = samples[test_count:]

    # Cap training at MAX_TRAIN_HOURS
    train_duration = sum(s.duration for s in train) / 3600
    if train_duration > MAX_TRAIN_HOURS_PER_LANG:
        target_duration = MAX_TRAIN_HOURS_PER_LANG * 3600
        capped = []
        current = 0
        for s in train:
            if current + s.duration <= target_duration:
                capped.append(s)
                current += s.duration
        train = capped
        LOGGER.info(
            f"  Capped training at {MAX_TRAIN_HOURS_PER_LANG}h ({len(train)} samples)"
        )

    LOGGER.info(f"  Test: {len(test)}, Train: {len(train)}")
    return test, train


def download_parquet_incremental(
    lang_code: str, cache_dir: Path, seed: int = 42, max_files: int = 100
) -> list[Sample]:
    """Download parquet files incrementally until enough samples collected.

    Args:
        lang_code:
            Language code (e.g., "da").
        cache_dir:
            Directory to cache metadata.
        seed:
            Random seed.
        max_files:
            Maximum parquet files to download.

    Returns:
        List of collected samples.
    """
    all_samples = []
    files_downloaded = 0

    for split in ["asr_only", "ast"]:
        for i in range(0, max_files, 2):
            if files_downloaded >= max_files:
                break

            base = f"{lang_code}{i:03d}"
            for suffix in ["_train", "_test"]:
                filename = f"data/{base}{suffix}.parquet"
                try:
                    local_path = hf_hub_download(
                        repo_id="espnet/yodas-granary",
                        filename=filename,
                        repo_type="dataset",
                        cache_dir=str(cache_dir),
                    )
                    table = pq.read_table(local_path)
                    for row in table.to_pylist():
                        dur = float(row.get("duration", 0))
                        if 0.3 < dur < 15:
                            all_samples.append(
                                Sample(
                                    utt_id=row["utt_id"],
                                    lang=LANG_CODES.get(lang_code, lang_code),
                                    duration=dur,
                                )
                            )
                    files_downloaded += 1
                except Exception:
                    continue

    random.seed(seed)
    random.shuffle(all_samples)
    return all_samples


def stream_audio_and_upload(
    test_samples: list[Sample],
    train_samples: list[Sample],
    lang_name: str,
    repo_id: str,
    split_name: str,
) -> list[Path]:
    """Stream audio for samples and upload as parquet shards.

    Args:
        test_samples:
            Test set samples.
        train_samples:
            Training set samples.
        lang_name:
            Full language name.
        repo_id:
            Hugging Face repo ID.
        split_name:
            Split name ("train" or "test").

    Returns:
        List of uploaded shard paths.
    """
    samples = test_samples if split_name == "test" else train_samples
    lang_code = LANG_CODES.get(lang_name, lang_name[:2].lower())

    LOGGER.info(f"Streaming {len(samples)} {split_name} samples for {lang_name}...")

    uploaded_shards = []
    shard_data = []

    for i, sample in enumerate(samples):
        if sample.utt_id is None:
            continue

        try:
            ds = load_dataset("espnet/yodas-granary", lang_code, streaming=True)
            found = False

            for split in ds.keys():
                for s in ds[split]:
                    if s.get("utt_id") == sample.utt_id:
                        shard_data.append({"audio": s["audio"], "lang": lang_name})
                        found = True
                        break
                if found:
                    break

            if not found:
                LOGGER.warning(f"  Not found: {sample.utt_id}")

        except Exception as e:
            LOGGER.warning(f"  Error streaming {sample.utt_id}: {e}")

        # Upload every 128 samples
        if len(shard_data) >= 128:
            shard_path = upload_shard(
                shard_data, repo_id, split_name, lang_name, len(uploaded_shards)
            )
            uploaded_shards.append(shard_path)
            shard_data = []

    # Upload remaining
    if shard_data:
        shard_path = upload_shard(
            shard_data, repo_id, split_name, lang_name, len(uploaded_shards)
        )
        uploaded_shards.append(shard_path)

    return uploaded_shards


def upload_shard(
    data: list[dict], repo_id: str, split: str, lang: str, shard_idx: int
) -> Path:
    """Upload a single parquet shard.

    Args:
        data:
            List of sample dicts.
        repo_id:
            Hugging Face repo ID.
        split:
            Split name.
        lang:
            Language name.
        shard_idx:
            Shard index.

    Returns:
        Local path to uploaded shard.
    """
    table = pa.Table.from_pylist(data)
    shard_dir = Path(tempfile.mkdtemp())
    shard_path = shard_dir / f"{lang.lower()}_{split}_{shard_idx:03d}.parquet"
    pq.write_table(table, shard_path)

    api = HfApi()
    path_in_repo = f"data/{shard_path.name}"

    try:
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        LOGGER.info(f"  Uploaded {shard_path.name}")
    except Exception as e:
        LOGGER.error(f"  Upload failed: {e}")

    return shard_path


def build_and_upload(
    repo_id: str = "saattrupdan/yodas-granary-language-detection",
    skip_upload: bool = False,
    test_only: bool = False,
    seed: int = 42,
) -> None:
    """Main build and upload pipeline.

    Args:
        repo_id:
            Target Hugging Face repo ID.
        skip_upload:
            If True, only build metadata locally.
        test_only:
            If True, only build test set.
        seed:
            Random seed.
    """
    # Create repo if needed
    if not skip_upload:
        api = HfApi()
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            LOGGER.info(f"Repo exists: {repo_id}")
        except RepositoryNotFoundError:
            LOGGER.info(f"Creating repo: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)

    for i, lang in enumerate(TARGET_LANGUAGES):
        LOGGER.info(f"[{i + 1}/{len(TARGET_LANGUAGES)}] {lang}")

        # Collect and split
        samples = collect_metadata(lang, seed)
        test, train = split_samples(samples, TEST_SAMPLES_PER_LANG)

        if test_only:
            continue

        # Upload
        if not skip_upload:
            stream_audio_and_upload(test, train, lang, repo_id, "test")
            stream_audio_and_upload(train, train, lang, repo_id, "train")

    LOGGER.info("✓ Done!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build and upload YODAS-Granary dataset"
    )
    parser.add_argument(
        "--repo-id",
        default="saattrupdan/yodas-granary-language-detection",
        help="Hugging Face repo ID",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Build metadata only, don't upload"
    )
    parser.add_argument("--test-only", action="store_true", help="Build test set only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_and_upload(
        repo_id=args.repo_id,
        skip_upload=args.skip_upload,
        test_only=args.test_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
