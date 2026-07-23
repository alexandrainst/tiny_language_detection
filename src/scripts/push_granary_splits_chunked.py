#!/usr/bin/env python3
"""Stream YODAS-Granary into a chunked Hugging Face dataset.

The generated dataset has two splits, ``train`` and ``test``, with only ``audio`` and
``lang`` columns.

Split strategy:
- test: 100 random valid samples per language
- train: remaining valid samples, capped at 20 hours per language

Audio is written and uploaded in small parquet shards. Multiple shards are batched
into a single commit to stay within the Hugging Face rate limit of 256 commits/hour.
The full dataset is never materialised on local disk.
"""

from __future__ import annotations

import argparse
import collections.abc as c
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

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

FEATURES = Features(
    {
        "audio": Audio(sampling_rate=16_000),
        "lang": Value("string"),
    }
)

# Upload this many shards per commit to stay well under the 256/hour limit.
SHARDS_PER_COMMIT = 50


@dataclass(frozen=True)
class SourceKey:
    """Unique identifier for a streamed source sample."""

    split: str
    utt_id: str


@dataclass(frozen=True)
class SampleMeta:
    """Metadata used for selecting samples without storing audio."""

    key: SourceKey
    duration: float
    lang: str


@dataclass
class LanguageStats:
    """Per-language split statistics."""

    language: str
    valid_samples: int
    train_samples: int
    train_duration_hours: float
    test_samples: int
    test_duration_hours: float


@dataclass
class UploadState:
    """Mutable shard upload state."""

    repo_id: str
    chunk_size: int
    api: HfApi
    tmp_dir: Path
    shard_indices: dict[str, int]
    buffers: dict[str, list[dict[str, object]]]
    pending_files: list[Path] = field(default_factory=list)


def main() -> None:
    """Build and upload the chunked dataset."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    api = HfApi()
    _create_empty_repo(api=api, repo_id=args.repo_id, private=args.private)

    completed_languages = _get_completed_languages(
        api=api, repo_id=args.repo_id
    )

    stats: list[LanguageStats] = []
    with tempfile.TemporaryDirectory(prefix="granary-shards-") as tmp_dir:
        state = UploadState(
            repo_id=args.repo_id,
            chunk_size=args.chunk_size,
            api=api,
            tmp_dir=Path(tmp_dir),
            shard_indices={"train": 0, "test": 0},
            buffers={"train": [], "test": []},
        )

        # Resume shard indices from existing files
        state.shard_indices = _get_shard_indices(
            api=api, repo_id=args.repo_id
        )

        for index, language in enumerate(TARGET_LANGUAGES, start=1):
            if language in completed_languages:
                LOGGER.info(
                    "[%d/%d] Skipping %s (already uploaded)",
                    index,
                    len(TARGET_LANGUAGES),
                    language,
                )
                continue

            LOGGER.info("[%d/%d] Processing %s", index, len(TARGET_LANGUAGES), language)
            language_stats = _process_language(
                language=language,
                state=state,
                test_samples=args.test_samples,
                max_train_seconds=args.max_train_hours * 3600,
                seed=args.seed,
            )
            stats.append(language_stats)
            LOGGER.info(
                "%s: test=%d (%.2fh), train=%d (%.2fh)",
                language,
                language_stats.test_samples,
                language_stats.test_duration_hours,
                language_stats.train_samples,
                language_stats.train_duration_hours,
            )

        _flush_all(state=state)

    readme_path = _write_readme(repo_id=args.repo_id, stats=stats)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    readme_path.unlink()

    LOGGER.info("Done: https://huggingface.co/datasets/%s", args.repo_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="saattrupdan/yodas-granary-language-detection-v2",
        help="Destination dataset repository. Must not already exist.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
        help="Number of test samples per language.",
    )
    parser.add_argument(
        "--max-train-hours",
        type=float,
        default=20.0,
        help="Maximum training duration per language.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of samples per uploaded parquet shard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic split selection.",
    )
    parser.add_argument(
        "--public",
        action="store_false",
        dest="private",
        help="Create the destination dataset as public instead of private.",
    )
    parser.set_defaults(private=True)
    return parser.parse_args()


def _create_empty_repo(api: HfApi, repo_id: str, private: bool) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=False,
        )
        LOGGER.info("Created dataset repo %s", repo_id)
        return

    LOGGER.info("Using existing dataset repo %s", repo_id)


def _get_completed_languages(api: HfApi, repo_id: str) -> set[str]:
    """Check which languages already have complete test+train data.

    Returns:
        Set of language names with >= 100 test samples uploaded.
    """
    completed: set[str] = set()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return completed

    # Check test files - each language should have 100 test samples
    test_files = sorted(f for f in files if f.startswith("data/test-"))
    if not test_files:
        return completed

    # Download test files and check languages
    with tempfile.TemporaryDirectory():
        test_langs: dict[str, int] = {}
        for f in test_files:
            local = api.hf_hub_download(repo_id, f, repo_type="dataset")
            table = pq.read_table(local)
            for lang in table.column("lang").to_pylist():
                test_langs[lang] = test_langs.get(lang, 0) + 1

        # Languages with 100 test samples are complete
        for lang, count in test_langs.items():
            if count >= 100:
                completed.add(_lang_name_from_tag(lang))

    return completed


def _get_shard_indices(api: HfApi, repo_id: str) -> dict[str, int]:
    """Get the next shard index for each split based on existing files.

    Returns:
        Dictionary mapping ``"train"`` and ``"test"`` to the next
        shard index to use.
    """
    indices = {"train": 0, "test": 0}
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return indices

    for f in files:
        if f.startswith("data/train-"):
            idx = int(f.replace("data/train-", "").replace(".parquet", ""))
            indices["train"] = max(indices["train"], idx + 1)
        elif f.startswith("data/test-"):
            idx = int(f.replace("data/test-", "").replace(".parquet", ""))
            indices["test"] = max(indices["test"], idx + 1)

    return indices


def _lang_name_from_tag(tag: str) -> str:
    """Convert language tag like '<bg>' to 'Bulgarian'.

    Returns:
        Human-readable language name, or the original tag if unknown.
    """
    tag_to_name = {
        "<bg>": "Bulgarian",
        "<hr>": "Croatian",
        "<cs>": "Czech",
        "<da>": "Danish",
        "<nl>": "Dutch",
        "<en>": "English",
        "<et>": "Estonian",
        "<fi>": "Finnish",
        "<fr>": "French",
        "<de>": "German",
        "<el>": "Greek",
        "<hu>": "Hungarian",
        "<it>": "Italian",
        "<lv>": "Latvian",
        "<lt>": "Lithuanian",
        "<pl>": "Polish",
        "<pt>": "Portuguese",
        "<ro>": "Romanian",
        "<ru>": "Russian",
        "<sk>": "Slovak",
        "<es>": "Spanish",
        "<sv>": "Swedish",
        "<uk>": "Ukrainian",
    }
    return tag_to_name.get(tag, tag)


def _process_language(
    language: str,
    state: UploadState,
    test_samples: int,
    max_train_seconds: float,
    seed: int,
) -> LanguageStats:
    metadata = _collect_metadata(language=language)
    rng = random.Random(f"{seed}:{language}")
    rng.shuffle(metadata)

    selected_test = metadata[:test_samples]
    selected_train = _select_train_samples(
        candidates=metadata[test_samples:],
        max_train_seconds=max_train_seconds,
    )

    test_keys = {sample.key for sample in selected_test}
    train_keys = {sample.key for sample in selected_train}
    lang_by_key = {
        sample.key: sample.lang for sample in selected_test + selected_train
    }

    _upload_selected_audio(
        language=language,
        state=state,
        train_keys=train_keys,
        test_keys=test_keys,
        lang_by_key=lang_by_key,
    )

    train_duration = sum(sample.duration for sample in selected_train)
    test_duration = sum(sample.duration for sample in selected_test)
    return LanguageStats(
        language=language,
        valid_samples=len(metadata),
        train_samples=len(selected_train),
        train_duration_hours=train_duration / 3600,
        test_samples=len(selected_test),
        test_duration_hours=test_duration / 3600,
    )


def _collect_metadata(language: str) -> list[SampleMeta]:
    metadata: list[SampleMeta] = []
    for source_split, sample in _iter_source_samples(language=language):
        duration = _duration(sample=sample)
        utt_id = _utt_id(sample=sample)
        if utt_id is None or not 0.3 < duration < 15:
            continue
        metadata.append(
            SampleMeta(
                key=SourceKey(split=source_split, utt_id=utt_id),
                duration=duration,
                lang=_lang(sample=sample, language=language),
            )
        )
    return metadata


def _select_train_samples(
    candidates: list[SampleMeta],
    max_train_seconds: float,
) -> list[SampleMeta]:
    selected: list[SampleMeta] = []
    duration = 0.0
    for sample in candidates:
        if duration + sample.duration > max_train_seconds:
            break
        selected.append(sample)
        duration += sample.duration
    return selected


def _upload_selected_audio(
    language: str,
    state: UploadState,
    train_keys: set[SourceKey],
    test_keys: set[SourceKey],
    lang_by_key: dict[SourceKey, str],
) -> None:
    remaining_train = set(train_keys)
    remaining_test = set(test_keys)

    for source_split, sample in _iter_source_samples(language=language):
        utt_id = _utt_id(sample=sample)
        if utt_id is None:
            continue
        key = SourceKey(split=source_split, utt_id=utt_id)
        if key in remaining_train:
            _append_row(
                state=state,
                split="train",
                sample=sample,
                lang=lang_by_key[key],
            )
            remaining_train.remove(key)
        elif key in remaining_test:
            _append_row(
                state=state,
                split="test",
                sample=sample,
                lang=lang_by_key[key],
            )
            remaining_test.remove(key)

        if not remaining_train and not remaining_test:
            break

    if remaining_train or remaining_test:
        LOGGER.warning(
            "%s: missing %d train and %d test samples during audio pass",
            language,
            len(remaining_train),
            len(remaining_test),
        )


def _iter_source_samples(
    language: str, timeout: int = 120
) -> c.Iterator[tuple[str, c.Mapping[str, object]]]:
    """Stream samples from YODAS-Granary, skipping files that hang.

    Uses the ``datasets`` library with streaming enabled. The
    ``HF_HUB_DOWNLOAD_TIMEOUT`` environment variable is set to prevent
    indefinite hangs on problematic parquet files.

    Args:
        language:
            Full language name (e.g. ``"Czech"``).
        timeout:
            Maximum seconds for HTTP downloads. Defaults to 120.

    Yields:
        Tuples of ``(split_name, sample)`` for each valid sample.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(timeout))
    dataset = load_dataset("espnet/yodas-granary", language, streaming=True)
    for split_name, split_dataset in dataset.items():
        for sample in split_dataset:
            yield split_name, sample


def _append_row(
    state: UploadState,
    split: str,
    sample: c.Mapping[str, object],
    lang: str,
) -> None:
    audio = _decode_audio(sample=sample)
    state.buffers[split].append({"audio": audio, "lang": lang})
    if len(state.buffers[split]) >= state.chunk_size:
        _flush_split(state=state, split=split)


def _decode_audio(sample: c.Mapping[str, object]) -> dict[str, object]:
    audio_decoder = sample["audio"]
    audio_samples = audio_decoder.get_all_samples()  # type: ignore[attr-defined]
    tensor = audio_samples.data
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.mean(dim=0)
    return {
        "array": tensor.numpy(),
        "sampling_rate": int(audio_samples.sample_rate),
    }


def _flush_all(state: UploadState) -> None:
    for split in ("train", "test"):
        _flush_split(state=state, split=split)
    _commit_pending(state=state)


def _flush_split(state: UploadState, split: str) -> None:
    rows = state.buffers[split]
    if not rows:
        return

    shard_index = state.shard_indices[split]
    shard_path = state.tmp_dir / f"{split}-{shard_index:05d}.parquet"
    dataset = Dataset.from_list(rows, features=FEATURES)
    dataset.to_parquet(shard_path)

    state.pending_files.append(shard_path)
    state.buffers[split] = []
    state.shard_indices[split] += 1

    if len(state.pending_files) >= SHARDS_PER_COMMIT:
        _commit_pending(state=state)


def _commit_pending(state: UploadState) -> None:
    """Upload all pending shard files in a single commit.

    Raises:
        HfHubHTTPError:
            If the commit fails after retrying on rate limits.
    """
    if not state.pending_files:
        return

    operations = []
    for shard_path in state.pending_files:
        path_in_repo = f"data/{shard_path.name}"
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=shard_path,
            )
        )

    try:
        state.api.create_commit(
            repo_id=state.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {len(operations)} shards",
        )
        LOGGER.info("Committed %d shards", len(operations))
    except HfHubHTTPError as exc:
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            LOGGER.warning("Rate limited, retrying in 60s...")
            time.sleep(60)
            state.api.create_commit(
                repo_id=state.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add {len(operations)} shards",
            )
            LOGGER.info("Committed %d shards (after retry)", len(operations))
        else:
            raise

    for shard_path in state.pending_files:
        shard_path.unlink()
    state.pending_files = []


def _write_readme(repo_id: str, stats: list[LanguageStats]) -> Path:
    total_train = sum(item.train_samples for item in stats)
    total_test = sum(item.test_samples for item in stats)
    total_train_hours = sum(item.train_duration_hours for item in stats)
    total_test_hours = sum(item.test_duration_hours for item in stats)

    rows = "\n".join(
        f"| {item.language} | {item.train_samples:,} | "
        f"{item.train_duration_hours:.2f} | {item.test_samples:,} | "
        f"{item.test_duration_hours:.2f} |"
        for item in stats
    )

    content = f"""---
license: cc-by-4.0
tags:
  - speech
  - language-detection
  - yodas-granary
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: test
        path: data/test-*.parquet
---

# YODAS-Granary Language Detection

Chunked language-detection subset from
[espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary).

## Split strategy

- Test: 100 valid samples per language.
- Train: remaining valid samples, capped at 20 hours per language.
- Valid duration range: 0.3-15 seconds.
- Columns: `audio`, `lang`.

## Summary

| Split | Samples | Duration |
| --- | ---: | ---: |
| Train | {total_train:,} | {total_train_hours:.2f}h |
| Test | {total_test:,} | {total_test_hours:.2f}h |

## Per-language counts

| Language | Train samples | Train hours | Test samples | Test hours |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", streaming=True)
train = dataset["train"]
test = dataset["test"]
```

## Columns

- `audio`: dict with `array` (np.ndarray) and `sampling_rate` (int)
- `lang`: Language tag (e.g., `<da>`, `<en>`)

## Source and licence

Source: `espnet/yodas-granary`. Licence: CC-BY-4.0.
"""
    path = Path("README.granary-upload.md")
    path.write_text(content)
    return path


def _duration(sample: c.Mapping[str, object]) -> float:
    value = sample.get("duration", 0.0)
    return float(value) if value is not None else 0.0


def _utt_id(sample: c.Mapping[str, object]) -> str | None:
    value = sample.get("utt_id")
    return str(value) if value else None


def _lang(sample: c.Mapping[str, object], language: str) -> str:
    value = sample.get("lang")
    return str(value) if value else f"<{language[:2].lower()}>"


if __name__ == "__main__":
    main()
