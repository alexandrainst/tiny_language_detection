#!/usr/bin/env python3
"""Build balanced train/test from YODAS-Granary - both splits, 0.5h test per language."""

import json
import random
from pathlib import Path

from datasets import load_dataset

TARGET_LANGS = [
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

TEST_DURATION_PER_LANG = 0.5  # hours


def sample_language(lang_name: str, output_dir: Path) -> dict:
    """Stream through both splits, allocate 0.5h to test, rest to train."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {lang_name}")
    print(f"{'=' * 60}")

    ds = load_dataset("espnet/yodas-granary", lang_name, streaming=True)

    # First pass: collect all samples from both splits
    all_samples = []
    for split in ds.keys():
        print(f"  Scanning split: {split}")
        for i, sample in enumerate(ds[split]):
            try:
                dur = float(sample.get("duration", 0))
                if 0.3 < dur < 15:
                    all_samples.append(
                        {
                            "lang": sample.get("lang", f"<{lang_name.lower()[:2]}>"),
                            "duration": dur,
                            "utt_id": sample.get("utt_id", f"{lang_name}_{i}"),
                            "split": split,
                        }
                    )
            except Exception:
                continue

        print(f"    Found {len(all_samples)} total so far")

    # Shuffle for random split
    random.shuffle(all_samples)

    # Allocate to test until we hit 0.5h
    test_samples = []
    train_samples = []
    test_dur = 0.0
    target_test_dur = TEST_DURATION_PER_LANG * 3600  # seconds

    for sample in all_samples:
        if test_dur < target_test_dur:
            test_samples.append(sample)
            test_dur += sample["duration"]
        else:
            train_samples.append(sample)

    # Save
    with open(output_dir / f"{lang_name.lower()}_test.json", "w") as f:
        json.dump(test_samples, f)
    with open(output_dir / f"{lang_name.lower()}_train.json", "w") as f:
        json.dump(train_samples, f)

    total_dur = sum(s["duration"] for s in all_samples)
    stats = {
        "language": lang_name,
        "train_samples": len(train_samples),
        "train_duration_h": (total_dur - test_dur) / 3600,
        "test_samples": len(test_samples),
        "test_duration_h": test_dur / 3600,
        "total_duration_h": total_dur / 3600,
    }

    print(f"  Test: {len(test_samples)} samples, {test_dur / 3600:.2f}h")
    print(
        f"  Train: {len(train_samples)} samples, {(total_dur - test_dur) / 3600:.2f}h"
    )

    return stats


def main() -> None:
    output_dir = Path("data/granary_temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files
    for f in output_dir.glob("*.json"):
        f.unlink()

    all_stats = []

    for lang in TARGET_LANGS:
        stats = sample_language(lang, output_dir)
        all_stats.append(stats)
        print(f"\nProgress: {len(all_stats)}/{len(TARGET_LANGS)}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total_train = sum(s["train_samples"] for s in all_stats)
    total_test = sum(s["test_samples"] for s in all_stats)
    total_train_h = sum(s["train_duration_h"] for s in all_stats)
    total_test_h = sum(s["test_duration_h"] for s in all_stats)

    for s in all_stats:
        print(
            f"{s['language']:15s}: Train {s['train_samples']:4d} ({s['train_duration_h']:5.2f}h) | Test {s['test_samples']:4d} ({s['test_duration_h']:5.2f}h)"
        )

    print(f"\nTotal Train: {total_train:,} samples, {total_train_h:.2f}h")
    print(f"Total Test:  {total_test:,} samples, {total_test_h:.2f}h")
    print(
        f"Grand Total: {total_train + total_test:,} samples, {total_train_h + total_test_h:.2f}h"
    )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "languages": all_stats,
                "total_train": total_train,
                "total_test": total_test,
                "total_train_h": total_train_h,
                "total_test_h": total_test_h,
            },
            f,
            indent=2,
        )

    print(f"\n✓ All {len(TARGET_LANGS)} languages processed!")
    print("Next: uv run python3 src/scripts/push_granary_test.py")


if __name__ == "__main__":
    main()
