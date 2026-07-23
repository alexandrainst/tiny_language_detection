#!/usr/bin/env python3
"""Stream YODAS-Granary and push directly to HF - no local storage."""

from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
import random

TARGET_LANGS = [
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "German", "Greek", "Hungarian", "Italian", "Latvian",
    "Lithuanian", "Polish", "Portuguese", "Romanian", "Russian", "Slovak",
    "Spanish", "Swedish", "Ukrainian",
]

TEST_SAMPLES = 100

# Collect all samples in memory (just IDs and lang, not audio yet)
print("Collecting sample IDs from all languages...")
train_ids = []
test_ids = []

for lang in TARGET_LANGS:
    print(f"  {lang}...", flush=True)
    ds = load_dataset("espnet/yodas-granary", lang, streaming=True)
    
    all_ids = []
    for split in ds.keys():
        for sample in ds[split]:
            dur = float(sample.get('duration', 0))
            if 0.3 < dur < 15:
                all_ids.append({
                    "lang": sample.get("lang", f"<{lang[:2].lower()}>"),
                    "utt_id": sample.get("utt_id"),
                    "lang_name": lang,  # to reload the right config
                })
    
    random.shuffle(all_ids)
    test_ids.extend(all_ids[:TEST_SAMPLES])
    train_ids.extend(all_ids[TEST_SAMPLES:])
    print(f"    Test: {len(all_ids[:TEST_SAMPLES])}, Train: {len(all_ids[TEST_SAMPLES:])}")

print(f"\nTotal: {len(train_ids)} train, {len(test_ids)} test")

# Now stream audio and build datasets
print("\nStreaming audio for train set...")
train_samples = []
for i, item in enumerate(train_ids):
    ds = load_dataset("espnet/yodas-granary", item["lang_name"], streaming=True)
    for split in ds.keys():
        for sample in ds[split]:
            if sample.get("utt_id") == item["utt_id"]:
                train_samples.append({
                    "audio": sample["audio"],
                    "lang": item["lang"],
                })
                break
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(train_ids)}...", flush=True)

print(f"\nStreaming audio for test set...")
test_samples = []
for i, item in enumerate(test_ids):
    ds = load_dataset("espnet/yodas-granary", item["lang_name"], streaming=True)
    for split in ds.keys():
        for sample in ds[split]:
            if sample.get("utt_id") == item["utt_id"]:
                test_samples.append({
                    "audio": sample["audio"],
                    "lang": item["lang"],
                })
                break
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_ids)}...", flush=True)

# Create and push
print(f"\nCreating datasets...")
train_dataset = Dataset.from_list(train_samples)
test_dataset = Dataset.from_list(test_samples)
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

repo = "saattrupdan/yodas-granary-test"
print(f"Pushing to {repo}...")
dataset_dict.push_to_hub(repo, private=True)

# README
readme = f"""---
license: cc-by-4.0
language:
"""
for lang in TARGET_LANGS:
    readme += f"  - {lang[:2].lower()}\n"

readme += f"""tags:
  - speech
  - language-detection
  - yodas-granary
---

# YODAS-Granary Language Detection

Multi-label language detection dataset from YODAS-Granary.

## Splits
- Train: {len(train_samples):,} samples
- Test: {len(test_samples):,} samples ({TEST_SAMPLES} per language)

## Languages: {len(TARGET_LANGS)}

## Usage
```python
from datasets import load_dataset
ds = load_dataset("saattrupdan/yodas-granary-test", streaming=True)
```

## License: CC-BY-4.0
"""
Path("README.md").write_text(readme)

from huggingface_hub import HfApi
HfApi().upload_file("README.md", "README.md", repo, repo_type="dataset")

print(f"\n✓ Done! https://huggingface.co/datasets/{repo}")
