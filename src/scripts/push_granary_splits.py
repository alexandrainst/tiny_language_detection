#!/usr/bin/env python3
"""Stream YODAS-Granary → train/test HF dataset.

Split strategy:
- Test: 100 samples per language
- Train: max 20 hours per language (remaining after test)
"""

from datasets import load_dataset, Dataset, DatasetDict
import random

TARGET_LANGS = [
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "German", "Greek", "Hungarian", "Italian", "Latvian",
    "Lithuanian", "Polish", "Portuguese", "Romanian", "Russian", "Slovak",
    "Spanish", "Swedish", "Ukrainian",
]

TEST_SAMPLES = 100
MAX_TRAIN_DURATION_S = 20 * 3600  # 20 hours in seconds

print("Step 1: Collecting sample IDs from all languages...")
all_train_ids = []  # List of (lang_name, utt_id, lang_tag)
all_test_ids = []

for lang_idx, lang in enumerate(TARGET_LANGS):
    print(f"  [{lang_idx+1}/23] {lang}...", flush=True)
    ds = load_dataset("espnet/yodas-granary", lang, streaming=True)
    
    # Collect all valid samples
    samples = []
    for split in ds.keys():
        for sample in ds[split]:
            dur = float(sample.get('duration', 0))
            if 0.3 < dur < 15:
                samples.append({
                    "utt_id": sample.get("utt_id"),
                    "duration": dur,
                    "lang": sample.get("lang", f"<{lang[:2].lower()}>"),
                })
    
    random.shuffle(samples)
    
    # First 100 to test, rest to train (up to 20h)
    test_samples = samples[:TEST_SAMPLES]
    train_pool = samples[TEST_SAMPLES:]
    
    train_samples = []
    train_dur = 0.0
    for s in train_pool:
        if train_dur < MAX_TRAIN_DURATION_S:
            train_samples.append(s)
            train_dur += s["duration"]
        else:
            break
    
    all_test_ids.extend([(lang, s["utt_id"], s["lang"]) for s in test_samples])
    all_train_ids.extend([(lang, s["utt_id"], s["lang"]) for s in train_samples])
    
    print(f"    Test: {len(test_samples)}, Train: {len(train_samples)} ({train_dur/3600:.2f}h)")

print(f"\nTotal IDs collected:")
print(f"  Train: {len(all_train_ids):,}")
print(f"  Test:  {len(all_test_ids):,}")

# Step 2: Stream audio and build datasets
print("\nStep 2: Streaming audio for datasets...")

def stream_audio(ids_list, name):
    samples = []
    for i, (lang_name, utt_id, lang_tag) in enumerate(ids_list):
        ds = load_dataset("espnet/yodas-granary", lang_name, streaming=True)
        found = False
        for split in ds.keys():
            if found:
                break
            for sample in ds[split]:
                if sample.get("utt_id") == utt_id:
                    samples.append({
                        "audio": sample["audio"],
                        "lang": lang_tag,
                    })
                    found = True
                    break
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i+1}/{len(ids_list)}...", flush=True)
    return samples

train_samples = stream_audio(all_train_ids, "Train")
test_samples = stream_audio(all_test_ids, "Test")

print(f"\nCreating HF datasets...")
train_ds = Dataset.from_list(train_samples)
test_ds = Dataset.from_list(test_samples)
dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

repo = "saattrupdan/yodas-granary-test"
print(f"Pushing to {repo} (private)...")
dataset_dict.push_to_hub(repo, private=True)

# README
readme = f"""---
license: cc-by-4.0
tags:
  - speech
  - language-detection
  - yodas-granary
---

# YODAS-Granary Language Detection

Multi-label language detection from [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary).

## Splits

| Split | Samples |
|-------|---------|
| Train | {len(train_samples):,} (max 20h/lang) |
| Test  | {len(test_samples):,} (100/lang) |

## Languages ({len(TARGET_LANGS)})

Danish, English, Swedish, German, Finnish, French, Spanish, Italian, Dutch, Polish, 
Bulgarian, Croatian, Czech, Estonian, Greek, Hungarian, Latvian, Lithuanian, 
Portuguese, Romanian, Russian, Slovak, Ukrainian

## Usage

```python
from datasets import load_dataset
ds = load_dataset("saattrupdan/yodas-granary-test", streaming=True)
```

## Columns

- `audio`: dict with `array` (np.ndarray) and `sampling_rate` (int)
- `lang`: Language tag (e.g., `<da>`, `<en>`)

## License

CC-BY-4.0
"""
from pathlib import Path
Path("README.md").write_text(readme)

from huggingface_hub import HfApi
HfApi().upload_file("README.md", "README.md", repo, repo_type="dataset")

print(f"\n✓ Done!")
print(f"  https://huggingface.co/datasets/{repo}")
