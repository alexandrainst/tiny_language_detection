#!/usr/bin/env python3
"""Push YODAS-Granary test set to HF - train/test splits, audio+lang columns only."""

import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict

TARGET_LANGS = [
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "German", "Greek", "Hungarian", "Italian", "Latvian",
    "Lithuanian", "Polish", "Portuguese", "Romanian", "Russian", "Slovak",
    "Spanish", "Swedish", "Ukrainian",
]

def main():
    output_dir = Path("data/granary_temp")
    
    print("Loading metadata...")
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    train_samples = []
    test_samples = []
    
    for lang in TARGET_LANGS:
        print(f"\nStreaming {lang}...")
        ds = load_dataset("espnet/yodas-granary", lang, streaming=True)
        
        # Load saved sample IDs
        with open(output_dir / f"{lang.lower()}_train.json") as f:
            train_ids = {s["utt_id"] for s in json.load(f)}
        with open(output_dir / f"{lang.lower()}_test.json") as f:
            test_ids = {s["utt_id"] for s in json.load(f)}
        
        # Stream and collect
        train_count = test_count = 0
        for split in ds.keys():
            for sample in ds[split]:
                utt_id = sample.get("utt_id", "")
                if utt_id in train_ids:
                    train_samples.append({
                        "audio": sample["audio"],
                        "lang": sample.get("lang", f"<{lang[:2].lower()}>"),
                    })
                    train_count += 1
                elif utt_id in test_ids:
                    test_samples.append({
                        "audio": sample["audio"],
                        "lang": sample.get("lang", f"<{lang[:2].lower()}>"),
                    })
                    test_count += 1
        
        print(f"  Collected: {train_count} train, {test_count} test")
    
    print(f"\nCreating datasets...")
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Test:  {len(test_samples):,} samples")
    
    train_dataset = Dataset.from_list(train_samples)
    test_dataset = Dataset.from_list(test_samples)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    
    repo = "saattrupdan/yodas-granary-test"
    print(f"\nPushing to {repo} (private)...")
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
size_categories:
  - 10K<n<100K
---

# YODAS-Granary Test Set

Balanced subset from [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary) for multi-label language detection.

## Splits

| Split | Samples | Duration |
|-------|---------|----------|
| Train | {len(train_samples):,} | {metadata['total_train_h']:.2f}h |
| Test  | {len(test_samples):,} | {metadata['total_test_h']:.2f}h |

## Languages ({len(TARGET_LANGS)} total)

Danish, English, Swedish, German, Finnish, French, Spanish, Italian, Dutch, Polish, 
Bulgarian, Croatian, Czech, Estonian, Greek, Hungarian, Latvian, Lithuanian, 
Portuguese, Romanian, Russian, Slovak, Ukrainian

## Usage

```python
from datasets import load_dataset

# Streaming
ds = load_dataset("saattrupdan/yodas-granary-test", streaming=True)
train = ds["train"]
test = ds["test"]

# Full download
ds = load_dataset("saattrupdan/yodas-granary-test")
```

## Columns

- `audio`: Audio sample (dict with `array`, `sampling_rate`)
- `lang`: Language tag (e.g., `<da>`, `<en>`)

## License

CC-BY-4.0 (inherited from YODAS-Granary)
"""
    
    Path("README.md").write_text(readme)
    
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    
    print(f"\n✓ Done!")
    print(f"  https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
