#!/usr/bin/env python3
"""Build train/test: 100 samples test per language, rest train."""

from datasets import load_dataset
import json, random
from pathlib import Path

TARGET_LANGS = [
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "German", "Greek", "Hungarian", "Italian", "Latvian",
    "Lithuanian", "Polish", "Portuguese", "Romanian", "Russian", "Slovak",
    "Spanish", "Swedish", "Ukrainian",
]

TEST_SAMPLES = 100
output_dir = Path("data/granary_temp")
output_dir.mkdir(parents=True, exist_ok=True)

for i, lang in enumerate(TARGET_LANGS):
    print(f"[{i+1}/23] {lang}...", flush=True)
    
    ds = load_dataset("espnet/yodas-granary", lang, streaming=True)
    all_samples = []
    
    for split in ds.keys():
        for sample in ds[split]:
            dur = float(sample.get('duration', 0))
            if 0.3 < dur < 15:
                all_samples.append({
                    "lang": sample.get("lang", f"<{lang[:2].lower()}>"),
                    "duration": dur,
                    "utt_id": sample.get("utt_id"),
                })
    
    random.shuffle(all_samples)
    test = all_samples[:TEST_SAMPLES]
    train = all_samples[TEST_SAMPLES:]
    
    with open(output_dir / f"{lang.lower()}_test.json", 'w') as f:
        json.dump(test, f)
    with open(output_dir / f"{lang.lower()}_train.json", 'w') as f:
        json.dump(train, f)
    
    print(f"  Test: {len(test)}, Train: {len(train)}", flush=True)

print("\n✓ Done!")
