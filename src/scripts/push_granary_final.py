#!/usr/bin/env python3
"""Build and upload a balanced YODAS-Granary language detection dataset."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

TARGET_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "pl", "pt", "ro", "ru", "sk",
    "es", "sv", "uk"
]

LANG_NAMES = {
    "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "de": "German", "el": "Greek", "hu": "Hungarian",
    "it": "Italian", "lv": "Latvian", "lt": "Lithuanian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "es": "Spanish", "sv": "Swedish", "uk": "Ukrainian"
}


@dataclass
class Sample:
    utt_id: str
    split: str
    parquet_file: str
    duration: float
    lang: str
    audio_path: str


def download_parquet_until_enough(lang_code: str, cache_dir: Path, seed: int = 42, max_files: int = 100) -> list[Sample]:
    """Download parquet files incrementally until we have enough samples."""
    rng = random.Random(f"{seed}:{lang_code}")
    all_samples: list[Sample] = []
    files_checked = 0
    
    for split in ["asr_only", "ast"]:
        for file_idx in range(max_files):
            parquet_file = f"data/{lang_code}000/{split}/{file_idx:08d}.parquet"
            try:
                local_path = hf_hub_download(
                    repo_id="espnet/yodas-granary", filename=parquet_file,
                    repo_type="dataset", revision="969944574ea3f37890beaf67ea651e160cfaf043",
                )
            except Exception as e:
                if "404" in str(e) or "403" in str(e):
                    break
                raise
            
            files_checked += 1
            table = pq.read_table(local_path)
            
            for row in table.to_pylist():
                utt_id, duration, lang = row.get("utt_id"), row.get("duration"), row.get("lang")
                if utt_id and duration and lang and 0.3 < duration < 15.0:
                    all_samples.append(Sample(utt_id, split, parquet_file, duration, lang, local_path))
            
            if len(all_samples) >= 10_500:
                break
        
        if files_checked >= max_files * 2:
            break
    
    rng.shuffle(all_samples)
    test_samples = all_samples[:100]
    train_samples, train_dur = [], 0.0
    for s in all_samples[100:]:
        if train_dur >= 20 * 3600:
            break
        train_samples.append(s)
        train_dur += s.duration
    
    LOGGER.info("Selected %d test + %d train (%.2fh) from %d files for %s",
                len(test_samples), len(train_samples), train_dur/3600, files_checked, lang_code)
    return test_samples + train_samples


def stream_audio(samples: list[Sample], lang_code: str) -> list[dict]:
    """Extract audio for selected samples."""
    rows = []
    by_parquet: dict[str, list[Sample]] = {}
    for s in samples:
        by_parquet.setdefault(s.parquet_file, []).append(s)
    
    LOGGER.info("Streaming audio for %d samples from %d files", len(samples), len(by_parquet))
    
    for parquet_file, samps in by_parquet.items():
        try:
            local_path = hf_hub_download(repo_id="espnet/yodas-granary", filename=parquet_file,
                                         repo_type="dataset", revision="969944574ea3f37890beaf67ea651e160cfaf043")
        except Exception as e:
            LOGGER.error("Failed to download %s: %s", parquet_file, e)
            continue
        
        table = pq.read_table(local_path)
        utt_to_row = {row["utt_id"]: row for row in table.to_pylist()}
        
        for s in samps:
            if s.utt_id in utt_to_row:
                audio = utt_to_row[s.utt_id].get("audio")
                if audio:
                    rows.append({"audio": audio, "lang": s.lang})
        
        Path(local_path).unlink(missing_ok=True)
    
    return rows


def main() -> None:
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
    
    repo_id = "saattrupdan/yodas-granary-language-detection"
    cache_dir = Path("data/granary_cache")
    output_dir = Path("data/granary_final")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_train, all_test = [], []
    
    for lang_code in TARGET_LANGUAGES:
        lang_name = LANG_NAMES[lang_code]
        LOGGER.info("Processing %s (%s)...", lang_name, lang_code)
        
        meta_file = cache_dir / lang_code / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("complete"):
                LOGGER.info("Skipping %s (cached)", lang_name)
                cached_data = pq.read_table(cache_dir / lang_code / f"cached-{lang_code}.parquet").to_pylist()
                for row in cached_data:
                    if row["split"] == "test":
                        all_test.append(row["data"])
                    else:
                        all_train.append(row["data"])
                continue
        
        samples = download_parquet_until_enough(lang_code, cache_dir)
        test_samples, train_samples = samples[:100], samples[100:]
        
        test_data = stream_audio(test_samples, lang_code)
        train_data = stream_audio(train_samples, lang_code)
        LOGGER.info("%s: test=%d, train=%d", lang_name, len(test_data), len(train_data))
        
        cached_data = [{"split": "test", "data": r, "lang_code": lang_code} for r in test_data]
        cached_data += [{"split": "train", "data": r, "lang_code": lang_code} for r in train_data]
        
        cache_dir_lang = cache_dir / lang_code
        cache_dir_lang.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(cached_data), cache_dir_lang / f"cached-{lang_code}.parquet")
        pq.write_table(pa.Table.from_pylist([{"split": m["split"], **m["data"]} for m in cached_data]),
                       output_dir / f"{lang_code}.parquet")
        
        for r in test_data:
            all_test.append(r)
        for r in train_data:
            all_train.append(r)
        
        with open(meta_file, "w") as f:
            json.dump({"lang_code": lang_code, "test": len(test_data), "train": len(train_data), "complete": True}, f, indent=2)
    
    LOGGER.info("Total: test=%d, train=%d", len(all_test), len(all_train))
    
    # Write final shards
    chunk_size = 1000
    for i in range(0, len(all_train), chunk_size):
        chunk = all_train[i:i+chunk_size]
        pq.write_table(pa.Table.from_pylist(chunk), output_dir / f"train-{i//chunk_size:06d}.parquet")
    for i in range(0, len(all_test), chunk_size):
        chunk = all_test[i:i+chunk_size]
        pq.write_table(pa.Table.from_pylist(chunk), output_dir / f"test-{i//chunk_size:06d}.parquet")
    
    LOGGER.info("Created %d train + %d test shards in %s",
                len(list(output_dir.glob("train-*.parquet"))),
                len(list(output_dir.glob("test-*.parquet"))), output_dir)
    
    # Upload
    LOGGER.info("Uploading to %s...", repo_id)
    result = subprocess.run(["uv", "run", "hf", "upload", repo_id, str(output_dir / "*.parquet"), "."],
                            capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.error("Upload failed:\n%s", result.stderr)
    else:
        LOGGER.info("Upload completed:\n%s", result.stdout)


if __name__ == "__main__":
    main()
