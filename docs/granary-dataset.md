# YODAS-Granary Language Detection Dataset

**Dataset:** https://huggingface.co/datasets/saattrupdan/yodas-granary-language-detection

**Created:** 2026-07-23  
**Source:** [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary)

## Summary

A balanced 23-language speech dataset for language detection, built from YODAS-Granary.

| Split | Samples | Duration |
|-------|---------|----------|
| Train | 211,217 | ~415 hours |
| Test  | 2,300   | ~4.5 hours |

**23 languages:** Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Ukrainian

**Per language:**
- Test: 100 samples (balanced)
- Train: up to 20 hours (remaining valid samples)

## Data Processing

**Filtering:** Only samples with duration 0.3–15 seconds  
**Sampling:** Random shuffle with deterministic seed per language  
**Format:** 16kHz mono audio, parquet shards (1000 samples each)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("saattrupdan/yodas-granary-language-detection")

# Train on 211k samples
train_audio = ds["train"][0]["audio"]  # 16kHz array

# Test on balanced 2.3k samples (100 per language)
test_sample = ds["test"][0]
```

## Scripts

- `src/scripts/push_granary_final.py` — Builds and uploads the dataset
- Caches metadata in `data/granary_cache/` for fast resume

## Notes

- Built using streaming download (no local caching of full source dataset)
- Upload parallelized with 4 concurrent workers
- Total upload: 215 parquet files (~28GB)
