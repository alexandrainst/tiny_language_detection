---
license: cc-by-4.0
language:
  - bu
  - cr
  - cz
  - da
  - du
  - en
  - es
  - fi
  - fr
  - ge
  - gr
  - hu
  - it
  - la
  - li
  - po
  - po
  - ro
  - ru
  - sl
  - sp
  - sw
  - uk
tags:
  - speech
  - language-detection
  - yodas-granary
size_categories:
  - 10K<n<100K
---

# YODAS-Granary Test Set

Balanced test subset from [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary) for multi-label language detection.

## Languages (23 total)

| Language | Code | Samples | Duration | Avg Duration |
|----------|------|---------|----------|--------------|
| Bulgarian | bu | 500 | 0.45h | 3.3s |
| Croatian | cr | 257 | 0.21h | 3.0s |
| Czech | cz | 500 | 0.43h | 3.1s |
| Danish | da | 500 | 0.39h | 2.8s |
| Dutch | du | 500 | 0.26h | 1.9s |
| English | en | 500 | 0.68h | 4.9s |
| Estonian | es | 500 | 0.44h | 3.1s |
| Finnish | fi | 500 | 0.39h | 2.8s |
| French | fr | 500 | 0.31h | 2.2s |
| German | ge | 500 | 0.37h | 2.6s |
| Greek | gr | 454 | 0.53h | 4.2s |
| Hungarian | hu | 500 | 0.37h | 2.6s |
| Italian | it | 500 | 0.36h | 2.6s |
| Latvian | la | 70 | 0.08h | 3.9s |
| Lithuanian | li | 366 | 0.48h | 4.7s |
| Polish | po | 500 | 0.32h | 2.3s |
| Portuguese | po | 500 | 0.16h | 1.2s |
| Romanian | ro | 500 | 0.69h | 4.9s |
| Russian | ru | 500 | 0.28h | 2.0s |
| Slovak | sl | 260 | 0.29h | 4.1s |
| Spanish | sp | 500 | 0.28h | 2.0s |
| Swedish | sw | 500 | 0.29h | 2.1s |
| Ukrainian | uk | 500 | 0.44h | 3.1s |

## Totals
- **Languages**: 23
- **Samples**: 10,407
- **Duration**: 8.48h

## Usage

```python
from datasets import load_dataset

# Streaming (recommended)
ds = load_dataset("saattrupdan/yodas-granary-test", streaming=True)

# Full download
ds = load_dataset("saattrupdan/yodas-granary-test")
```

## License

CC-BY-4.0 (inherited from YODAS-Granary)
