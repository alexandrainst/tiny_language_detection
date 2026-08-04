# Tiny language detection implementation plan

## Goal

Build small audio language-detection models for edge deployment. The ideal deployed
model size is around 100 KB, but experiments should cover a Pareto frontier from roughly
50 KB to 1 MB.

The main question is whether a compact model should prioritise narrower, higher-quality
coverage or broader language coverage:

1. **Major European languages** — focused deployment track.
2. **Broad European coverage** — all languages available in the 23-language
   YODAS-Granary language-detection dataset.

The broad track is preferable if a compact model maintains strong classification
performance across all languages. If the major-language track performs substantially
better, the focused model may be more useful for deployment.

Chinese is out of scope for now because it is not part of the YODAS-Granary dataset.

## Critical caveat: EU languages

Do not assume that the 23-language YODAS-Granary inventory exactly equals the official
EU-language list.

The desired broad direction is EU-relevant coverage, but the current dataset inventory
must be verified before describing it externally as "all official EU languages". If true
official-EU coverage is required, the dataset may need to be changed or augmented.

Implementation task:

```python
from datasets import load_dataset

ds = load_dataset("saattrupdan/yodas-granary-language-detection", split="train")
print(sorted(set(ds["lang"])))
```

Use the actual values returned by the dataset as the canonical language IDs.

## Language tracks

### Track A: major European languages

Use this focused set:

- English, covering British and American English
- Danish
- German
- Dutch
- French
- Spanish
- Italian

After verifying dataset IDs, define this as a constant in package code, for example in
`src/tiny_language_detection/data/languages.py`.

### Track B: broad 23-language YODAS coverage

Use all 23 languages available in `saattrupdan/yodas-granary-language-detection`.

The old dataset card has listed the following languages, but the implementation must
verify the actual dataset IDs before hard-coding anything:

- Bulgarian
- Croatian
- Czech
- Danish
- Dutch
- English
- Estonian
- Finnish
- French
- German
- Greek
- Hungarian
- Italian
- Latvian
- Lithuanian
- Polish
- Portuguese
- Romanian
- Russian
- Slovak
- Spanish
- Swedish
- Ukrainian

## Model-size Pareto curve

Run experiments across model artefact sizes from approximately 50 KB to 1 MB. Treat
50 KB intervals as a target for the final plot, not as a guarantee that every model can
hit every exact size.

Prioritise dense coverage near the deployment target:

| Bucket | Target artefact size |
| --- | ---: |
| S0 | 50 KB |
| S1 | 100 KB |
| S2 | 150 KB |
| S3 | 200 KB |
| S4 | 300 KB |
| S5 | 500 KB |
| S6 | 750 KB |
| S7 | 1 MB |

For each trained model, record actual size and assign it to the nearest bucket. Use the
actual measured size for Pareto plots.

### Size measurement

Measure storage size from the deployable artefact, not from a training checkpoint.
Training checkpoints include optimiser and scheduler state and are not deployment-size
measurements.

Required artefacts:

- FP32 model export
- quantised export, where applicable
- metadata file with language IDs, feature config, architecture config, and decision
  method

Suggested size fields:

- `storage_kb`: deployable model artefact size on disk
- `ram_kb`: estimated or measured peak runtime RAM
- `params`: trainable parameter count

## Architecture and compression grid

Use the existing `CompactCNNLanguageDetector` as the first architecture family. Generate
model sizes by varying:

- CNN channels
- classifier hidden size
- Mel feature size, if needed
- quantisation precision

Start with this grid, then adjust after measuring actual artefact sizes:

| ID prefix | Channels | Hidden | Notes |
| --- | --- | ---: | --- |
| `micro` | 8, 16, 24 | 32 | 50 KB target after quantisation |
| `tiny` | 12, 24, 32 | 48 | 100 KB target after quantisation |
| `small` | 16, 32, 64 | 64 | 150-300 KB target after quantisation |
| `base` | 32, 64, 128 | 128 | Existing CompactCNN baseline |
| `wide` | 48, 96, 160 | 160 | Upper Pareto reference |

Quantisation variants:

- FP32 baseline
- BF16 or FP16 if supported by export/runtime
- INT8 post-training quantisation
- INT4 or QAT only if the INT8 Pareto curve is insufficient

Do not optimise only for accuracy. The deployment candidate must be judged by the joint
trade-off between size, latency, memory, and classification quality.

## Training protocol

### Dataset filtering

Current code loads all languages in `MulticlassDataset`. Add support for language
filtering before constructing the label map.

Suggested change:

- add `include_languages: set[str] | None` to `MulticlassDataset`
- filter samples before building `lang_to_id`
- save `lang_to_id` and `id_to_lang` with every model artefact
- expose a CLI option such as `--languages major-european`, `--languages yodas-23`, or
  `--language-ids en da ...`

Do not rely on `--num-languages` alone. It changes the output dimension but does not
filter the dataset or define the label mapping.

### Splits

Do not select checkpoints based on test-set accuracy.

Use one of these protocols:

1. If the HF dataset has train/validation/test splits, train on train, select on
   validation, and report final metrics on test.
2. If it only has train/test, create a deterministic stratified validation split from
   train and reserve test for final reporting.

Use the same split seed for all architecture variants unless testing seed sensitivity.

### Default hyperparameters

Use current defaults for the first sweep unless there is a reason to change them:

- epochs: 60
- batch size: 32
- learning rate: 1e-4
- weight decay: 1e-4
- max gradient norm: 0.5
- SpecAugment enabled for training only
- class weights enabled for multi-class training

For final candidates, run at least three random seeds. For exploratory sweeps, one seed
is acceptable, but mark results as preliminary.

## Configurable candidate languages

Users should be able to restrict candidate languages at inference time. For example, a
user may want to choose only between English and Danish.

Restricted candidate sets should improve performance relative to full-set
classification. Do not implement restricted inference only by discarding non-selected
logits and renormalising. That method should exist as a baseline, because the full
language distribution may contain useful evidence for restricted decisions.

## Decision methods

### Method 0: forced full-track prediction

Predict over all languages in the trained track. This is the ordinary multi-class model
and is the baseline for full-track evaluation.

### Method 1: selected-logit renormalisation

Train a full-track classifier. At inference time:

1. compute logits for all languages
2. keep only logits for selected candidate languages
3. apply softmax over the selected logits

This is the simplest restricted-inference baseline.

### Method 2: subset-aware decision head

Train a shared encoder and full-language classifier. Add a small decision head that
sees:

- full logits and/or the penultimate embedding
- a binary mask for selected candidate languages

The head should output corrected logits for all languages. Apply the selected-language
mask before softmax, so probabilities are returned only for the selected set.

Training procedure:

1. sample a training example with true language `y`
2. sample a candidate subset that always includes `y`
3. include between 1 and `k - 1` distractor languages
4. compute the subset-aware loss only over the selected languages

Subset sizes to sample:

- binary subsets
- ternary subsets
- 4-way subsets
- full major-language track
- full 23-language track, where relevant

Start by training the subset-aware head on frozen encoder logits. If it improves over
renormalisation, test end-to-end fine-tuning.

### Method 3: subset-specific calibration heads

For common subsets, train tiny calibration heads on top of the full logits or
embeddings. At minimum, include:

- English+Danish
- English+Danish+German
- Germanic languages: English, Danish, German, Dutch
- Romance languages: French, Spanish, Italian, Portuguese if using the broad track
- the full major-language set

These heads are useful if common deployed subsets matter more than arbitrary subsets.

## Subset evaluation protocol

Evaluate subset methods separately from full-track classification.

Fixed subsets:

- English+Danish
- English+Danish+German
- English+Danish+German+Dutch
- Germanic languages
- Romance languages
- full major-language track
- full 23-language track

Random subsets:

- 20 binary subsets per track
- 20 ternary subsets per track
- 20 four-way subsets per track
- 10 medium subsets per track

For each subset, report macro-F1 and accuracy over examples whose true label is in the
subset. Separately evaluate out-of-selection behaviour if supported.

## Open-set question

Restricted inference may be either forced-choice or open-set:

1. **Forced-choice**: if the user selects English+Danish, every input is classified as
   English or Danish.
2. **Open-set**: the model may return `none_of_the_selected_languages`.

Forced-choice is simpler and should be implemented first. The evaluation code should be
structured so open-set rejection can be added later without redesigning result storage.

## Metrics

Report these metrics for every full-track experiment:

- accuracy
- macro-F1
- balanced accuracy
- per-language accuracy
- confusion matrix
- calibration error, if practical
- latency
- RAM
- storage size
- parameter count

Report these metrics for subset experiments:

- subset accuracy
- subset macro-F1
- average subset accuracy by subset size
- average subset macro-F1 by subset size
- comparison against selected-logit renormalisation

## Result storage

Current result tracking in `src/tiny_language_detection/data/tracking.py` is too narrow
for this plan. Extend it or add a new schema while preserving backwards compatibility
with `data/results.jsonl`.

Recommended fields:

- `model_id`
- `run_id`
- `language_track`: `major-european` or `yodas-23`
- `language_ids`
- `size_target_kb`
- `storage_kb`
- `ram_kb`
- `params`
- `architecture`
- `architecture_config`
- `feature_config`
- `precision`
- `quantisation_method`
- `decision_method`
- `seed`
- `epochs`
- `best_epoch`
- `accuracy`
- `macro_f1`
- `balanced_accuracy`
- `per_language_accuracy`
- `confusion_matrix_path`
- `subset_metrics_path`
- `latency_ms_mean`
- `latency_ms_p95`
- `notes`

Store large nested metrics, such as full subset results and confusion matrices, in
separate JSON files and reference them from JSONL.

## Pareto-frontier comparison

Generate Pareto plots for:

1. major-language accuracy vs storage size
2. broad 23-language accuracy vs storage size
3. major-language macro-F1 vs storage size
4. broad 23-language macro-F1 vs storage size
5. latency vs storage size for deployment candidates

Use actual measured storage size on the x-axis. Use macro-F1 as the main comparison
metric when language imbalance is possible.

Decision rule for coverage:

- Prefer the broad 23-language model if its macro-F1 is within 3 percentage points of
  the major-language model at the same size bucket and latency budget.
- Prefer the major-language model if it is more than 5 percentage points better in
  macro-F1 or is substantially better calibrated.
- Treat differences between 3 and 5 percentage points as a product decision.

These thresholds are defaults and should be replaced if product requirements define a
minimum acceptable accuracy.

## Implementation roadmap

### Phase 1: make language tracks explicit

Files likely to change:

- `src/tiny_language_detection/data/languages.py`
- `src/tiny_language_detection/data/multiclass_dataset.py`
- `src/scripts/train_cnn.py`
- `src/scripts/train_cnn_rnn.py`

Tasks:

1. Inspect actual HF language IDs.
2. Define constants for `MAJOR_EUROPEAN_LANGUAGES` and `YODAS_23_LANGUAGES`.
3. Add dataset filtering by language ID.
4. Add CLI options for language tracks and explicit language IDs.
5. Save language mapping in each model artefact.

### Phase 2: fix evaluation protocol

Files likely to change:

- `src/tiny_language_detection/training.py`
- `src/tiny_language_detection/eval/metrics.py`
- `src/scripts/train_cnn.py`
- new script: `src/scripts/evaluate_model.py`

Tasks:

1. Add validation split support.
2. Select checkpoints on validation metrics, not test metrics.
3. Reserve test metrics for final reporting.
4. Add macro-F1, balanced accuracy, and confusion matrices.
5. Write metrics to structured JSON files.

### Phase 3: run first Pareto sweep

Run the CompactCNN grid for both language tracks. Start with one seed per configuration.
Measure deployable artefact size, latency, and RAM where possible.

Deliverables:

- updated `data/results.jsonl`
- per-run metric JSON files
- Pareto plots for both tracks
- short notes identifying promising size buckets near 100 KB

### Phase 4: add restricted-language evaluation

Files likely to change:

- new script: `src/scripts/evaluate_subsets.py`
- `src/tiny_language_detection/eval/metrics.py`
- possibly `src/tiny_language_detection/inference.py`

Tasks:

1. Implement selected-logit renormalisation baseline.
2. Evaluate fixed and random subsets.
3. Store subset metrics separately from full-track metrics.
4. Compare restricted inference against full-track predictions.

### Phase 5: add subset-aware heads

Files likely to change:

- new module: `src/tiny_language_detection/models/subset_heads.py`
- new script: `src/scripts/train_subset_head.py`
- `src/tiny_language_detection/inference.py`

Tasks:

1. Export logits or embeddings from trained full-track models.
2. Train subset-aware decision heads on sampled language subsets.
3. Train subset-specific heads for common subsets.
4. Compare against selected-logit renormalisation.
5. Include head size in deployment artefact size.

### Phase 6: final candidates

For the best candidates near 100 KB and the best candidates up to 1 MB:

1. rerun with at least three seeds
2. report mean and standard deviation
3. compare major-language and 23-language tracks
4. select recommended deployment candidates

## Acceptance criteria

A result is implementation-ready when another agent can reproduce it from:

- dataset name and split seed
- language track and language IDs
- model architecture config
- feature config
- quantisation config
- training hyperparameters
- decision method
- random seed
- exact git commit

A model is deployment-promising when it:

- is close to the 100 KB target or sits on the Pareto frontier
- has acceptable macro-F1 for its language track
- has no severe per-language failure hidden by aggregate accuracy
- improves when candidate languages are restricted
- has measured latency and memory within the target hardware budget

## Open questions

- Should English be a single class covering both British and American English, or should
  accents be modelled separately and merged at inference time?
- Does the project need true official-EU-language coverage, or is YODAS-Granary's
  23-language inventory the practical broad-coverage target?
- Should restricted inference remain forced-choice, or should it support an
  out-of-selection option?
- Which hardware should define the latency and memory budget?
- What minimum acceptable accuracy or macro-F1 is required around the 100 KB target?
