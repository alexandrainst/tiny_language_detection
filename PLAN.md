# Tiny language detection plan

## Goal

Build a small audio language detection model for edge deployment. The ideal model size
is around 100 KB, but experiments should cover a Pareto frontier from roughly 50 KB to
1 MB.

The model should support language detection among:

- English, covering both British and American English
- Danish
- German
- Dutch
- French
- Spanish
- Italian
- Chinese

The user should also be able to restrict the candidate language set, for example to
English and Danish. Restricted language sets should improve classification performance
relative to the full multi-language setting.

## Core requirements

### Model-size Pareto curve

Run experiments across model sizes from approximately 50 KB to 1 MB.

Use 50 KB intervals as the nominal target, but allow practical deviations because model
size changes discretely with architecture, feature configuration, and quantisation.
Prioritise dense coverage around the expected deployment target of 50-200 KB, especially
near 100 KB.

For each model variant, report:

- model size on disk
- number of parameters
- quantisation method, if any
- latency on target or representative hardware
- peak RAM usage, if practical
- full-language accuracy and macro-F1
- per-language accuracy
- confusion matrix
- calibration metrics, if practical

The main deliverable should be a Pareto curve showing size against accuracy, macro-F1,
latency, and memory use.

### Configurable candidate languages

The model should support inference with a user-selected subset of candidate languages,
such as:

- English + Danish
- English + Danish + German
- Germanic languages only
- the full language set

When a subset is selected, the model should output probabilities over only that subset.
However, the implementation should not simply discard the logits for non-selected
languages and renormalise the remaining logits. The full language distribution may
contain useful evidence for restricted classification. For example, high German or Dutch
scores may still help distinguish Danish from English.

## Candidate approaches

### Baseline: selected-logit renormalisation

Train a standard full-language classifier. At inference time, keep only the logits for
the selected languages and renormalise them.

This is simple and should be included as a baseline, but it is not expected to be the
best approach for restricted language sets.

### Subset-aware decision head

Train a shared tiny audio encoder on the full language set. Add a subset-aware decision
head that receives:

- the model embedding and/or full-language logits
- a mask indicating the user-selected candidate languages

The head should output probabilities over the selected languages. During training,
sample many candidate-language subsets, including binary, ternary, medium-sized, and
full-set classification tasks. This teaches the model to use information from
non-selected languages when making a restricted decision.

### Subset-specific calibration heads

For common user-selected subsets, train very small calibration heads on top of the full
model logits. Examples include English+Danish and English+Danish+German.

These heads may provide strong performance improvements with very little extra memory.
They are also useful as a comparison against the general subset-aware head.

## Experiment matrix

Evaluate at least the following dimensions:

| Dimension | Values |
| --- | --- |
| Model size | ~50 KB to ~1 MB, dense around ~100 KB |
| Language set | full set, common subsets, random subsets |
| Decision method | renormalised logits, subset-aware head, subset-specific heads |
| Quantisation | unquantised baseline, int8, lower-bit options if supported |
| Architecture | current CNN baseline, smaller/larger variants, optional CNN-RNN |

## Evaluation tasks

For the full language set, evaluate standard multi-class classification over all target
languages.

For restricted language sets, evaluate:

- binary subsets, especially English+Danish
- ternary subsets
- random subsets of different sizes
- linguistically related subsets, such as Germanic languages
- the full language set as a control

Report whether restricted candidate sets improve performance compared with full-set
classification.

Also decide whether inference should be forced-choice or allow an out-of-selection
prediction. For example, if the user selects English+Danish but the input is actually
German, the model could either:

1. force a choice between English and Danish, or
2. return `none_of_the_selected_languages`.

This decision affects both training data construction and evaluation metrics.

## Initial implementation steps

1. Add experiment configuration for the expanded language set.
2. Define model-size targets and architecture/quantisation variants.
3. Implement the selected-logit renormalisation baseline.
4. Implement evaluation for arbitrary language subsets.
5. Train and evaluate the first Pareto sweep.
6. Add the subset-aware decision head.
7. Add subset-specific calibration heads for common subsets.
8. Compare all decision methods across the Pareto curve.

## Open questions

- Should English be a single class covering both British and American English, or should
  accents be modelled separately and merged at inference time?
- Should restricted inference be forced-choice, or should it support an
  out-of-selection option?
- Which hardware should define the latency and memory budget?
- What minimum acceptable accuracy is required around the 100 KB target?
- Which Chinese varieties or datasets should be included, and should they be treated as
  one class?
