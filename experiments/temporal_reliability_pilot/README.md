# Temporal grounding and reliability: first experimental pilot

This is a **standalone, leakage-conscious pilot** for the India street-view crop-mapping project. It operates on precomputed ground-image and satellite-season embeddings. It does **not** claim to implement the full proposed temporal cross-attention model or calibrated source reliability.

## Research questions, in order

1. **Independent signal**: does a ground representation add held-out crop information beyond a satellite representation?
2. **Temporal anchoring**: does a month/year ground-image timestamp improve fusion beyond untimed fusion, and beyond a timestamp-shuffled control?
3. **Observation quality**: does a simple, measured-quality-conditioned fusion feature improve robustness to bad ground-field associations or missing satellite signal?
4. **Spatial generalization**: do these results survive blocked holdout and a configurable geographic buffer?

Start with these falsifiable, inexpensive tests before building a transformer, multimodal teacher, or uncertainty gate. Do not interpret ordinary classifier confidence or a linear quality interaction as calibrated modality reliability.

## Inputs

Prepare a CSV with:
- `field_id`, `lat`, `lon`, `capture_year`, `capture_month`, `crop_label`;
- numerical `s_*` columns: satellite seasonal embeddings (same season as label);
- numerical `g_*` columns: ground/VLM image embeddings;
- optional `ground_quality` (numeric in [0,1]); can be an audited visibility score, **not** a ground-truth crop correctness indicator.

Keep all field repeats together. For best scientific validity, test labels should be *independently expert verified*, not VLM pseudo-labels used for training. Do not include district crop proportions for the target year. The labels must refer to crops in the ground image's growing season.

Example:
```bash
pip install -r requirements.txt
python generate_demo.py --output /tmp/crop_pilot.csv
python pilot.py --input /tmp/crop_pilot.csv --output /tmp/crop_pilot_results.json
python -m unittest discover -s tests -v
```
The demo is **synthetic and not a scientific result**. To use real data, export aligned embeddings and metadata following [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Experimental matrix

- `satellite`: only `s_*`
- `ground`: only `g_*`
- `concat`: satellite + ground embeddings
- `month`: concatenation + cyclic capture-month features
- `month_shuffled`: same architecture with capture months independently permuted in train and test
- `month_interaction`: ground features multiplied by cyclic month encodings
- `quality_interaction`: time interactions plus ground-quality and ground × quality

Reported: accuracy, macro-F1, per-class F1, train/test counts. `concat` also undergoes a **test-only** ground-feature permutation (wrong-field association proxy) and satellite zeroing (missing satellite signal proxy); this exposes brittleness without retraining.

## Limitations and follow-ups

- The representation is per-season, so the month interaction is a **screening test**, not genuine per-date temporal alignment. A positive result justifies a timestamp-aware satellite sequence encoder; a negative result does not disprove it.
- The quality interaction is *not* learned predictive-risk supervision. A later reliability experiment must use out-of-fold source errors on independent expert labels, with calibration and selective-risk evaluation.
- Shuffling ground embeddings is a deliberately harsh correspondence corruption, not a realistic error distribution. Stratify by crop, month and region in follow-up tests.
- Geographic grids use approximate latitude/longitude blocks; the optional haversine purge suppresses near-neighbor leakage. Spatial block size and buffer must be chosen independently of final test results.
- Month and geography may be correlated. Report month-by-class and region-by-month distributions; test additional years and the European LUCAS dataset after harmonizing crop labels and point/parcel support.
- Future historical district crop statistics must be **lagged and available before the target season**; compare true, stale and shuffled history with regional holdouts.

No existing repository scripts are modified.

---

## Updated PI-facing experiments (v2, recommended)

The original `pilot.py` is retained as a cheap **pooled-embedding diagnostic**. It cannot test genuine temporal interaction or reliable modality-risk estimation. The recommended first study uses real time-indexed satellite features and separates expert-verified labels from VLM pseudo-labels.

Read **[PI_EXPERIMENT_DESIGN.md](PI_EXPERIMENT_DESIGN.md)** first: for each experiment it specifies the scientific motivation, contrast, expected interpretation, decision gate, and what could falsify our hypothesis.

Read **[SEQUENCE_DATA_CONTRACT.md](SEQUENCE_DATA_CONTRACT.md)** to prepare paired input files. This repository does **not** yet contain aligned real S1/S2 sequences exported in this format; building that exporter is the first real-data integration step.

```bash
pip install -r requirements.txt
python make_synthetic_sequence.py --output-dir /tmp/crop_sequence_demo
python run_study.py \
  --metadata /tmp/crop_sequence_demo/metadata.csv \
  --sequences /tmp/crop_sequence_demo/sequences.npz \
  --output /tmp/crop_sequence_demo/results.json \
  --block-degrees 1.0 --buffer-km 5
python -m unittest discover -s tests -v
```

`run_study.py` implements four stages:

- **E0 / data audit**: VLM-vs-expert error distribution by capture month and crop, expert prevalence, audited field matching.
- **E1 / temporally indexed satellite baselines**: satellite sequence; ground; late fusion; satellite + date alone; dated fusion; month-interval local satellite fusion; timestamp-shuffled negative controls. The interval operator uses **month/year, never an invented day**.
- **E2 / source error predictability**: spatially cross-fitted expert errors, modality-specific error prediction from ex-ante quality and uncertainty features, equal-weight versus confidence-weight versus risk-weight fusion. The risk models are **not yet externally calibrated**.
- **E3 / satellite-only deployment**: expert-only hard labels, hard VLM pseudo-labels, and group-out-of-fold multimodal soft-teacher targets; all students use satellite data only at test time.

**Caution:** The method is deliberately small. This is a *diagnostic experiment suite*, not yet a proposed novel architecture or empirical evidence. Synthetic outputs show only that code runs. No real Indian measurements or LUCAS back-test have been made.

The key choice is **not** to build a large transformer first. We first ask whether temporal correspondence and individual modality reliability provide measurable, spatially transferable information, then develop an interval-aware cross-attention / factorized uncertainty method only if justified by the data.
