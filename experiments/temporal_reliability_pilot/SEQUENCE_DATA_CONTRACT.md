# Aligned sequence data contract for `run_study.py`

## Files

**`metadata.csv`**: one row per **unique field × growing season**. `field_id` unique within the file. Its row order **must exactly match the NPZ first axis**; reject incomplete or reordered exports before research analysis.

Required columns:
- `field_id`, `lat`, `lon`, `capture_year`, `capture_month`
- `crop_label`: same-season target; expert-verified when `label_source=expert`, otherwise VLM pseudo-label
- `label_source`: exactly `expert` or `vlm`
- `g_0, g_1, ...`: fixed-length ground-image embedding, finite numeric values

Optional:
- `vlm_label`: VLM's prediction for the same image, also on expert-labelled fields, for error audit
- `ground_quality`: ex-ante quality score in [0,1]; no outcome-derived quality
- `image_field_match_audited`: audited 0/1 for subset only, diagnostic (NEVER model input)
- `district_id`, `agroclimatic_zone`, `season_id`, `expert_annotation_source`

**`sequences.npz`** contains exactly:
- `satellite`: float `[N,T,D]`; bands or satellite representations per acquisition/window
- `observed`: boolean `[N,T]`; valid satellite measurement. Missing rows may contain NaN in raw NPZ and become zero-filled *after* masking
- `yearmonth`: int `[N,T]` in `YYYYMM`; physical calendar month for every time bin, including missing positions

Every row uses the same ordered time-bin convention. Do not encode a fictional acquisition day for the ground image. If multiple S1/S2 acquisitions exist within a month, aggregate consistently with documented features; keep separate S1/S2 channels and missingness features when implementing sensor-specific uncertainty.

Only features available at the intended deployment prediction time can appear in the satellite sequence. For post-season crop mapping, the full growing season is allowed; for early-season forecasting, truncate all inputs consistently.

## Labels and leakage

Expert-labelled test/validation rows must be independently labelled for the same growing season. VLM labels can train weak-supervision models but NEVER count as external validation truth. Separate expert and VLM sources explicitly. No target-year census crop proportions, label-derived confidence features, geographic identifier proxies or validation/test-derived normalizations should enter features.

If there are multiple images per field or repeated years, current prototype needs aggregation/grouped split rules before use; do not silently duplicate field records.

## Usage

```bash
pip install -r requirements.txt
python make_synthetic_sequence.py --output-dir /tmp/crop_sequence_demo
python run_study.py --metadata /tmp/crop_sequence_demo/metadata.csv \
  --sequences /tmp/crop_sequence_demo/sequences.npz \
  --output /tmp/crop_sequence_demo/results.json \
  --block-degrees 1.0 --buffer-km 5
python -m unittest discover -s tests -v
```

The demo is synthetic and tests **software behavior only**. First real run requires extracting temporally resolved aligned satellite features from your actual S1/S2 data and audited expert labels.
