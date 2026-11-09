# Hypothesis-driven experiment roadmap

## Research setting
- Crops: rice, cotton, soybean, sugarcane, maize, millets, groundnuts, and not cropland.
- Ground: typically one image/location; timestamp is **month + year**, not exact day. Images span many months within a common growing season.
- Approximately 90% of images are associated with the correct field; treat this as a dataset-level estimate, not an oracle per-image reliability score.
- Evaluation: independently expert-verified labels for the **same growing season** as the ground image; keep VLM pseudo-labels out of the gold test set.

## Stage 0: data and confounder audit (must happen first)
1. Compare crop counts by capture month, district, agroclimatic zone, and data source.
2. Assess correspondence with a stratified manual audit by crop, month, region, image quality.
3. For each field, plot S1/S2 missingness, cloud-free acquisition counts and seasonal temporal coverage.
4. Check spatial overlap of field polygons and whether satellite features include mixed neighboring parcels.
5. Record image availability selection bias (roads, collection campaigns, region).
6. Construct spatially held-out, purged test sets **before tuning** model decisions.

**Gate:** if ground-image month strongly predicts class because of geographic sampling alone, temporal methods require stronger controls.

## Stage 1: minimal hypothesis test (implemented in pilot.py)
- Satellite-only, ground-only, concat, concat + month, shuffled month and ground × month interactions.
- Ground-quality interaction tests whether simple quality conditioning is useful.
- Wrong-field and satellite-missing perturbations measure brittleness.
- Report macro-F1, per-class F1, month-stratified errors, geographic-distance performance.
- **Caveat:** seasonal pooled features cannot establish time-series temporal anchoring; these are low-cost screening experiments.

**Gate:** if ground contributes no improvement beyond satellite under spatial validation, audit field alignment and crop-label informativeness before building a transformer.

## Stage 2: temporal anchoring with actual per-date satellite features (future)
- Fix a satellite time-series encoder and compare:
  1. satellite-only seasonal encoder;
  2. ground + satellite late fusion without dates;
  3. ground + explicit capture-month embedding;
  4. ground-conditioned cross-attention over satellite dates;
  5. cross-attention with interval-aware timing bias (ground timestamp represented as an entire month);
  6. timestamp-shuffle and within-region month-shuffle controls;
  7. satellite-only + same time metadata (controls for shortcut).
- Probe challenging visually fallow/early-growth/harvest-month observations using expert annotations.
- Keep full seasonal satellite context: a bare image at capture month does not imply noncropland.

## Stage 3: factorized modality reliability (future)
- Distinguish (a) ground semantic ambiguity (b) spatial correspondence uncertainty and (c) per-date S1/S2 measurement/phenology uncertainty.
- Estimate source predictive risk from **out-of-fold source predictions on expert labels**. Do not use fitted classifier probabilities as self-validating trust.
- Compare confidence-only, calibrated per-expert risks, end-to-end gating, risk-guided fusion, and fixed fusion.
- Evaluate ECE, Brier score, NLL, risk-coverage, calibration by region/month and performance with both modalities corrupted.
- Use controlled blur, cloud masking, random field reassignment, and temporal offsets; test whether estimated risk rises *and* fusion performance recovers.
- Avoid inferring ground-field match correctness solely from ground/satellite agreement; both can agree and be wrong.

## Stage 4: historical census as a **testable** prior (future)
- Use lagged district crop statistics (e.g., t-1, trailing 3-year average, crop trends) available before target-season inference.
- Compare none, fixed prior, learned prior, district-shuffled prior, stale prior, and prior gating under changed crop distributions.
- Evaluate effects by district stability and crop prevalence shift; **never** use same-year district crop proportions as target inputs.
- Test whether geography introduces shortcuts under leave-district/agrozone-out evaluation.

## Stage 5: satellite-only student and LUCAS external replication (future)
- Distill multimodal teacher into satellite-only student, and compare against satellite-only supervised, hard pseudo-label and soft-label baselines.
- Deploy inference without ground images or census features unless specifically declared.
- Harmonize Indian crop taxonomy with LUCAS agricultural classes **before** asserting transfer.
- LUCAS offers multi-country and multi-year survey points with point and landscape photos, **not** necessarily repeated views within one growing season.
- Test within-Europe spatial/country/year holdouts first; any India-to-Europe zero-shot comparison needs compatible classes and matched spatial support.

## Predefined research decision points
- A model whose gate fails corruption tests is not a reliable fusion method.
- A gain observed only with a random split is not a geographic generalization result.
- A gain from historical priors that vanishes on shuffled-district controls is suspect.
- Negative pilot results should influence problem formulation rather than justify adding modules.
