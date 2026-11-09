# PI review: what question does each experiment answer?

**Paper hypothesis:** A single, month-dated street-view observation provides a partial and noisy measurement of a field's crop identity. The usefulness of that measurement depends on crop stage, view quality, field correspondence and the satellite season trajectory. Explicit temporal grounding and learned reliability should make satellite-only crop learning more accurate, calibrated, and spatially transferable.

This is a **hypothesis**, not an established finding. Merely improving fused accuracy when the street view is available at test time does not answer the paper's question.

## Prerequisites and measurements

- Labels refer to the crop grown in the field during the **same growing season** as the photo; classes: rice, cotton, soybean, sugarcane, maize, millets, groundnuts, not cropland.
- A ground image is available for one **month/year** per location, not an exact day. Treat as a whole-month interval.
- Approximately **90% field associations are correct overall**. Audit the variation across month, location, quality and crop; never treat that 90% as a per-sample oracle.
- Require separate provenance of labels: `expert` or `vlm`. Expert test labels must be independent of training VLM predictions.
- S1/S2 sequence features must retain real dates and missing-observation masks. Pooling a whole season destroys the temporal-anchoring test.
- Fix all feature extraction, crop taxonomy and split rules before evaluating the final test holdout.
- Existing project has ground acquisition/VLM scripts but no standardized exported per-field monthly S1/S2 embeddings. Exporting **aligned real sequences** is an explicit prerequisite, not something these scripts fabricate.

## E0 — Failure-mode audit (before optimizing a model)

**Why:** If VLM confusion is not related to image acquisition month/quality and satellite gaps, an elaborate reliability gate is poorly motivated.

**Test:** On expert-verified examples only, compare the VLM to the expert label by crop, ground month and region. Audit visibility, correctness of image-to-field association (~10% error), satellite clear-observation coverage and field boundary quality.

**Expected insight:** Identify whether VLM errors concentrate in visually fallow/early-growth months; whether clouds and image mismatch create predictable failures; and which classes have no trustworthy data.

**Decision:** If month and quality barely explain failures, prioritize semantic ambiguity, crop taxonomic confusion or correspondence correction instead. *An error-by-month curve alone does not prove temporal causality.*

## E1 — Temporal information ablation

**Why:** Timestamp concatenation and cross-attention are standard. We must isolate information from *correct pairing of an image's month and a satellite trajectory*.

**Models:** satellite time series alone; satellite + photo; satellite + timestamp with **no photo**; satellite + photo + timestamp; local interval-weighted satellite features + photo; same architecture with ground dates independently shuffled in training/test; local features with date removed.

**Controls:** identical predefined satellite sequence features for all models, spatially separated expert test, same model family, seed and train cohort. Shuffled timestamps are a diagnostic negative control, not a perfect causal experiment—month and geographic acquisition biases remain.

**What it tells us:** Whether month carries signal beyond satellite phenology, and whether ground-temporal pairing improves crop discrimination beyond a strong satellite-only baseline.

**Decision:** If explicit time does not help, avoid claiming temporal anchoring as novel; explore time-conditioned representation training only if the failure audit shows stage-specific ambiguities.

## E2 — Is observation reliability actually predictable?

**Why:** A gate that shifts weights is not automatically a calibrated risk estimate. Need error-supervised, *out-of-fold* evidence.

**Method:** Fit satellite-only and ground-only experts to training expert labels. For every reliability training example, generate predictions from a spatial fold model that **did not train on that example**. The risk estimator uses prediction entropy/margin, image quality, satellite coverage and ground month to predict each expert's misclassification indicator. Refit base experts on training expert labels and score calibrated error proxies on held-out expert fields.

**Comparisons:** equal average probabilities; confidence-weighted average; spatial-OOF-risk-weighted average; ground/satellite experts.

**Measurements:** per-expert error detection AUROC (when defined), risk Brier score, predicted versus actual error rate; fused macro-F1/NLL/Brier. The pilot does **not** yet tune risk calibration on an independent validation set; call resulting risks preliminary rather than proven calibrated. Later add validation-only calibration.

**Decision:** If predicted per-source risk does not beat confidence heuristics, a reliability-weighted teacher is not yet justified. This is a possible **negative result** central to a credible proposal.

## E3 — Does additional multimodal supervision benefit the actual deployment target?

**Why:** The final India map will not have street-level observations at every prediction location. Fused teacher accuracy with photo-at-test is not the paper's deployment metric.

**Models:** satellite-only trained from expert labels; satellite-only trained with all hard expert/VLM pseudo-labels; satellite-only trained against a **spatially cross-fitted multimodal teacher's soft labels** (expert labels remain one-hot). At test time ALL three receive satellite only. Teacher is baseline cross-fitted distillation, not yet uncertainty-aware distillation.

**What it tells us:** Whether weak image-derived information can make a better satellite-only student than hard pseudo-label training at equal available training examples.

**Decision:** If soft teacher distillation fails to improve held-out satellite-only metrics, improving teacher-at-inference performance is insufficient; reconsider the learning objective or the value of ground data.

## E4 — Robustness and geographic generalization (follow-up)

**Why:** Random splits are optimistic on nearby agricultural fields. Repeated sampling conditions create spatial shortcuts.

**Plan:** Several predeclared spatial block sizes and seeds; test-only geographic buffers; distance-to-training and agroclimatic-distance curves; leave-district/zone-out; held-out season when available. Report macro-F1, per-class F1, NLL/Brier and uncertainty calibration, not just average accuracy. Confidence intervals should use spatial-block bootstrap rather than treating nearby fields as IID.

**Corruption experiments:** Ground blur/occlusion, label-independent wrong-field reassignment, physically plausible cloud/missing-date degradation, and temporal offsets. For each perturbation test BOTH error detection and downstream prediction, including when *both modalities* are bad. Existing pilot contains only blunt corruptions; a realistic suite awaits real imagery and S1/S2 quality flags.

**Decision:** If improvements vanish with spatial buffers or under geographic shifts, do not frame the method as geographically robust.

## E5 — Later extensions (not prerequisites to first results)

- Historical district agriculture: lagged statistics only, with stale/shuffled history, crop-mix transition analyses, and district holdout to detect shortcuts. Contemporary district crop proportions are forbidden as per-field truth.
- LUCAS Europe: different ground survey geometry; multiple countries and survey campaigns but not generally dense within-season revisits. Harmonize crop taxonomy and point/parcel support before comparing.
- Stronger model: genuine temporal cross-attention, explicit sensor-specific uncertainty, correspondence-aware latent-variable models, calibrated risk, weak-label training and teacher–student distillation.

## Interpretation and statistical reporting

Do not select the winning architecture by the test set. Use the `val` expert holdout for model choice and calibration in a subsequent training stage, and freeze the test set for one final report. The current scripts compare *prespecified* default hyperparameters rather than search them. The CSV/NPZ order must be verified by the data exporter. Never include a private answer label inside an embedding or quality feature.

Pre-register two **primary outcomes** for a full study:
1. Delta macro-F1 (temporally paired vs date-shuffled multimodal model) on buffered expert-only held-out regions.
2. Delta macro-F1 (satellite-only student trained on OOF teacher soft labels vs hard VLM pseudo-label baseline) on the same held-out regions.

Secondary: per-expert error detection AUROC, log loss, multiclass Brier, selective risk, per-crop/month performance and across-distance degradation. Use at least three spatially separated folds for reliability training. Do not claim statistical significance from one random partition or synthetic demo.
