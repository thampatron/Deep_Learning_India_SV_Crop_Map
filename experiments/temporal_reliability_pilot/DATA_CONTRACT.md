# Data contract for initial pilot

Each row is one **field × growing season** with one matched ground image, or a fixed aggregation of images. Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `field_id` | string | Stable parcel/field identifier. All repeated observations of a field share ID. |
| `lat`, `lon` | float | Parcel center coordinates in WGS84 decimal degrees. |
| `capture_year` | int | Year of street-view photograph. |
| `capture_month` | int, 1–12 | Ground date, **month resolution only**; never invent a day. |
| `crop_label` | string | Crop grown in the field during the same growing season. |
| `s_0...s_n` | float | Satellite representation, made consistently across the season and fields. |
| `g_0...g_m` | float | Ground image representation; may include VLM vision embeddings. |
| `ground_quality` | optional float 0–1 | Ex-ante visibility/image-quality estimate, never correctness measured from test labels. |

Supported crop vocabulary should be selected explicitly (rice, cotton, soybean, sugarcane, maize, millets, groundnuts, not cropland). The script uses whatever label set occurs in the input; **do not mix expert labels with unverified VLM labels in an evaluation set**.

Prioritize a small independently expert-labelled evaluation set, with geographic/season diversity and image-to-field correspondence audits. Existing estimated correspondence is ~90%, but reliability must be measured by crop, month and region, not assumed uniform.

### Leakage precautions
- Ground embeddings must be derived from images without target labels encoded in them.
- Satellite embeddings must use only the intended agricultural season, with clear time-window documentation.
- Do not put crop_label, VLM target label, same-year census composition, or expert confidence calculated from the answer inside features.
- Keep duplicate imagery, parcel repeats, and derived crop labels on the same partition side.
- Use spatial block and physical exclusion buffer; no random spatially neighboring test samples for main claims.
- If multiple years are available, hold out an entire growing season as a separate test.
- For a true satellite-only deployment evaluation, train and score the student with **no ground embedding at inference**.

### Additional columns for follow-up
`image_field_match_audited`, `field_size_ha`, `sv_visibility`, `s2_clear_fraction`, `s1_coverage`, `agroclimatic_zone`, `district_id`, `expert_label_source`, `growing_season_id`.

The CSV is a *feature interface*; exporting embeddings from the existing pipeline is a separate integration task.
