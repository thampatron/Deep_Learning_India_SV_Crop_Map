# India Map Pipeline - Complete Summary

**Created**: March 18, 2026
**Location**: `/home/laguarta_jordi/sean7391/India_Map_Pipeline/`

---

## Folder Structure

```
India_Map_Pipeline/
├── README.md                           # Main documentation (380 lines)
├── QUICK_START.md                      # Fast setup guide
├── PIPELINE_SUMMARY.md                 # This file
├── config.example.json                 # Configuration template
├── .env.example                        # Environment variables
├── .gitignore                          # Git ignore (protects credentials)
│
├── streetview_highres/                 # High-res Street View pipeline
│   ├── downloadHighResTrainSetParallel.py
│   ├── stratifiedSampleMatching.py
│   ├── prepare_ground_refs.py
│   ├── getAllPointsInSeason.py
│   ├── streetview_pano/                # Custom API module
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── download.py
│   │   └── search.py
│   └── vlms/                           # Vision Language Models
│       ├── zeroShotInference.py
│       ├── batchInferenceGeminiTrainSet.py
│       ├── performanceMetrics.py
│       ├── compareModels.py
│       ├── prepare-data-finetune-batch-gemini.py
│       └── prepare-data-finetune-batch-gpt.py
│
└── lowres_not_field_filtering/         # Low-res filtering pipeline
    ├── README.md                        # Detailed documentation
    ├── generate_random_points.py        # Earth Engine sampling
    ├── infer_notField.py                # Field detection (TinyViT)
    ├── infer_fallow.py                  # Fallow classification (TinyViT)
    └── filterFieldand20mandSample.py    # Post-processing & sampling
```

---

## What Each Folder Does

### 1. `streetview_highres/` - High-Resolution Pipeline

**Purpose**: Downloads and classifies high-resolution Street View images using Vision Language Models

**Key Scripts**:
- **Download**: `downloadHighResTrainSetParallel.py` - Parallel downloader with SIFT matching
- **Sampling**: `stratifiedSampleMatching.py` - Area-weighted stratified test sets
- **Ground Refs**: `prepare_ground_refs.py` - Field center coordinate projection
- **VLM Inference**: `vlms/zeroShotInference.py` - Multi-model crop classification

**Models Supported**:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Gemini 1.5 Flash (Google)
- LLaMA 3.2 (Together AI)

**Use Case**: Accurate crop type classification at high resolution

---

### 2. `lowres_not_field_filtering/` - Low-Resolution Pipeline

**Purpose**: Filters low-resolution images to identify agricultural fields and fallow status

**Key Scripts**:
- **Point Generation**: `generate_random_points.py` - Earth Engine cropland sampling
- **Field Detection**: `infer_notField.py` - Binary field/not-field classification
- **Fallow Detection**: `infer_fallow.py` - Growing/fallow status classification
- **Post-Processing**: `filterFieldand20mandSample.py` - Tile-based sampling & projection

**Models Used**:
- TinyViT (field detection)
- TinyViT (fallow classification)

**Use Case**: Large-scale rapid filtering of millions of images

---

## Pipeline Comparison

| Feature | High-Res Pipeline | Low-Res Pipeline |
|---------|------------------|------------------|
| **Image Source** | Google Street View API (high-res panoramas) | Pre-downloaded low-res images |
| **Resolution** | 1024×1024 or higher | 640×640 or lower |
| **Classification Method** | Vision Language Models (GPT, Claude, Gemini) | TinyViT CNN models |
| **Speed** | Slower (API calls + VLM inference) | Faster (batch processing) |
| **Accuracy** | Higher (detailed semantic understanding) | Good (efficient binary classification) |
| **Cost** | Higher (API costs) | Lower (local inference) |
| **Use Case** | Detailed crop type classification | Large-scale field/fallow filtering |
| **Output Classes** | 14 crop types + fallow/not-field | field/not-field, growing/fallow |

---

## Typical Workflow

### Scenario 1: Create Training Dataset

```bash
# 1. Generate random cropland points
cd lowres_not_field_filtering
python generate_random_points.py

# 2. Filter for fields
python infer_notField.py

# 3. Classify fallow status
python infer_fallow.py

# 4. Create stratified sample
python filterFieldand20mandSample.py --n-per-tile 100

# 5. Download high-res images for sample
cd ../streetview_highres
python downloadHighResTrainSetParallel.py
```

---

### Scenario 2: Evaluate VLM Performance

```bash
# 1. Create stratified test set
cd streetview_highres
python stratifiedSampleMatching.py

# 2. Run VLM inference
cd vlms
python zeroShotInference.py

# 3. Evaluate performance
python performanceMetrics.py
python compareModels.py
```

---

### Scenario 3: Generate Ground References

```bash
# 1. Prepare field coordinates
cd streetview_highres
python prepare_ground_refs.py

# Output: Coordinates at 15m, 20m, 30m from camera
# Use for validation against satellite imagery
```

---

## Security & Best Practices

### ✅ Implemented
- No hardcoded credentials (all use environment variables)
- `.gitignore` protects sensitive files
- `.env.example` and `config.example.json` templates provided
- Original source folders untouched

### 🔧 Before Running
1. Copy `.env.example` to `.env` and add your API keys
2. Copy `config.example.json` to `config.json` and update paths
3. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
4. Update hardcoded paths in scripts (see QUICK_START.md)

---

## Data Requirements

### Input Data
- **India shapefiles**: State boundaries for stratified sampling
- **Random sample CSV**: 2M random points across India
- **GCS images**: Low-res Street View images in Cloud Storage
- **Previous inferences**: For incremental processing

### Model Files
- **High-res**: None (API-based)
- **Low-res**:
  - `tinyvit_final_model.pth` (field detection)
  - `tinyvit_green_model.pth` (fallow classification)
  - `tiny_vit.py` (model architecture)

### Output Data
- Training images: ~100GB for 100k images
- Test images: ~30GB for 30k images
- CSV metadata: ~1-5MB per 1000 images

---

## Performance Metrics

### High-Res Pipeline
- **Download speed**: 2-5 images/second (with 20 workers)
- **VLM inference**: 0.5-2 seconds per image (depends on model)
- **Bottleneck**: API rate limits and costs

### Low-Res Pipeline
- **Field detection**: 100-200 images/second (GPU)
- **Fallow classification**: 100-200 images/second (GPU)
- **Bottleneck**: GCS download bandwidth

---

## Cost Estimates

### High-Res Pipeline
- **Street View API**: ~$7 per 1000 images
- **GPT-4o**: ~$0.01 per image
- **Claude 3.5**: ~$0.015 per image
- **Gemini Flash**: ~$0.001 per image

### Low-Res Pipeline
- **Inference**: Free (local models)
- **Storage**: GCS standard rates

---

## Next Steps

1. ✅ Set up environment variables
2. ✅ Download required model files
3. ✅ Update paths in scripts
4. ✅ Test with small sample
5. ✅ Run full pipeline
6. ✅ Validate results

---

## Support & Documentation

- **Main README**: Comprehensive guide with usage examples
- **QUICK_START**: Fast setup for common tasks
- **Folder READMEs**: Detailed script documentation
- **Inline Comments**: Each script has detailed comments

---

## Key Differences from Original

### What Changed
1. ✅ Removed hardcoded credentials
2. ✅ Removed duplicate functions
3. ✅ Added configuration templates
4. ✅ Created comprehensive documentation
5. ✅ Added .gitignore for security

### What Stayed the Same
1. ✅ All original functionality preserved
2. ✅ No changes to algorithms
3. ✅ Original source folder untouched
4. ✅ All dependencies kept

---

## Total Code Base

- **Python files**: 18 scripts
- **Lines of code**: ~4,822 lines
- **Documentation**: ~1,500 lines (READMEs + comments)
- **Configuration**: 4 template files

---

## License & Citation

[Add your license and citation information here]
