# Quick Start Guide

## First Time Setup (5 minutes)

### 1. Configure Environment
```bash
cd /home/laguarta_jordi/sean7391/India_Map_Pipeline

# Copy templates
cp .env.example .env
cp config.example.json config.json

# Edit .env with your API keys
nano .env
```

### 2. Set Required API Keys
Add to `.env`:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-gcp-credentials.json
GOOGLE_MAPS_API_KEY=your_google_api_key
```

### 3. Load Environment
```bash
export $(cat .env | xargs)
```

---

## Common Tasks

### Download Training Images
```bash
cd streetview_highres
python downloadHighResTrainSetParallel.py
```

### Create Test Set (3k samples, Kharif 2023)
```bash
python stratifiedSampleMatching.py
```
Edit script to configure:
- `total_n`: Sample size (line 636)
- `year_needed`: Filter year (line 638)
- `season_needed`: "Kharif" or "Rabi" (line 639)

### Run VLM Classification
```bash
cd vlms
python zeroShotInference.py
```
Choose model by uncommenting in `__main__` section (lines 505-522):
- `query_chatGPT()` for GPT-4
- `query_claude()` for Claude
- `query_gemini()` for Gemini
- `query_togetherAI()` for LLaMA

### Evaluate Model Performance
```bash
cd vlms
python performanceMetrics.py
```

---

## File Paths to Update

Before running scripts, update these hardcoded paths:

### In `downloadHighResTrainSetParallel.py`:
- Line 21: `RECORD_FILE`
- Line 710: `outfolder`
- Line 711: `meta_path`

### In `stratifiedSampleMatching.py`:
- Line 619: `out_csv`
- Line 620: `random_csv_path`
- Line 622: `inferred_folder`
- Line 623: `shapefile_path`

### In `prepare_ground_refs.py`:
- Line 215: `INPUT_FOLDER`
- Line 216: `OUTPUT_FOLDER`

---

## Troubleshooting

**Import Error**: If you see "No module named 'streetview_pano'":
```bash
# Make sure you're in the streetview_highres directory
cd /home/laguarta_jordi/sean7391/India_Map_Pipeline/streetview_highres
python your_script.py
```

**Credentials Error**:
```bash
# Check environment variable is set
echo $GOOGLE_APPLICATION_CREDENTIALS
# Should print path to your JSON credentials file
```

**Out of Memory**:
Edit script parameters:
- Reduce `max_workers` (lower parallelism)
- Reduce `batch_size` (process fewer at once)
- Process one state at a time using `selected_states`

---

## Next Steps

1. ✅ Set up API keys in `.env`
2. ✅ Update file paths in scripts
3. ✅ Test with small sample (`max_images=10`)
4. ✅ Run full pipeline
5. ✅ Evaluate results
