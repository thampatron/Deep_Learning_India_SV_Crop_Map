# India SV Pipeline

A comprehensive pipeline for downloading, processing, and classifying agricultural field imagery using SV and Vision Language Models (VLMs).

## Overview

This pipeline enables:
- **High-resolution Street View image download** 
- **Stratified sampling** by geographic area and crop season
- **Ground reference preparation** with adjusted field coordinates
- **Zero-shot and fine-tuned crop classification** using multiple VLMs (GPT-4, Claude, Gemini, LLaMA)
- **Model performance evaluation** and comparison

## Project Structure

```
India_SV_Pipeline/
├── streetview_highres/
│   ├── downloadHighResTrainSetParallel.py    # Main parallel downloader
│   ├── stratifiedSampleMatching.py           # Creates stratified test sets
│   ├── prepare_ground_refs.py                # Generates field center coordinates
│   ├── getAllPointsInSeason.py               # Season-based filtering
│   ├── streetview_pano/                      # Street View API module
│   │   ├── __init__.py
│   │   ├── api.py                            # API wrapper
│   │   ├── download.py                       # Panorama download
│   │   └── search.py                         # Panorama search
│   └── vlms/                                 # Vision Language Model scripts
│       ├── zeroShotInference.py              # Multi-model inference
│       ├── batchInferenceGeminiTrainSet.py   # Gemini batch processing
│       ├── performanceMetrics.py             # Model evaluation
│       ├── compareModels.py                  # Model comparison
│       ├── prepare-data-finetune-batch-gemini.py
│       └── prepare-data-finetune-batch-gpt.py
├── config.example.json                       # Configuration template
├── .env.example                              # Environment variables template
├── .gitignore                                # Git ignore rules
└── README.md                                 # This file
```

## Setup

### 1. Prerequisites

- Python 3.8+
- Google Cloud Platform account with Street View API enabled
- (Optional) API keys for OpenAI, Anthropic, Google Gemini, Together AI

### 2. Installation

```bash
# Clone or navigate to the project directory
cd India_SV_Pipeline

# Install required Python packages
pip install pandas numpy pillow opencv-python scikit-learn geopandas shapely scipy tqdm google-cloud-storage anthropic openai together google-generativeai
```

### 3. Configuration

#### Step 1: Copy configuration templates
```bash
cp config.example.json config.json
cp .env.example .env
```

#### Step 2: Edit `config.json`
Update paths to match your local setup:
```json
{
  "paths": {
    "project_root": "/your/path/to/India_SV_Pipeline",
    "output_folder": "/your/output/path",
    ...
  }
}
```

#### Step 3: Edit `.env`
Add your API keys:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-credentials.json
GOOGLE_MAPS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

#### Step 4: Set environment variables
```bash
source .env
# OR
export $(cat .env | xargs)
```

### 4. Google Cloud Setup

1. Create a Google Cloud Project
2. Enable Street View Static API
3. Create a service account and download credentials JSON
4. Create a Cloud Storage bucket for storing images
5. Set the credentials path in `.env`

## Usage

### 1. Download High-Resolution Training Images

Downloads Street View panoramas in parallel with intelligent caching and resume capability.

```bash
cd streetview_highres
python downloadHighResTrainSetParallel.py
```

**Configuration** (edit in script):
- `max_workers`: Number of parallel workers (default: 20)
- `max_images`: Maximum images to download (default: 100000)
- `batch_size`: Processing batch size (default: 50)
- `selected_states`: Filter by specific states (default: all)

**Features**:
- Area-based state mapping
- Duplicate detection and skipping
- Black image filtering
- SIFT-based feature matching for best panorama selection
- Detailed timing statistics and progress tracking
- Automatic resume from interruptions

**Output**:
- Images saved to `highrest_trainset_*/` folders organized by state
- Progress tracked in `trainSetDownloaded.csv`

---

### 2. Create Stratified Test Sets

Generates geographically representative test sets using area-weighted sampling.

```bash
python stratifiedSampleMatching.py
```

**What it does**:
1. Loads inferred crop labels from previous classifications
2. Filters by year and season
3. Matches to random sample points using state area proportions
4. Ensures minimum distance between points (default: 100m)
5. Exports matched points with coordinates

**Output**:
- `stratifiedSampleMatching_Nk_YEAR_SEASON.csv`
- Includes both original and matched random coordinates

---

### 3. Prepare Ground References

Calculates field center coordinates from Street View camera positions.

```bash
python prepare_ground_refs.py
```

**What it does**:
- Extracts camera heading angles from image metadata
- Projects coordinates at multiple distances (15m, 20m, 30m) toward field center
- Filters by crop season
- Removes spatially clustered points (50m threshold)

**Output**:
- CSV files with projected coordinates for each distance
- `total.csv` with combined filtered results

---

### 4. VLM Inference

Run crop classification using Vision Language Models.

```bash
cd vlms

# Zero-shot inference with multiple models
python zeroShotInference.py

# Batch inference with Gemini
python batchInferenceGeminiTrainSet.py
```

**Supported Models**:
- **GPT-4o** (OpenAI) - Best general performance
- **Claude 3.5 Sonnet** (Anthropic) - Strong reasoning
- **Gemini 1.5 Flash** (Google) - Fast, cost-effective
- **LLaMA 3.2** (Together AI) - Open-source option

**Crop Classes**:
```
Kharif Season:
1: Peas/Beans/Lentils
3: Maize/Sorghum/Millet
5: Rice
6: Soybean/Peanut
7: Cotton
8: Sugarcane
9: Fallow
10: Shrubs
11: Field too far away
12: Not cropland
13: Other crop
14: Seedlings
```

**Output**:
- CSV files with predictions per model
- Saved to `labels/` subdirectories

---

### 5. Performance Evaluation

Compare model accuracy and generate metrics.

```bash
cd vlms
python performanceMetrics.py
python compareModels.py
```

**Metrics calculated**:
- Overall accuracy
- Per-class F1 scores
- Confusion matrices
- State-wise performance

**Output**:
- Performance comparison tables
- Visualization plots (F1 scores, distributions)

---

## Pipeline Workflow

```
1. Data Preparation
   └─> getAllPointsInSeason.py (filter by season)

2. Sample Creation
   └─> stratifiedSampleMatching.py (create test set)

3. Image Download
   └─> downloadHighResTrainSetParallel.py (download Street View)

4. Ground References
   └─> prepare_ground_refs.py (calculate field coordinates)

5. Classification
   ├─> zeroShotInference.py (zero-shot VLM)
   └─> batchInferenceGeminiTrainSet.py (batch processing)

6. Evaluation
   ├─> performanceMetrics.py (calculate metrics)
   └─> compareModels.py (model comparison)
```

---

## Key Considerations

### Data Leakage Prevention
- Separate train/test splits by geographic proximity
- Minimum distance constraints between samples
- State-level stratification

### Cost Optimization
- Intelligent caching to avoid re-downloading
- Black image detection to skip invalid data
- Batch processing for API efficiency

### Reproducibility
- Random seed control in sampling
- Comprehensive logging and progress tracking
- Version-controlled configuration files

---

## Troubleshooting

### Issue: "GOOGLE_APPLICATION_CREDENTIALS not set"
**Solution**: Ensure `.env` is loaded and credentials path is correct
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Issue: "No module named 'streetview_pano'"
**Solution**: Ensure you're running from the correct directory
```bash
cd India_Map_Pipeline/streetview_highres
python -c "import streetview_pano"  # Should work
```

### Issue: Download failures / timeout errors
**Solution**:
- Check internet connection
- Verify Google Cloud quotas and billing
- Reduce `max_workers` to avoid rate limits

### Issue: Out of memory during download
**Solution**:
- Reduce `batch_size` in downloadHighResTrainSetParallel.py
- Lower `max_workers`
- Process states individually

---

## Data Requirements

### Input Data (not included)
You will need:
1. **Random sample CSV**: Randomly sampled points across your region
2. **State shapefiles** (`secondLevelIndiaShp.shp`): Second level state boundaries
3. **Inferred crop data**: Previous crop classification results
4. **Metadata CSV**: Field coordinates with season/date information

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## Acknowledgments

- Google Street View Static API
- OpenAI, Anthropic, Google DeepMind for VLM APIs


## MIT License

Copyright (c) 2026 Jordi Laguarta Soler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
