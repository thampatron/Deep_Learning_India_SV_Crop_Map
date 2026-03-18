# Low-Resolution Image Filtering Pipeline

This folder contains a complete pipeline for processing low-resolution Street View images to classify crop fields and generate sampling datasets.

## Overview

The pipeline includes:
1. **Random Point Generation** (`generate_random_points.py`): Creates cropland sample points using Earth Engine
2. **Field Detection** (`infer_notField.py`): Identifies agricultural fields using TinyViT
3. **Fallow Classification** (`infer_fallow.py`): Classifies fields as "growing" or "fallow"
4. **Post-Processing** (`filterFieldand20mandSample.py`): Filters, samples, and projects coordinates

## Complete Pipeline Flow

```
Earth Engine Cropland Data
         ↓
 generate_random_points.py
         ↓
  [Random Cropland Points CSV]
         ↓
Google Cloud Storage Images
         ↓
    infer_notField.py
         ↓
  [field / not-field labels]
         ↓
    infer_fallow.py (processes only "field" images)
         ↓
  [growing / fallow labels]
         ↓
  filterFieldand20mandSample.py
         ↓
  [Stratified sample with 20m projections]
```

---

## Scripts

### 1. `infer_notField.py`

**Purpose**: First-stage filter to identify agricultural fields vs non-field imagery.

**Model**: `tinyvit_final_model.pth` (binary classifier)

**Process**:
1. Downloads images from Google Cloud Storage bucket
2. Crops images (removes top/bottom 150px)
3. Classifies as `field` or `not-field`
4. Saves results to CSV with columns: `[image_path, label]`

**Features**:
- Multiprocessing for parallel inference
- Duplicate detection (skips already processed images)
- Resume capability from interruptions
- Batch writing (saves every 2000 predictions)

**Configuration** (in `main()` function):
```python
bucket_name = 'india_croptype_streetview'
duplicates_file = '/path/to/kharif2023_allpoints.csv'  # Line 190
csv_file_list = [
    'inferred_final/area_{i}.csv',   # Output location
    'inferred_final1/area_{i}.csv'   # Secondary check location
]
```

**Output**: CSV files saved to `inferred_final/area_{0-36}.csv`

---

### 2. `infer_fallow.py`

**Purpose**: Second-stage classifier for crop status (growing vs fallow).

**Model**: `tinyvit_green_model.pth` (binary classifier for vegetation presence)

**Process**:
1. Reads CSV from `infer_notField.py`
2. Downloads only images labeled as `field`
3. Crops images (same 150px crop)
4. Classifies as `growing` or `fallow`
5. Appends `fallow_label` column to CSV

**Features**:
- Processes only field images (skips `not-field`)
- Sequential processing with progress tracking
- Resume from previous runs (skips already classified images)
- Adds `N/A` label for non-field images

**Configuration** (in `main()` function):
```python
csv_file = 'inferred_final/area_{i}.csv'  # Input from infer_notField.py
# Output: inferred_fallow/area_{i}.csv
```

**Output**: CSV files saved to `inferred_fallow/area_{0-36}.csv` with columns:
- `image_path`
- `label` (field/not-field)
- `fallow_label` (growing/fallow/N/A)

---

## Setup

### Prerequisites

```bash
pip install torch torchvision pillow google-cloud-storage tqdm timm
```

### Model Files Required

Place these model files in the same directory as the scripts:
1. `tinyvit_final_model.pth` - Field detection model
2. `tinyvit_green_model.pth` - Fallow classification model
3. `tiny_vit.py` - TinyViT model architecture

### Google Cloud Setup

Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## Usage

### Step 1: Field Detection

```bash
python infer_notField.py
```

This will:
- Process all 37 area folders (India10k_0 through India10k_36)
- Skip duplicates from `duplicates_file`
- Save results to `inferred_final/area_{i}.csv`

**Expected Runtime**: ~2-4 hours per area folder (depends on GPU/CPU)

---

### Step 2: Fallow Classification

```bash
python infer_fallow.py
```

This will:
- Load results from Step 1
- Process only images labeled as `field`
- Save enhanced results to `inferred_fallow/area_{i}.csv`

**Expected Runtime**: ~1-2 hours per area folder (fewer images to process)

---

## Key Configuration

### Paths to Update Before Running

#### In `infer_notField.py`:
```python
# Line 190: Duplicates file (images to skip)
duplicates_file = '/home/laguarta_jordi/sean7391/streetview_highres/all_kharif20m/kharif2023_allpoints.csv'

# Line 194: GCS subfolder pattern
subfolder = f'Sept29-GSVimages-unlabeled/imagesHead/India10k_{i}/'

# Lines 197-199: Output CSV paths
csv_file_list = [
    f'inferred_final/area_{i}.csv',
    f'inferred_final1/area_{i}.csv'
]
```

#### In `infer_fallow.py`:
```python
# Line 60: Fallow model path
fallow_model_path = 'tinyvit_green_model.pth'

# Line 108: GCS bucket name
client.bucket('india_croptype').blob(image_path).download_to_filename(...)

# Line 188: Input CSV path
csv_file = f'inferred_final/area_{i}.csv'

# Line 132: Output CSV path
temp_csv_file = f"inferred_fallow/{os.path.basename(csv_file)}"
```

---

## Image Processing Details

### Image Cropping
Both scripts apply the same crop to remove sky/horizon artifacts:
```python
crop_rectangle = (0, 150, width, height - 150)
```
- Removes top 150px (often sky/clouds)
- Removes bottom 150px (often car hood/road markings)

### Image Transformation
```python
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5174, 0.4975, 0.4587],
        std=[0.2094, 0.2133, 0.2612]
    )
])
```

---

## Output Format

### After `infer_notField.py`:
```csv
image_path,label
Sept29-GSVimages-unlabeled/imagesHead/India10k_0/image1.jpg,field
Sept29-GSVimages-unlabeled/imagesHead/India10k_0/image2.jpg,not-field
```

### After `infer_fallow.py`:
```csv
image_path,label,fallow_label
Sept29-GSVimages-unlabeled/imagesHead/India10k_0/image1.jpg,field,growing
Sept29-GSVimages-unlabeled/imagesHead/India10k_0/image2.jpg,not-field,N/A
Sept29-GSVimages-unlabeled/imagesHead/India10k_0/image3.jpg,field,fallow
```

---

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Ensure model files are in the script directory:
```bash
ls tinyvit_final_model.pth tinyvit_green_model.pth tiny_vit.py
```

### Issue: "Failed to load model"
**Solution**: Check CUDA availability and model compatibility:
```python
import torch
print(torch.cuda.is_available())  # Should print True for GPU
```

### Issue: Out of memory
**Solution**: Reduce batch processing:
- Lower `multiprocessing.Pool()` worker count (default uses all CPUs)
- Process fewer areas at once (modify loop range)

### Issue: GCS authentication error
**Solution**: Verify credentials:
```bash
echo $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth application-default login
```

### Issue: Images already processed but script reprocessing
**Solution**: Check that CSV files are being read correctly:
```python
# Verify existing_image_paths is populated
print(f"Found {len(existing_image_paths)} existing images")
```

---

## Performance Optimization

### GPU Acceleration
- Both scripts automatically detect and use CUDA if available
- Expected speedup: 10-20x faster with GPU vs CPU

### Parallel Processing
- `infer_notField.py` uses multiprocessing pool
- Each worker loads the model independently
- Adjust pool size based on available CPU/GPU memory

### Batch Writing
- Results written every 2000 images to prevent data loss
- Modify in line 171 of `infer_notField.py`:
```python
if len(results) % 2000 == 0:  # Change 2000 to desired batch size
```

---

## Data Requirements

### Input
- Low-resolution Street View images in GCS bucket
- Image format: `.jpg` or `.png`
- Expected size: 640x640 or similar

### Output Size
- ~1-2MB per 10,000 images (CSV files)
- Total for 37 areas: ~50-100MB

---

## Dependencies

### Model Architecture
This pipeline depends on `tiny_vit.py` which must contain:
- `TinyViT` class
- `PatchEmbed`, `Conv2d_BN`, `ConvLayer`, `MBConv`
- `Attention`, `Mlp` modules

**Note**: If `tiny_vit.py` is missing, copy it from the model training repository.

---

## Logging

Both scripts write logs to:
- `process.log` (infer_notField.py)
- `process_fallow.log` (infer_fallow.py)

View logs in real-time:
```bash
tail -f process.log
tail -f process_fallow.log
```

---

## Citation

If you use these models or scripts, please cite the TinyViT paper:

```bibtex
@article{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  journal={arXiv preprint arXiv:2207.10666},
  year={2022}
}
```
