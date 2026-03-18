import os
import pandas as pd
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from google.cloud import storage
from streetview_pano import get_panorama_side
from io import BytesIO
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
import argparse
import torch
from functools import partial
import time

# Add detailed timing dictionary to track performance
TIMING_STATS = defaultdict(list)

RECORD_FILE = '/home/laguarta_jordi/sean7391/streetview_highres/trainSetDownloaded.csv'

# State mapping
AREA_TO_STATE = {
    '0': 'andaman_and_nicobar',
    '1': 'andhra_pradesh',
    '2': 'arunachal_pradesh',
    '3': 'assam',
    '4': 'bihar',
    '5': 'chandigarh',
    '6': 'chhattisgarh', 
    '7': 'dadra_and_nagar_haveli',
    '8': 'daman_and_diu',
    '9': 'goa',
    '10': 'gujarat',
    '11': 'haryana',
    '12': 'himachal_pradesh',
    '13': 'jammu_and_kashmir',
    '14': 'jharkhand',
    '15': 'karnataka',
    '16': 'kerala',
    '17': 'lakshadweep',
    '18': 'madhya_pradesh',
    '19': 'maharashtra',
    '20': 'manipur',
    '21': 'meghalaya',
    '22': 'mizoram',
    '23': 'nagaland',
    '24': 'nct_of_delhi',
    '25': 'odisha',
    '26': 'puducherry',
    '27': 'punjab',
    '28': 'rajasthan',
    '29': 'sikkim',
    '30': 'tamil_nadu',
    '31': 'telangana',
    '32': 'tripura',
    '33': 'uttar_pradesh',
    '34': 'uttarakhand',
    '35': 'west_bengal'
}

STATE_TO_AREA = {v: k for k, v in AREA_TO_STATE.items()}

# ---------------------- Helper Functions ----------------------
def load_downloaded_record():
    """
    Load the record of already downloaded files.
    Returns a set of basenames and the full dataframe.
    """
    start_time = time.time()
    if os.path.exists(RECORD_FILE):
        try:
            # Add error handling for CSV parsing
            df = pd.read_csv(RECORD_FILE, on_bad_lines='skip')
            
            # Extract just the basenames for more reliable comparison
            file_paths = df['image_path'].tolist()
            base_names = [os.path.basename(path) for path in file_paths]
            
            # Add old_image_path basenames if that column exists
            if 'old_image_path' in df.columns:
                old_file_paths = df['old_image_path'].tolist()
                old_base_names = [os.path.basename(path) for path in old_file_paths]
                # Combine both sets of basenames
                all_base_names = set(base_names + old_base_names)
            else:
                all_base_names = set(base_names)
                
            print(f"Loaded {len(df)} records with {len(all_base_names)} unique filenames")
            result = all_base_names, df
        except Exception as e:
            print(f"Error reading {RECORD_FILE}: {e}")
            # Return empty set and dataframe if file can't be read
            result = set(), pd.DataFrame(columns=['image_path', 'area'])
    else:
        result = set(), pd.DataFrame(columns=['image_path', 'area'])
    TIMING_STATS['load_records'].append(time.time() - start_time)
    return result

def update_downloaded_record(processed_df):
    start_time = time.time()
    if os.path.exists(RECORD_FILE):
        processed_df.to_csv(RECORD_FILE, mode='a', header=False, index=False)
    else:
        processed_df.to_csv(RECORD_FILE, index=False)
    TIMING_STATS['update_records'].append(time.time() - start_time)

def get_state_from_area(area):
    if isinstance(area, (int, float)):
        area = str(int(area))
    return AREA_TO_STATE.get(area, 'unknown')

def count_images_by_state(df):
    start_time = time.time()
    if df.empty:
        result = {}
    else:
        if 'area' in df.columns:
            area_counts = df['area'].value_counts().to_dict()
            state_counts = defaultdict(int)
            for area, count in area_counts.items():
                state = get_state_from_area(area)
                state_counts[state] += count
            result = dict(state_counts)
        else:
            result = {}
    TIMING_STATS['count_by_state'].append(time.time() - start_time)
    return result

def is_black_image(image, threshold=0.90):
    if image is None:
        return True
        
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    if len(image_array.shape) == 3:
        image_gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        image_gray = image_array

    dark_pixels = np.sum(image_gray < 10)
    total_pixels = image_gray.size
    return (dark_pixels / total_pixels) > threshold

def download_base_image_from_gcs_with_bucket_name(bucket_name, blob_name):
    """Version of download function that creates its own client, to avoid pickling issues"""
    start_time = time.time()
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Try the original path as provided
        try:
            blob = bucket.blob(blob_name)
            image_bytes = blob.download_as_bytes()
            result = Image.open(BytesIO(image_bytes))
            download_time = time.time() - start_time
            return result, download_time
        except Exception:
            # Original blob not found, let's try alternatives
            pass
        
        # Extract area code from filename to try alternatives
        area_match = re.search(r'area(\d+)\.jpg$', blob_name)
        if not area_match:
            area_match = re.search(r'area(\d+)', blob_name)
        
        # Also extract folder prefix
        folder_match = re.search(r'(India10k_\d+/)', blob_name)
        
        if area_match and folder_match:
            original_area = area_match.group(1)
            folder_prefix = folder_match.group(1)
            
            # Try alternative area codes
            for area_code in AREA_TO_STATE.keys():
                if area_code == original_area:
                    continue
                
                # Create a path with the new area code in the folder name
                alt_folder = f"India10k_{area_code}/"
                alt_path = blob_name.replace(folder_prefix, alt_folder)
                
                try:
                    alt_blob = bucket.blob(alt_path)
                    image_bytes = alt_blob.download_as_bytes()
                    result = Image.open(BytesIO(image_bytes))
                    print(f"Successfully found image in alternative folder: {alt_path}")
                    download_time = time.time() - start_time
                    return result, download_time
                except Exception:
                    # Silently continue to next area code
                    pass
        
        # If we got here, all attempts failed
        print(f"Failed to find file in any area folder: {os.path.basename(blob_name)}")
        return None, 0
                
    except Exception as e:
        print(f"Error in download function: {e}")
        return None, 0

def crop_center_percentage(image, percentage=0.5):
    if image is None:
        return None
        
    width, height = image.size
    crop_width = int(width * percentage)
    crop_height = int(height * percentage)
    startx = width // 2 - crop_width // 2
    starty = height // 2 - crop_height // 2
    return image.crop((startx, starty, startx + crop_width, starty + crop_height))

def save_subimage(image, SaveLoc, meta, area):
    start_time = time.time()
    save_path = os.path.join(SaveLoc, f'India10k_highres_{area}/', meta)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path, "jpeg")
    saved = os.path.exists(save_path)
    TIMING_STATS['save_image'].append(time.time() - start_time)
    return saved

# Note: load_downloaded_record() is defined earlier in the file (lines 66-99)
# The duplicate definition has been removed to avoid confusion

def process_one_image(data, bucket_name, outfolder, existing_files, batch_panoramas_left=None, batch_panoramas_right=None):
    """
    Process a single image, checking if it exists first.
    """
    try:
        process_start = time.time()
        old_filename, filename, panoid, head, area, k = data
        old_base_name = os.path.basename(old_filename)
        base_name = os.path.basename(filename)

        step_times = {
            'check_existing': 0,
            'download_base': 0,
            'get_panoramas': 0,
            'check_black': 0,
            'feature_match': 0,
            'save_image': 0,
            'total': 0
        }

        # Enhanced file existence checking
        check_start = time.time()
        
        # Check against existing_files set using basenames
        if base_name in existing_files or old_base_name in existing_files:
            if k % 100 == 0:  # Limit logging to avoid spam
                print(f"Skipping {base_name} - found in existing_files set")
            return None, {}

        # Check if output file exists physically using the corrected area
        output_path = os.path.join(outfolder, f'India10k_highres_{area}/', base_name)
        if os.path.exists(output_path):
            if k % 100 == 0:  # Limit logging to avoid spam
                print(f"Skipping {base_name} - file already exists at {output_path}")
            return None, {}
        
        step_times['check_existing'] = time.time() - check_start

        # Get base image - ENSURE we use old_filename for downloading
        base_dl_start = time.time()
        base_image, base_dl_time = download_base_image_from_gcs_with_bucket_name(bucket_name, old_filename)
        step_times['download_base'] = base_dl_time
        if base_image is None:
            return None, {}

        # Get panoramas from batched results if available
        pano_start = time.time()
        if batch_panoramas_left and batch_panoramas_right and panoid in batch_panoramas_left and panoid in batch_panoramas_right:
            left_panorama = batch_panoramas_left[panoid]
            right_panorama = batch_panoramas_right[panoid]
        else:
            # Fallback to individual downloads
            left_panorama = get_panorama_side(panoid, 5, 'left', multi_threaded=True)
            right_panorama = get_panorama_side(panoid, 5, 'right', multi_threaded=True)
        step_times['get_panoramas'] = time.time() - pano_start

        # Check for black images
        black_check_start = time.time()
        if (left_panorama is None and right_panorama is None) or (is_black_image(left_panorama) and is_black_image(right_panorama)):
            return None, {}
        step_times['check_black'] = time.time() - black_check_start

        # Skip feature matching if one panorama is None or black
        if left_panorama is None or is_black_image(left_panorama):
            selected_panorama = right_panorama
        elif right_panorama is None or is_black_image(right_panorama):
            selected_panorama = left_panorama
        else:
            # Feature matching
            feature_start = time.time()
            # Crop images before feature matching for efficiency
            cropped_base = crop_center_percentage(base_image, percentage=0.5)
            cropped_left = crop_center_percentage(left_panorama, percentage=0.5)
            cropped_right = crop_center_percentage(right_panorama, percentage=0.5)
            
            # Perform SIFT feature matching
            sift = cv2.SIFT_create()
            
            # Process base and left panorama
            if isinstance(cropped_base, Image.Image):
                cropped_base = np.array(cropped_base)
            if isinstance(cropped_left, Image.Image):
                cropped_left = np.array(cropped_left)
                
            kp1, des1 = sift.detectAndCompute(cropped_base, None)
            kp2, des2 = sift.detectAndCompute(cropped_left, None)
            
            left_matches = 0
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                left_matches = len([m for m, n in matches if m.distance < 0.75 * n.distance])
            
            # Process base and right panorama
            if isinstance(cropped_right, Image.Image):
                cropped_right = np.array(cropped_right)
                
            kp3, des3 = sift.detectAndCompute(cropped_right, None)
            
            right_matches = 0
            if des1 is not None and des3 is not None and len(des1) > 0 and len(des3) > 0:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des3, k=2)
                right_matches = len([m for m, n in matches if m.distance < 0.75 * n.distance])
            
            selected_panorama = left_panorama if left_matches > right_matches else right_panorama
            step_times['feature_match'] = time.time() - feature_start

        if is_black_image(selected_panorama):
            return None, {}

        # Save the image - use base_name (corrected filename) and area (corrected area) for saving
        save_start = time.time()
        save_success = save_subimage(selected_panorama, outfolder, base_name, area)
        step_times['save_image'] = time.time() - save_start
        
        step_times['total'] = time.time() - process_start
        return k if save_success else None, step_times
    except Exception as e:
        print(f"Error in process_one_image for {os.path.basename(old_filename)}: {e}")
        return None, {}

def load_and_extract_panoid_and_head(csv_path):
    """
    Load CSV and extract required fields, with added filter for mismatch=True
    """
    start_time = time.time()
    
    # First, check which columns exist in the CSV file
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns in CSV: {', '.join(df.columns)}")
    
    # Filter for growing
    if 'fallow_label' in df.columns:
        df = df[df['fallow_label'] == 'growing']
        print(f"Rows after filtering for 'growing': {len(df)}")
    
    # Add filter for mismatch=True
    if 'mismatch' in df.columns:
        df = df[df['mismatch'] == True]
        print(f"Rows after filtering for 'mismatch=True': {len(df)}")
    
    # Only filter by got_high_res if the column exists
    if 'got_high_res' in df.columns:
        df = df[df['got_high_res'] != '1']
        print(f"Rows after filtering out existing high-res: {len(df)}")

    # Keep original image_path in old_image_path
    df['old_image_path'] = df['image_path'].copy()
    
    # Sample some filenames to understand their structure
    print("\nSample filenames:")
    for path in df['image_path'].sample(min(5, len(df))).tolist():
        print(f"  {path}")

    # Improved area extraction with multiple patterns and debugging
    if 'area' not in df.columns:
        # Try multiple patterns to catch different variations
        patterns = [
            re.compile(r'&area(\d+)'),          # &area followed by digits
            re.compile(r'area(\d+)\.jpg'),      # area followed by digits and .jpg
            re.compile(r'area(\d+)')            # area followed by digits (anywhere)
        ]
        
        def extract_area(path):
            for pattern in patterns:
                match = pattern.search(path)
                if match:
                    return match.group(1)  # Return STRING, not int(match.group(1))
            return None
            
        df['area'] = df['image_path'].apply(extract_area)
        
        # Check area extraction results
        print(f"\nArea extraction summary:")
        print(f"  Successful extractions: {df['area'].notna().sum()} out of {len(df)}")
        print(f"  Failed extractions: {df['area'].isna().sum()} out of {len(df)}")
        
        # Check state mapping
        df['state'] = df['area'].apply(lambda x: get_state_from_area(x) if pd.notna(x) else 'unknown')
        state_counts = df['state'].value_counts().to_dict()
        print("\nState mapping summary:")
        for state, count in sorted(state_counts.items()):
            print(f"  {state.upper()}: {count} images")
        
        # Check if there are area values that don't map to states
        unmapped_areas = set()
        for area in df['area'].dropna().unique():
            if get_state_from_area(area) == 'unknown':
                unmapped_areas.add(area)
        
        if unmapped_areas:
            print("\nWARNING: Found area values without state mapping:")
            for area in sorted(unmapped_areas):
                print(f"  Area {area} has no state mapping")

    def extract_pano_and_head(image_path):
        pano_id_match = re.search(r'panoid([a-zA-Z0-9_-]+)&', image_path)
        head_match = re.search(r'head([\d\.]+)&', image_path)
        
        # Add better error handling to prevent the unpack error
        pano_id = pano_id_match.group(1) if pano_id_match else None
        head = float(head_match.group(1)) if head_match else 0.0  # Default to 0.0 if not found
        
        return pano_id, head

    # Add error handling for extraction
    pano_heads = []
    error_count = 0
    for path in df['image_path']:
        try:
            result = extract_pano_and_head(path)
            pano_heads.append(result)
        except Exception as e:
            print(f"Error extracting from {path}: {e}")
            pano_heads.append((None, 0.0))  # Default values on error
            error_count += 1
    
    if error_count > 0:
        print(f"WARNING: Failed to extract pano_id and head from {error_count} paths")
    
    df['pano_id'], df['head'] = zip(*pano_heads)
    
    TIMING_STATS['load_csv'].append(time.time() - start_time)
    return df

def batch_download_panoramas(batch_panoids, side='left', multi_threaded=True):
    """
    Download multiple panoramas in a single batch for efficiency
    """
    start_time = time.time()
    results = {}
    
    with ThreadPoolExecutor(max_workers=min(len(batch_panoids), 20)) as executor:
        future_to_panoid = {
            executor.submit(get_panorama_side, panoid, 5, side, multi_threaded): panoid 
            for panoid in batch_panoids if panoid is not None
        }
        
        for future in as_completed(future_to_panoid):
            panoid = future_to_panoid[future]
            try:
                result = future.result()
                results[panoid] = result
            except Exception as e:
                print(f"Error downloading panorama {panoid}: {e}")
                results[panoid] = None
    
    TIMING_STATS[f'download_panoramas_{side}'].append(time.time() - start_time)            
    return results

# ---------------------- Parallel Image Processing ----------------------
def download_base_image_from_gcs_with_bucket_name(bucket_name, blob_name):
    """Version of download function that creates its own client, to avoid pickling issues"""
    start_time = time.time()
    
    # Just to be absolutely sure, verify that we're using old_image_path
    # print(f"Attempting to download using path: {blob_name}")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Try the original path as provided
        try:
            blob = bucket.blob(blob_name)
            image_bytes = blob.download_as_bytes()
            result = Image.open(BytesIO(image_bytes))
            download_time = time.time() - start_time
            return result, download_time
        except Exception as e:
            # Original blob not found, let's extract components and try alternatives
            pass
        
        # Extract area code from filename to try alternatives
        area_match = re.search(r'area(\d+)\.jpg$', blob_name)
        if not area_match:
            area_match = re.search(r'area(\d+)', blob_name)
        
        # Also extract folder prefix
        folder_match = re.search(r'(India10k_\d+/)', blob_name)
        
        if area_match and folder_match:
            original_area = area_match.group(1)
            folder_prefix = folder_match.group(1)
            
            # Try alternative area codes
            for area_code in AREA_TO_STATE.keys():
                if area_code == original_area:
                    continue
                
                # Create a path with the new area code in the folder name
                alt_folder = f"India10k_{area_code}/"
                alt_path = blob_name.replace(folder_prefix, alt_folder)
                
                try:
                    alt_blob = bucket.blob(alt_path)
                    image_bytes = alt_blob.download_as_bytes()
                    result = Image.open(BytesIO(image_bytes))
                    print(f"Successfully found image in alternative folder: {alt_path}")
                    download_time = time.time() - start_time
                    return result, download_time
                except Exception:
                    # Silently continue to next area code
                    pass
        
        # If we got here, all attempts failed
        return None, 0
                
    except Exception as e:
        print(f"Error in download function: {e}")
        return None, 0

# This function now returns detailed timing for each step
def process_one_image(data, bucket_name, outfolder, existing_files, batch_panoramas_left=None, batch_panoramas_right=None):
    """
    Process a single image, checking if it exists first.
    """
    try:
        process_start = time.time()
        old_filename, filename, panoid, head, area, k = data
        old_base_name = os.path.basename(old_filename)
        base_name = os.path.basename(filename)

        step_times = {
            'check_existing': 0,
            'download_base': 0,
            'get_panoramas': 0,
            'check_black': 0,
            'feature_match': 0,
            'save_image': 0,
            'total': 0
        }

        # Enhanced file existence checking
        check_start = time.time()
        
        # Check against existing_files set using basenames
        if base_name in existing_files or old_base_name in existing_files:
            if k % 100 == 0:  # Limit logging to avoid spam
                print(f"Skipping {base_name} - found in existing_files set")
            return None, {}

        # Check if output file exists physically using the corrected area
        output_path = os.path.join(outfolder, f'India10k_highres_{area}/', base_name)
        if os.path.exists(output_path):
            if k % 100 == 0:  # Limit logging to avoid spam
                print(f"Skipping {base_name} - file already exists at {output_path}")
            return None, {}
        
        step_times['check_existing'] = time.time() - check_start

        # Get base image - ENSURE we use old_filename for downloading
        base_dl_start = time.time()
        base_image, base_dl_time = download_base_image_from_gcs_with_bucket_name(bucket_name, old_filename)
        step_times['download_base'] = base_dl_time
        if base_image is None:
            return None, {}

        # Get panoramas from batched results if available
        pano_start = time.time()
        if batch_panoramas_left and batch_panoramas_right and panoid in batch_panoramas_left and panoid in batch_panoramas_right:
            left_panorama = batch_panoramas_left[panoid]
            right_panorama = batch_panoramas_right[panoid]
        else:
            # Fallback to individual downloads
            left_panorama = get_panorama_side(panoid, 5, 'left', multi_threaded=True)
            right_panorama = get_panorama_side(panoid, 5, 'right', multi_threaded=True)
        step_times['get_panoramas'] = time.time() - pano_start

        # Check for black images
        black_check_start = time.time()
        if (left_panorama is None and right_panorama is None) or (is_black_image(left_panorama) and is_black_image(right_panorama)):
            return None, {}
        step_times['check_black'] = time.time() - black_check_start

        # Skip feature matching if one panorama is None or black
        if left_panorama is None or is_black_image(left_panorama):
            selected_panorama = right_panorama
        elif right_panorama is None or is_black_image(right_panorama):
            selected_panorama = left_panorama
        else:
            # Feature matching
            feature_start = time.time()
            # Crop images before feature matching for efficiency
            cropped_base = crop_center_percentage(base_image, percentage=0.5)
            cropped_left = crop_center_percentage(left_panorama, percentage=0.5)
            cropped_right = crop_center_percentage(right_panorama, percentage=0.5)
            
            # Perform SIFT feature matching
            sift = cv2.SIFT_create()
            
            # Process base and left panorama
            if isinstance(cropped_base, Image.Image):
                cropped_base = np.array(cropped_base)
            if isinstance(cropped_left, Image.Image):
                cropped_left = np.array(cropped_left)
                
            kp1, des1 = sift.detectAndCompute(cropped_base, None)
            kp2, des2 = sift.detectAndCompute(cropped_left, None)
            
            left_matches = 0
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                left_matches = len([m for m, n in matches if m.distance < 0.75 * n.distance])
            
            # Process base and right panorama
            if isinstance(cropped_right, Image.Image):
                cropped_right = np.array(cropped_right)
                
            kp3, des3 = sift.detectAndCompute(cropped_right, None)
            
            right_matches = 0
            if des1 is not None and des3 is not None and len(des1) > 0 and len(des3) > 0:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des3, k=2)
                right_matches = len([m for m, n in matches if m.distance < 0.75 * n.distance])
            
            selected_panorama = left_panorama if left_matches > right_matches else right_panorama
            step_times['feature_match'] = time.time() - feature_start

        if is_black_image(selected_panorama):
            return None, {}

        # Save the image - use base_name (corrected filename) and area (corrected area) for saving
        save_start = time.time()
        save_success = save_subimage(selected_panorama, outfolder, base_name, area)
        step_times['save_image'] = time.time() - save_start
        
        step_times['total'] = time.time() - process_start
        return k if save_success else None, step_times
    except Exception as e:
        print(f"Error in process_one_image for {os.path.basename(old_filename)}: {e}")
        return None, {}

# ---------------------- Main Download Function ----------------------
def downloadHighResFromRandom(selected_states=None, max_workers=None, max_images=None, batch_size=50):
    print("Starting download process...")
    overall_start = time.time()
    
    # Determine optimal number of workers based on available CPUs
    if max_workers is None:
        max_workers = min(os.cpu_count() * 2, 40)  # Default to 2x CPU cores, max 40
    
    bucket_name = "india_croptype_streetview"
    outfolder = f'/home/laguarta_jordi/sean7391/streetview_highres/highrest_trainset_Kharif_Rabi_2023/'
    meta_path = '/home/laguarta_jordi/sean7391/streetview_highres/kharif_rabi_2023_allpoints.csv'

    # Load existing records and count by state
    existing_files, existing_df = load_downloaded_record()
    existing_state_counts = count_images_by_state(existing_df)
    
    # Load and filter dataframe
    dataframe = load_and_extract_panoid_and_head(meta_path)
    
    # Filter by selected states if specified
    if selected_states:
        print(f"\nFiltering for selected states: {', '.join(selected_states)}")
        # Convert state names to area codes
        selected_areas = []
        for state in selected_states:
            state_lower = state.lower().replace(' ', '_')
            if state_lower in STATE_TO_AREA:
                selected_areas.append(int(STATE_TO_AREA[state_lower]))
            else:
                print(f"Warning: State '{state}' not found in mapping")
        
        # Filter dataframe to only include selected areas
        if selected_areas:
            original_length = len(dataframe)
            dataframe = dataframe[dataframe['area'].isin(selected_areas)]
            print(f"Filtered from {original_length} to {len(dataframe)} images based on selected states")
    
    # Count images available by state before filtering
    total_available_by_state = count_images_by_state(dataframe)
    
    # Filter out already downloaded - use old_image_path for checking existing files
    dataframe = dataframe[~dataframe['old_image_path'].isin(existing_files)]
    
    # Limit to max_images if specified
    if max_images and len(dataframe) > max_images:
        print(f"\nLimiting to {max_images} images (from {len(dataframe)} available)")
        dataframe = dataframe.sample(max_images, random_state=42)
    
    # Count images to download by state after filtering
    to_download_by_state = count_images_by_state(dataframe)
    
    # Print initial summaries
    print("\n===== IMAGES AVAILABLE BY STATE =====")
    for state, count in sorted(total_available_by_state.items()):
        existing = existing_state_counts.get(state, 0)
        remaining = to_download_by_state.get(state, 0)
        print(f"{state.upper()}: Total {count} | Downloaded {existing} | Remaining {remaining}")
    print("=====================================\n")

    # Get lists for processing - note the order is old_file_names first, then file_names
    old_file_names = dataframe['old_image_path'].tolist()
    file_names = dataframe['image_path'].tolist()
    panoids = dataframe['pano_id'].tolist()
    heads = dataframe['head'].tolist()
    areas = dataframe['area'].tolist()
    indices = list(range(len(file_names)))

    os.makedirs(outfolder, exist_ok=True)
    for area in set(areas):
        os.makedirs(os.path.join(outfolder, f'India10k_highres_{area}/'), exist_ok=True)

    # Setup timing tracking for each step in the process
    timing_per_file = []
    timing_per_batch = []

    processed_df = pd.DataFrame(columns=dataframe.columns)
    state_download_counter = defaultdict(int)
    total_downloaded = 0
    
    # Process in batches for better efficiency
    num_batches = (len(file_names) + batch_size - 1) // batch_size
    
    print(f"Starting to process {len(file_names)} files with {max_workers} workers in {num_batches} batches...")
    
    for batch in tqdm(range(num_batches)):
        batch_start_time = time.time()
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(file_names))
        
        batch_old_file_names = old_file_names[start_idx:end_idx]
        batch_file_names = file_names[start_idx:end_idx]
        batch_panoids = panoids[start_idx:end_idx]
        batch_heads = heads[start_idx:end_idx]
        batch_areas = areas[start_idx:end_idx]
        batch_indices = indices[start_idx:end_idx]
        
        # Get unique panorama IDs in this batch
        unique_panoids = list(set([p for p in batch_panoids if p is not None]))
        
        # Time the panorama downloads
        download_start = time.time()
        batch_panoramas_left = batch_download_panoramas(unique_panoids, side='left')
        batch_panoramas_right = batch_download_panoramas(unique_panoids, side='right')
        batch_download_time = time.time() - download_start
        
        # Create list of arguments for each task - note we pass old_filename first, then filename
        task_args = [
            (batch_old_file_names[i-start_idx], batch_file_names[i-start_idx], batch_panoids[i-start_idx], 
             batch_heads[i-start_idx], batch_areas[i-start_idx], i) 
            for i in batch_indices
        ]
        
        # Process this batch in parallel]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed arguments
            process_func = partial(
                process_one_image, 
                bucket_name=bucket_name,  # Pass bucket name instead of bucket object
                outfolder=outfolder, 
                existing_files=existing_files,
                batch_panoramas_left=batch_panoramas_left,
                batch_panoramas_right=batch_panoramas_right
            )
            
            futures = [executor.submit(process_func, arg) for arg in task_args]
            
            for future in as_completed(futures):
                result_idx, step_times = future.result()
                if result_idx is not None:
                    # Get the original dataframe row for this index
                    result_row = dataframe.iloc[result_idx]
                    processed_df = pd.concat([processed_df, pd.DataFrame([result_row])], ignore_index=True)
                    
                    area = result_row['area']
                    state = get_state_from_area(area)
                    state_download_counter[state] += 1
                    total_downloaded += 1
                    
                    # Add this file's timing to our records
                    if step_times:
                        timing_per_file.append(step_times)
                    
                    # Log progress every 25 images
                    if total_downloaded % 25 == 0:
                        elapsed_time = time.time() - overall_start
                        images_per_second = total_downloaded / elapsed_time
                        eta_seconds = (len(file_names) - total_downloaded) / images_per_second if images_per_second > 0 else 0
                        eta_hours = eta_seconds / 3600
                        
                        # Calculate and display average processing times
                        if timing_per_file:
                            avg_times = {
                                step: sum(t.get(step, 0) for t in timing_per_file) / len(timing_per_file) 
                                for step in ['check_existing', 'download_base', 'get_panoramas', 
                                             'check_black', 'feature_match', 'save_image', 'total']
                            }
                            
                            print(f"\nProgress update - {total_downloaded}/{len(file_names)} images downloaded "
                                  f"({images_per_second:.2f} img/s, ETA: {eta_hours:.1f} hours):")
                            print("\nAverage processing times per file (seconds):")
                            for step, avg_time in avg_times.items():
                                print(f"  {step}: {avg_time:.4f}s")
                            
                            print("\nDownloads by state:")
                            for s, c in sorted(state_download_counter.items()):
                                print(f"  {s.upper()}: {c} images")

            # Update records after each batch for better fault tolerance
            if not processed_df.empty:
                update_downloaded_record(processed_df)
                processed_df = pd.DataFrame(columns=dataframe.columns)
                
        # Record batch timing stats
        batch_time = time.time() - batch_start_time
        batch_stats = {
            'batch_num': batch,
            'batch_size': len(batch_indices),
            'download_time': batch_download_time,
            'total_time': batch_time,
            'time_per_image': batch_time / len(batch_indices) if batch_indices else 0
        }
        timing_per_batch.append(batch_stats)
        
        # Report batch stats
        print(f"\nBatch {batch+1}/{num_batches} completed in {batch_time:.2f}s "
              f"({batch_stats['time_per_image']:.2f}s per image)")
        print(f"Panorama download time: {batch_download_time:.2f}s "
              f"({batch_download_time / len(unique_panoids) if unique_panoids else 0:.2f}s per unique panorama)")

    # Final counts (existing + new)
    final_state_counts = dict(existing_state_counts)
    for state, count in state_download_counter.items():
        final_state_counts[state] = final_state_counts.get(state, 0) + count

    total_time = time.time() - overall_start
    
    # Print download summary with timing information
    print("\n===== DOWNLOAD SUMMARY =====")
    print(f"Total new images downloaded: {total_downloaded}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_downloaded/total_time:.2f} images/second)")
    
    # Print detailed timing statistics
    if timing_per_file:
        print("\n--- Average Processing Times Per Image ---")
        avg_times = {
            step: sum(t.get(step, 0) for t in timing_per_file) / len(timing_per_file) 
            for step in ['check_existing', 'download_base', 'get_panoramas', 
                         'check_black', 'feature_match', 'save_image', 'total']
        }
        for step, avg_time in sorted(avg_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{step}: {avg_time:.4f}s ({avg_time/avg_times['total']*100:.1f}% of total)")
    
    if timing_per_batch:
        print("\n--- Average Times Per Batch ---")
        avg_batch_time = sum(b['total_time'] for b in timing_per_batch) / len(timing_per_batch)
        avg_download_time = sum(b['download_time'] for b in timing_per_batch) / len(timing_per_batch)
        avg_time_per_image = sum(b['time_per_image'] for b in timing_per_batch) / len(timing_per_batch)
        print(f"Average batch time: {avg_batch_time:.2f}s")
        print(f"Average panorama download time: {avg_download_time:.2f}s")
        print(f"Average time per image: {avg_time_per_image:.2f}s")
    
    print("\n--- New Downloads By State ---")
    for state, count in sorted(state_download_counter.items()):
        print(f"{state.upper()}: {count} images")
    
    print("\n--- Total Images Available By State (After Download) ---")
    for state, count in sorted(final_state_counts.items()):
        total = total_available_by_state.get(state, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{state.upper()}: {count}/{total} ({percentage:.1f}%)")
    print("============================")
    
    return total_downloaded, state_download_counter

if __name__ == "__main__":
    # Define states to download directly in the code
    # Comment/uncomment the list below to specify which states to download
    # Set to None to download from all states
    selected_states = []
        # 'uttar_pradesh'?, 'maharashtra', 'karnataka']
    #     'gujarat',
    #     'kerala',
    #     'maharashtra'
    #     # Add more states as needed
    # ]
    
    # Uncomment to download from all states
    # selected_states = None
    
    # Configuration parameters
    max_workers = 20       # Number of parallel workers, None for auto-detection
    max_images = 100000       # Maximum number of images to download, None for no limit
    batch_size = 50        # Batch size for processing
    
    print("Starting main process with TIMING METRICS...")
    print(f"Selected states: {', '.join(selected_states) if selected_states else 'All states'}")
    print(f"Max workers: {max_workers or 'Auto'}, Max images: {max_images or 'No limit'}, Batch size: {batch_size}")
    
    # Print hardware info
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU only")
    print(f"Number of CPU cores: {os.cpu_count()}")
    
    try:
        num_images, state_counts = downloadHighResFromRandom(
            selected_states=selected_states,
            max_workers=max_workers,
            max_images=max_images,
            batch_size=batch_size
        )
        print(f"Process completed. Total images downloaded: {num_images}")
        
        # Print overall timing statistics
        print("\n===== OVERALL TIMING STATISTICS =====")
        for key, times in TIMING_STATS.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"{key}: {len(times)} calls, avg {avg_time:.4f}s, total {total_time:.2f}s")
        print("====================================")
        
    except Exception as e:
        print(f"Process failed with error: {e}")
        import traceback
        traceback.print_exc()