import torch
from torchvision import transforms
from PIL import Image
from google.cloud import storage
import csv
from tqdm import tqdm 
import itertools
from typing import Tuple, List
import torch.nn as nn
import os
import logging
import multiprocessing

logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if torch.cuda.is_available():
    print('CUDA available')
else:
    print('using CPU')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5174, 0.4975, 0.4587], std=[0.2094, 0.2133, 0.2612])
])

model = None

def init_worker():
    global model
    try:
        model_path = 'tinyvit_final_model.pth'
        model = torch.load(model_path, map_location=device)
        model = model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

def process_and_predict(image_path):
    """Process the image and predict the label."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_rectangle = (0, 150, width, height - 150)
            cropped_img = img.crop(crop_rectangle)
            
            input_tensor = transform(cropped_img)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_batch)
                predicted = (output > 0).item()
        
        return 'field' if predicted == 1 else 'not-field'
    except Exception as e:
        logging.error(f"Failed to process and predict {image_path}: {e}")
        return None

def process_blob(blob_info):
    """Download and process a single blob."""
    image_path = None
    try:
        bucket_name, blob_name = blob_info
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if blob.name.endswith(('.png', '.jpg')):
            image_path = f"/tmp/{blob.name.replace('/', '_')}"
            blob.download_to_filename(image_path)
            label = process_and_predict(image_path)
            if label is not None:
                return blob.name, label
    except Exception as e:
        logging.error(f"Failed to process blob {blob_info}: {e}")
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    return None

def load_duplicates(duplicates_file):
    """Load filenames from a CSV file that should be considered duplicates."""
    duplicate_files = set()
    if os.path.exists(duplicates_file):
        try:
            with open(duplicates_file, 'r') as file:
                reader = csv.reader(file)
                # Assume the first row might be a header
                header = next(reader)
                # Find the filename column index
                filename_col = 0  # Default to first column
                for idx, col_name in enumerate(header):
                    if 'file' in col_name.lower() or 'name' in col_name.lower() or 'path' in col_name.lower():
                        filename_col = idx
                        break
                
                # Read filenames from the appropriate column
                for row in reader:
                    if row and len(row) > filename_col:
                        duplicate_files.add(row[filename_col])
                        
            print(f"Loaded {len(duplicate_files)} duplicate filenames from {duplicates_file}")
        except Exception as e:
            logging.error(f"Failed to read duplicates file {duplicates_file}: {e}")
            print(f"Error reading duplicates file: {e}")
    else:
        print(f"Duplicates file {duplicates_file} not found")
    
    return duplicate_files

def collect_processed_images(csv_file_list, duplicates_file=None):
    """Check multiple CSV files and collect all processed image paths."""
    processed_images = set()
    
    # Load from CSV files
    for csv_file in csv_file_list:
        if os.path.exists(csv_file):
            try:
                with open(csv_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    for row in reader:
                        if row and len(row) > 0:  # Make sure row is not empty
                            processed_images.add(row[0])
            except Exception as e:
                logging.error(f"Failed to read CSV file {csv_file}: {e}")
    
    # Add duplicates if provided
    if duplicates_file:
        duplicate_files = load_duplicates(duplicates_file)
        processed_images.update(duplicate_files)
    
    return processed_images

def download_and_predict(bucket_name, subfolder, csv_file_list, duplicates_file=None):
    """Download images from the bucket and predict their labels."""
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=subfolder))
        blob_infos = [(bucket_name, blob.name) for blob in blobs]

        # Get all processed images from all CSV files in the list and the duplicates file
        processed_images = collect_processed_images(csv_file_list, duplicates_file)
        print(f"Found {len(processed_images)} already processed or duplicate images")

        # Ensure the target CSV file (first in the list) exists with header
        target_csv_file = csv_file_list[0]
        os.makedirs(os.path.dirname(target_csv_file), exist_ok=True)
        if not os.path.exists(target_csv_file):
            with open(target_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image_path', 'label'])

        # Filter out already processed images and duplicates
        new_blob_infos = [info for info in blob_infos if info[1] not in processed_images]
        print(f"Number of new samples: {len(new_blob_infos)}")

        if not new_blob_infos:
            print(f"No new images to process in {subfolder}")
            return

        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(initializer=init_worker) as pool:
            results = []
            for result in tqdm(pool.imap(process_blob, new_blob_infos), total=len(new_blob_infos), desc=f"Processing images in {subfolder}"):
                if result is not None:
                    results.append(result)
                    if len(results) % 2000 == 0:
                        with open(target_csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(results)
                        results = []
 
        if results:
            with open(target_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(results)
            print(f"Successfully wrote {len(results)} results to {target_csv_file}")
    except Exception as e:
        logging.error(f"Failed to download and predict for {subfolder}: {e}")
        print(f"Error in download_and_predict: {e}")

def main():
    bucket_name = 'india_croptype_streetview'
    
    # Path to the CSV file containing duplicate filenames to exclude
    duplicates_file = '/home/laguarta_jordi/sean7391/streetview_highres/all_kharif20m/kharif2023_allpoints.csv'
    
    try:
        for i in range(0, 37):
            subfolder = f'Sept29-GSVimages-unlabeled/imagesHead/India10k_{i}/'
            
            # List of CSV files to check, in order of priority
            csv_file_list = [
                f'inferred_final/area_{i}.csv',  # Primary storage location (will be used to save results)
                f'inferred_final1/area_{i}.csv',    # Secondary location to check
                # Add more CSV paths if needed
            ]
            
            download_and_predict(bucket_name, subfolder, csv_file_list, duplicates_file)
    except Exception as e:
        logging.error(f"Failed in main function: {e}")
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
