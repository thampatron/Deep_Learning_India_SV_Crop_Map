import torch
from torchvision import transforms
from PIL import Image
from google.cloud import storage
import csv
from tqdm import tqdm 
import itertools
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
try:
    from timm.models._builder import build_model_with_cfg
except (ImportError, ModuleNotFoundError):
    from timm.models.helpers import build_model_with_cfg
from tiny_vit import TinyViT, PatchEmbed, Conv2d_BN, ConvLayer, MBConv, DropPath, PatchMerging, BasicLayer, TinyViTBlock
from tiny_vit import Attention, Mlp
import multiprocessing
import pickle
import os
import logging
import multiprocessing

import torch
from torchvision import transforms
from PIL import Image
from google.cloud import storage
import csv
from tqdm import tqdm 
import logging
import os

logging.basicConfig(filename='process_fallow.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if torch.cuda.is_available():
    print('CUDA available')
else:
    print('using CPU')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation for the new model (change if necessary)
fallow_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5174, 0.4975, 0.4587], std=[0.2094, 0.2133, 0.2612])
])

# New model for predicting fallow/growing
fallow_model = None

def init_fallow_model():
    global fallow_model
    try:
        fallow_model_path = 'tinyvit_green_model.pth'  # Path to the new model
        fallow_model = torch.load(fallow_model_path, map_location=device)
        fallow_model = fallow_model.to(device)
        fallow_model.eval()
    except Exception as e:
        logging.error(f"Failed to load fallow model: {e}")

def predict_fallow_label(image_path):
    """Process the image and predict the fallow/growing label."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_rectangle = (0, 150, width, height - 150)  # Same cropping as before
            cropped_img = img.crop(crop_rectangle)

            input_tensor = fallow_transform(cropped_img)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = fallow_model(input_batch)
                predicted = (output > 0).item()

        return 'growing' if predicted == 1 else 'fallow'
    except Exception as e:
        logging.error(f"Failed to process and predict fallow label {image_path}: {e}")
        return None

# def process_fallow_images(csv_file):
    """Process images labeled as 'field' in the CSV file and predict fallow/growing."""
    try:
        client = storage.Client()
        
        temp_csv_file = f"inferred_fallow/{os.path.basename(csv_file)}"
        
        with open(csv_file, 'r') as infile, open(temp_csv_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            headers = next(reader)
            headers.append('fallow_label')
            writer.writerow(headers)

            for row in tqdm(reader, desc="Processing field images"):
                image_path, label = row[0], row[1]

                if label == 'field':
                    # Download and predict only for 'field' images
                    image_local_path = f"/tmp/{image_path.replace('/', '_')}"
                    client.bucket('india_croptype').blob(image_path).download_to_filename(image_local_path)

                    fallow_label = predict_fallow_label(image_local_path)

                    if fallow_label is not None:
                        row.append(fallow_label)
                        writer.writerow(row)

                    os.remove(image_local_path)  # Clean up local image file
                else:
                    row.append('N/A')
                    writer.writerow(row)

        # Replace original CSV file with the updated one
        # os.replace(temp_csv_file, csv_file)

    except Exception as e:
        logging.error(f"Failed to process fallow images: {e}")

def process_fallow_images(csv_file):
    """Process images labeled as 'field' in the CSV file and predict fallow/growing."""
    try:
        client = storage.Client()
        
        temp_csv_file = f"inferred_fallow/{os.path.basename(csv_file)}"

        # Load existing classifications if the temp file already exists
        existing_rows = set()
        if os.path.exists(temp_csv_file):
            with open(temp_csv_file, 'r') as outfile:
                reader = csv.reader(outfile)
                next(reader)  # Skip header
                for row in reader:
                    image_path = row[0]
                    existing_rows.add(image_path)

        new_rows_count = 0  # Counter for new rows

        with open(csv_file, 'r') as infile, open(temp_csv_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            headers = next(reader)
            headers.append('fallow_label')
            writer.writerow(headers)

            for row in tqdm(reader, desc="Processing field images"):
                image_path, label = row[0], row[1]

                if image_path in existing_rows:
                    # Image already classified, skip it
                    continue

                if label == 'field':
                    # Download and predict only for 'field' images
                    image_local_path = f"/tmp/{image_path.replace('/', '_')}"
                    client.bucket('india_croptype').blob(image_path).download_to_filename(image_local_path)

                    fallow_label = predict_fallow_label(image_local_path)

                    if fallow_label is not None:
                        row.append(fallow_label)
                        writer.writerow(row)
                        new_rows_count += 1  # Increment counter for each new row

                    os.remove(image_local_path)  # Clean up local image file
                else:
                    row.append('N/A')
                    writer.writerow(row)

        # Print the number of new rows saved
        print(f"Number of new rows saved in {os.path.basename(csv_file)}: {new_rows_count}")

    except Exception as e:
        logging.error(f"Failed to process fallow images: {e}")


def main():
    init_fallow_model()
    for i in range(0, 37):
        csv_file = f'inferred_final/area_{i}.csv'  # The CSV file from the first script
        process_fallow_images(csv_file)

if __name__ == "__main__":
    main()

# transform = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5174, 0.4975, 0.4587], std=[0.2094, 0.2133, 0.2612])
# ])

# model = None

# def init_worker():
#     global model
#     try:
#         model_path = 'tinyvit_green_model.pth'
#         model = torch.load(model_path, map_location=device)
#         model = model.to(device)
#         model.eval()
#     except Exception as e:
#         logging.error(f"Failed to load model: {e}")

# def process_and_predict(image_path):
#     """Process the image and predict the label."""
#     try:
#         with Image.open(image_path) as img:
#             width, height = img.size
#             crop_rectangle = (0, 150, width, height - 150)
#             cropped_img = img.crop(crop_rectangle)
            
#             input_tensor = transform(cropped_img)
#             input_batch = input_tensor.unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 output = model(input_batch)
#                 predicted = (output > 0).item()
        
#         return 'fallow' if predicted == 1 else 'growing'
#     except Exception as e:
#         # print(f"Failed to process and predict {image_path}: {e}")
#         logging.error(f"Failed to process and predict {image_path}: {e}")
#         return None

# def process_blob(blob_info):
#     """Download and process a single blob."""
#     image_path = None
#     try:
#         bucket_name, blob_name = blob_info
#         client = storage.Client()
#         bucket = client.get_bucket(bucket_name)
#         blob = bucket.blob(blob_name)
        
#         if blob.name.endswith(('.png', '.jpg')):
#             image_path = f"/tmp/{blob.name.replace('/', '_')}"
#             blob.download_to_filename(image_path)
#             label = process_and_predict(image_path)
#             if label is not None:
#                 return blob.name, label
#     except Exception as e:
#         # print(f"Failed to process blob {blob_info}: {e}")
#         logging.error(f"Failed to process blob {blob_info}: {e}")
#     finally:
#         if image_path and os.path.exists(image_path):
#             os.remove(image_path)
#     return None

# def download_and_predict(bucket_name, subfolder, csv_file):
#     """Download images from the bucket and predict their labels."""
#     try:
#         client = storage.Client()
#         bucket = client.get_bucket(bucket_name)
#         blobs = list(bucket.list_blobs(prefix=subfolder))
#         blob_infos = [(bucket_name, blob.name) for blob in blobs]

#         existing_image_paths = set()
#         if os.path.exists(csv_file):
#             with open(csv_file, 'r') as file:
#                 reader = csv.reader(file)
#                 next(reader)  
#                 existing_image_paths = set(row[0] for row in reader)
#         else:
#             with open(csv_file, 'w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(['image_path', 'label'])

#         new_blob_infos = [info for info in blob_infos if info[1] not in existing_image_paths]

#         multiprocessing.set_start_method('spawn', force=True)
#         with multiprocessing.Pool(initializer=init_worker) as pool:
#             results = []
#             for result in tqdm(pool.imap(process_blob, new_blob_infos), total=len(new_blob_infos), desc=f"Processing images in {subfolder}"):
#                 if result is not None:
#                     results.append(result)
#                     if len(results) % 2000 == 0:
#                         with open(csv_file, 'a', newline='') as file:
#                             # print('WRITING TO FILE')
#                             writer = csv.writer(file)
#                             writer.writerows(results)
#                         results = []
 
#         if results:
#             with open(csv_file, 'a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerows(results)
#             print(f"Successfully wrote {len(results)} results to {csv_file}")
#     except Exception as e:
#         # print(f"Failed to download and predict for {subfolder}: {e}")
#         logging.error(f"Failed to download and predict for {subfolder}: {e}")



