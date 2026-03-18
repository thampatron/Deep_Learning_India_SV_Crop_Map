import time
import vertexai
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage
import os
import json
import datetime
import random
import tqdm
import csv
import re  # Import the regular expression library
import threading
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor
import glob  # Added for finding CSV files

# import google.cloud.aiplatform as aiplatform

print(f"Vertex AI SDK version: {vertexai.__version__}")

# Google Cloud Project and Bucket (REPLACE THESE WITH YOUR VALUES)
PROJECT_ID = "street-view-crop-type"
BUCKET_NAME = "india_croptype_streetview"

PROCESSED_IMAGES_FILE = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/trainSet_kharif_8Sept/processed_images.json"  # File to store processed image paths
TRAIN_SET_FOLDER = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/trainSet_kharif_8Sept"  # Path to the trainSet folder
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/laguarta_jordi/sean7391/street-view-crop-type-82f4bc0e078d.json"

vertexai.init(project=PROJECT_ID, location="us-central1")
# aiplatform.init(project=PROJECT_ID, location="us-central1", staging_bucket="india_croptype_streetview")

prompt = (
    "1. Goal:\n"
    "   You are a crop classification expert. Your task is to analyze an input image and determine with high precision the crop type of the first field located at the center of the image, approximately 20 meters in front.\n"
    "\n"
    "2. Return Format:\n"
    "   Here are the classes: [ {1: Legumes}, {2: Rapeseed}, {3: Maize or Sorghum or Millet}, {5: Rice}, {7: Cotton}, {8: Sugarcane}, {9: Fallow}, {10: Multiple crop types with boundary in the middle}, {11: Too hard to tell}, {12: Not cropland}, {13: Other crop}, {14: Too early in growing stage}, {15: Image quality poor}, {16: Ambiguous}, {17: Grass/Pasture Land}]\n"
    "   Answer with the number and crop type, for example: '{5: Rice}'\n"
    "\n"
    "   I WANT YOU TO PLACE A VERY VERY STRONG EMPHASIS ON TRULY CLASSIFYING THE POINT 20-25 METERS AWAY FROM WHERE THE CAMERA IS (since that is where I will will pull the time series from.)"
    "   - When unsure between a specific crop and a catch-all class like 'Not Possible', choose the catch-all class.\n"
    "   Focus on the field's appearance and the plant's shape to make your best guess.\n"
    "   Use the provided class descriptions as a guide, but remember that real-world fields may not perfectly match these descriptions and trust your best guess. Use your judgment and definitely quadruple check the image before answering.\n"
    "   Think that these images are in India. If it is really not clear what the class is then label the image as one of the catch classes.\n"
    "\n"
    "   Classes:\n"
    "   {1: Legumes} - This includes classes like soybean, peanuts, beans, peas, etc in India, since there are many different types the description might now match all. Medium row spacing. Not often a perfect looking field. Round, small leaves. Tangled growth, visible pods. Beans have larger, heart-shaped leaves (mung bean). Peas grow taller. Lentils have small, thin leaves on offshoots. Peanut has small, round leaves close to the ground. Soybean has darker, pointer leaves, wider row spacing.\n"
    "   {2: Rapeseed} - Scruffy light green, medium row width. Broad leaves. Bright yellow flowers, visible pods.\n"
    "   {3: Maize or Sorghum or Millet} - Wide row spacing. Straight growth, leaves fold to the side. Millet has thinnest leaves, then maize, then sorghum. Tall growth. Millet thins out, goes yellow. Maize stays greener, darker brown. Sorghum has wide, floppy leaves, dark later stages.\n"
    "   {5: Rice} - Sometimes flooded fields with flood walls. Thin, vertical dark green leaves. Heads drop near harvest, crops fall over.\n"
    "   {7: Cotton} - Tall early growth, three-pointed leaves. Flowers early, more prevalent later. Leaves lose color, go dark brown.\n"
    "   {8: Sugarcane} - Thin, long leaves, tillers out. Leaves only at the top. Extremely tall, dense growth.\n"
    "   {9: Fallow} - Bare soil that is likely used for agriculture but not at the moment.\n"
    "   {10: Multiple crop types with boundary in the middle} - Clear divide in front of the image. Small, adjacent fields.\n"
    "   {11: Too hard to tell} - Obstructed field, crop type unclear, or amiguous fields.\n"
    "   {12: Not cropland} - Wild vegetation, urban areas, wastelands, building sites.\n"
    "   {13: Other crop} - Identifiable crop not on the list.\n"
    "   {14: Too early in growing stage} - Bare soil, small shoots, crop type unclear.\n"
    "   {15: Image quality poor} - Blurry, overexposed, distorted, obscured by blur. \n"
    "   {16: Ambiguous} - Too amiguous to determine what crop is growing at the field 20m in front. \n"
    "   {17: Grass/Pasture Land} - Grass or unmanaged land for animals."
)

def get_highest_existing_batch_index(train_set_folder):
    """
    Find the highest existing batch index from output CSV files.
    
    Args:
        train_set_folder: Path to the trainSet folder.
        
    Returns:
        The highest existing batch index, or -1 if none exist.
    """
    csv_pattern = os.path.join(train_set_folder, "batch_inference_output_batch_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    highest_index = -1
    for csv_file in csv_files:
        # Extract the index from the filename
        filename = os.path.basename(csv_file)
        match = re.search(r'batch_inference_output_batch_(\d+)\.csv', filename)
        if match:
            index = int(match.group(1))
            highest_index = max(highest_index, index)
    
    return highest_index

def upload_image_get_url_file(image_path, bucket_name, expiration_time=12000):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_name = os.path.basename(image_path)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(image_path)
    url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=expiration_time), method='GET')
    return url, blob


def get_image_paths_from_subfolders(root_folder):
    image_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths


def load_processed_images(filepath):
    try:
        with open(filepath, 'r') as f:
            # Convert paths to filenames if needed (for backward compatibility)
            processed = json.load(f)
            return [get_filename(path) for path in processed]
    except FileNotFoundError:
        return []

def get_filename(filepath):
    """Extract the filename from a full filepath."""
    return os.path.basename(filepath)

def save_processed_images(filepath, processed_images):
    # Make sure we're saving filenames, not full paths
    with open(filepath, 'w') as f:
        json.dump(processed_images, f)


def create_batch_input_jsonl(image_paths, output_jsonl, bucket_name, processed_images, image_path_map):
    """
    Creates the batch input JSONL and populates the image_path_map.

    Args:
        image_paths: List of local image paths.
        output_jsonl: Path to the output JSONL file.
        bucket_name: GCS bucket name.
        processed_images: List of already processed image paths.
        image_path_map: Dictionary to store mapping of GCS URL to original path.
    """
    new_processed_images = []
    with open(output_jsonl, 'w') as f:
        for image_path in image_paths:  # No tqdm here, handled in main
            if image_path not in processed_images:
                gcs_url, _ = upload_image_get_url_file(image_path, bucket_name)
                input_data = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "fileData": {
                                            "mimeType": "image/jpeg",
                                            "fileUri": gcs_url
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
                f.write(json.dumps(input_data) + '\n')
                new_processed_images.append(image_path)
                image_path_map[gcs_url] = image_path  # Store the mapping
    return new_processed_images


def run_batch_prediction(input_jsonl_uri, output_uri_prefix):
    # MODEL_ARTIFACT_URI = "gs://mco-mm/churn"

    # model = aiplatform.Model.upload(
    #     display_name="churn",
    #     artifact_uri=MODEL_ARTIFACT_URI,
    #     serving_container_image_uri=DEPLOY_IMAGE,
    #     sync=True,
    # )

    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-2.5-flash",
        # source_model="projects/613594576412/locations/us-central1/endpoints/5678587230835179520",
        input_dataset=input_jsonl_uri,
        output_uri_prefix=output_uri_prefix,
    )
    # Don't wait for completion here, return the job
    return batch_prediction_job


def download_gcs_file(gcs_uri, local_filepath):
    storage_client = storage.Client()
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_filepath)
    print(f"Downloaded {gcs_uri} to {local_filepath}")


def create_batches(image_paths, batch_size):
    """Creates batches of image paths."""
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batches.append(image_paths[i:i + batch_size])
    return batches


def process_batch(image_batch, output_jsonl, bucket_name, processed_filenames, image_path_map):
    """Processes a single batch of images."""
    new_processed = []
    with open(output_jsonl, 'w') as f:
        for image_path in image_batch:
            # Compare by filename
            if get_filename(image_path) not in processed_filenames:
                gcs_url, _ = upload_image_get_url_file(image_path, bucket_name)
                input_data = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "fileData": {
                                            "mimeType": "image/jpeg",
                                            "fileUri": gcs_url
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
                f.write(json.dumps(input_data) + '\n')
                new_processed.append(get_filename(image_path))  # Store just the filename
                image_path_map[gcs_url] = image_path  # Keep the full path in the map
    return new_processed


def extract_crop_type(response_text):
    """
    Extracts the crop type number from the model's response.
    """
    match = re.search(r'{(\d+):', response_text)
    if match:
        return int(match.group(1))
    return None


def process_and_save_results(output_jsonl_path, output_csv_path, image_path_map):
    """
    Processes the JSONL output, extracts crop types, and saves to CSV.

    Args:
        output_jsonl_path: Path to the downloaded JSONL file.
        output_csv_path: Path to the output CSV file.
        image_path_map: Dictionary mapping GCS URLs to original image paths.
    """
    results = []
    with open(output_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_url = data['request']['contents'][0]['parts'][1]['fileData']['fileUri']
            original_image_path = image_path_map.get(image_url, image_url)  # Get original path or fallback
            response_text = data['response']['candidates'][0]['content']['parts'][0]['text']
            crop_type = extract_crop_type(response_text)
            if crop_type is not None:
                results.append({'image_path': original_image_path, 'crop_type': crop_type})

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'crop_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Processed and saved results to {output_csv_path}")


def process_batch_and_run_prediction(index, batch, output_gcs_folder, processed_filenames, image_path_map):
    """
    Processes a batch, uploads, runs prediction, and downloads.
    This function is designed to be run in a separate thread.
    """
    output_jsonl_local = f'{TRAIN_SET_FOLDER}/batch_inference_input_batch_{index}.jsonl'
    try:
        new_processed = process_batch(batch, output_jsonl_local, BUCKET_NAME, processed_filenames, image_path_map)
        
        if new_processed:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(os.path.basename(output_jsonl_local))
            blob.upload_from_filename(output_jsonl_local)

            input_jsonl_gcs = f"gs://{BUCKET_NAME}/{os.path.basename(output_jsonl_local)}"
            output_gcs_uri = run_batch_prediction(input_jsonl_gcs, output_gcs_folder)

            # Wait for the job to complete
            while not output_gcs_uri.has_ended:
                time.sleep(5)
                output_gcs_uri.refresh()

            if output_gcs_uri.has_succeeded:
                output_uri = output_gcs_uri.output_location
                output_jsonl_download_path = f'{TRAIN_SET_FOLDER}/batch_inference_output_batch_{index}.jsonl'
                download_gcs_file(output_uri + "/predictions.jsonl", output_jsonl_download_path)
                output_csv_path = f'{TRAIN_SET_FOLDER}/batch_inference_output_batch_{index}.csv'
                process_and_save_results(output_jsonl_download_path, output_csv_path, image_path_map)
            else:
                print(f"Batch prediction job failed for batch {index}: {output_gcs_uri.error}")
        else:
            print(f"No new images to process in batch {index}.")

    except Exception as e:
        print(f"Exception in thread for batch {index}: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback


def get_processed_images_from_csvs(train_set_folder):
    """
    Get all already processed image filenames from the CSV files in the trainSet folder.
    
    Args:
        train_set_folder: Path to the trainSet folder.
        
    Returns:
        List of image filenames that have already been processed.
    """
    processed_images = []
    csv_pattern = os.path.join(train_set_folder, "batch_inference_output_batch_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    print(f"Found {len(csv_files)} CSV output files in {train_set_folder}")
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'image_path' in row and row['image_path']:
                        # Store only the filename
                        processed_images.append(get_filename(row['image_path']))
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
    
    return processed_images


def main():
    root_folder = '/home/laguarta_jordi/sean7391/streetview_highres/highrest_trainset_Kharif_2023'
    output_gcs_folder = f"gs://{BUCKET_NAME}/batch_inference_output"
    
    # Load images that have been processed before (from JSON)
    processed_filenames_from_json = load_processed_images(PROCESSED_IMAGES_FILE)
    
    # Load images that have been processed and saved in CSV files
    processed_filenames_from_csvs = get_processed_images_from_csvs(TRAIN_SET_FOLDER)
    
    # Combine both sets of processed images
    processed_filenames = list(set(processed_filenames_from_json + processed_filenames_from_csvs))
    print(f"Total already processed images: {len(processed_filenames)}")
    
    # Get all image paths
    image_paths = get_image_paths_from_subfolders(root_folder)
    total_images = len(image_paths)
    print(f"Total images found: {total_images}")
    
    # Filter out already processed images - compare by filename
    unprocessed_images = [img for img in image_paths if get_filename(img) not in processed_filenames]
    print(f"Images remaining to be processed: {len(unprocessed_images)}")
    
    if len(unprocessed_images) == 0:
        print("No new images to process. Exiting.")
        return
    
    # Shuffle remaining images
    random.shuffle(unprocessed_images)

    batch_size = 300
    batches = create_batches(unprocessed_images, batch_size)
    
    # Find the highest existing batch index
    start_index = get_highest_existing_batch_index(TRAIN_SET_FOLDER) + 1
    print(f"Starting from batch index {start_index}")
    
    all_new_processed = []
    image_path_map = {}
    max_workers = 30
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, batch in enumerate(tqdm.tqdm(batches, desc="Processing Batches")):
            # Use start_index + i as the new index
            index = start_index + i
            future = executor.submit(
                process_batch_and_run_prediction, index, batch, output_gcs_folder,
                processed_filenames, image_path_map
            )
            futures.append(future)

        # Wait for all futures to complete (implicitly done by the 'with' block)
        for future in futures:
            try:
                future.result()  # Get the result (or exception if one occurred)
            except Exception as e:
                print(f"Exception in a thread: {e}")
                import traceback
                traceback.print_exc()

    # Update processed_images.json with all processed filenames
    save_processed_images(PROCESSED_IMAGES_FILE, processed_filenames + all_new_processed)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
