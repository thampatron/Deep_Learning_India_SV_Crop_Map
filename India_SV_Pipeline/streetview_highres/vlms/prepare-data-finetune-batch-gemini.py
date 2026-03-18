import csv
import json
from google.cloud import storage
import os
import datetime
import random
from collections import defaultdict


# prompt = (
#     "Please play the role of a crop classification expert. Which of the classes below best describes "
#     "the first field at the center of the input image? Look at the field and the plant shape and "
#     "think about what it could be. Don't be lazy, try your best guess to match the actual class. Make sure to really think about based on your knowledge of how the different fields would look and triple check your answer."
#     "Here are the classes: [ "
#     "{0: Wheat or Barley} - Field: Dense, uniform carpet, green to golden. Plant: Thin, grass-like stalks with narrow leaves. When mature, 'bearded' (awned) seed heads appear, bending the tops. "
#     "{1: Peas or Beans or Lentils} - Field: May appear patchy, but still shows some row structure. Not as uniform as wheat, but plants grow in clusters or bands. "
#     "Plant: Peas/Beans: Tendrils present, with compound leaves (multiple leaflets per stem). Beans tend to be taller, while peas may be bushier or vining. "
#     "Lentils: Very low-growing, forming a dense carpet-like mat, almost hugging the soil. Pods (if visible): Small green pods hanging from stems in later growth stages. "
#     "{2: Rapeseed} - Field: Bright yellow flowers when in bloom. Dark green before flowering but still uniform in coverage. "
#     "Plant: Broad green leaves, tall thin stems, and four-petaled yellow flowers in the reproductive stage. "
#     "{3: Maize or Sorghum or Millet} - Field: Tall, well-spaced plants in rows. "
#     "Plant: Maize: Broad leaves, tassels at the top, and large ears (cobs) visible along stems. "
#     "Sorghum: Resembles maize but with thicker seed heads at the top, sometimes reddish. "
#     "Millet: Shorter than maize or sorghum, with small, clustered seed heads that vary by type. "
#     "{4: Potato} - Field: Low, bushy plants with visible mounded soil ridges between rows. Often dark green. "
#     "Plant: Thick stems, compound leaves, and occasionally small white or purple flowers. "
#     "{5: Rice} - Field: Flooded paddies, appearing watery or mirror-like in early stages. "
#     "Plant: Thin, bright green leaves early, then drooping golden seed heads when mature. "
#     "{6: Soybean or Peanut} - Field: Soybeans: Dense leafy canopy with a uniform green look. Peanuts: Lower-growing, more spaced, with soil visible between plants. "
#     "Plant: Soybeans: Trifoliate leaves (three leaflets per stem). Peanuts: Four-leaflet leaves and yellow flowers close to the ground. "
#     "{7: Cotton} - Field: Bushy, dark green plants before maturity. White cotton bolls become visible when mature. "
#     "Plant: Broad lobed leaves, showy flowers (white/pink/yellow), and green bolls that open into fluffy cotton. "
#     "{8: Sugarcane} - Field: Tall, dense, 'wall-like' plants, often lining roadsides. "
#     "Plant: Thick jointed stalks, long sharp leaves, and a woody, segmented cane structure. "
#     "{9: Fallow} - Field: Irregular, unmanaged, with bare soil, weeds, or leftover stubble. No clear planting pattern. "
#     "Plant: A mix of wild grasses, weeds, or dried plants. May have patchy green areas or completely dry brown soil. "
#     "{10: Multiple crop types with boundary in the middle} - Field: A clear visible division at the center of the image between two crop fields with different crops growing in each one."
#     # "{11: Too hard to tell} - Field: Obstructed view, unclear angle, or poor resolution makes it impossible to determine the crop type. "
#     "{12: Not cropland} - Field: Shurbs or unmanaged land, urban, forested, or barren land. No visible signs of organized agriculture. "
#     "{13: Other crop} - Field: Clearly cultivated, but does not match any of the listed crop categories. "
#     "{14: Too early in growing stage} - Field: Sparse, very small plants, making it difficult to distinguish between crop types. "
#     "{15: Image quality poor} - Field: The image is blurry, distorted, or obstructed, preventing classification. "
#     "{16 : Grass} - Field: Appears as a uniform or patchy green coverage, with varying densities. Can be short and even like a lawn or taller with mixed textures. "
#     "]. Answer with the number and crop type, for example '{0: Wheat or Barley}'"
# )

# prompt = (
#     "1. Goal:\n"
#     "   You are a crop classification expert. Your task is to analyze an input image and determine with high precision the crop type of the first field located at the center of the image, approximately 20 meters in front.\n"
#     "\n"
#     "2. Return Format:\n"
#     "   Here are the classes: [ {0: Wheat or Barley}, {1: Peas or Beans or Lentils or Soybean or Peanuts}, {2: Rapeseed}, {3: Maize or Sorghum or Millet}, {4: Potato}, {5: Rice}, {7: Cotton}, {8: Sugarcane}, {9: Fallow}, {10: Multiple crop types with boundary in the middle}, {11: Too hard to tell}, {12: Not cropland}, {13: Other crop}, {14: Too early in growing stage}, {15: Image quality poor}, {16: Ambiguous} ]\n"
#     "   Answer with the number and crop type, for example: '{0: Wheat or Barley}'\n"
#     "\n"
#     "   - When unsure between a specific crop and a catch-all class like 'Not Possible', choose the catch-all class.\n"
#      "   Focus on the field's appearance and the plant's shape to make your best guess. Prioritize high precision for crop classes over catch-all classes like 'Not Possible'.\n"
#     "   Use the provided class descriptions as a guide, but remember that real-world fields may not perfectly match these descriptions. Use your judgment and definitely quadruple check the image before answering.\n"
#     "   Consider the image's filename for the date to understand the Kharif growing season stage in India.\n"
#     "\n"
#     "   Classes:\n"
#     "   {0: Wheat or Barley} - Close row spacing, wider leaves than rice. Floppy leaves, multiple tillers. Long single-piece heads.\n"
#     "   {1: Legumes} - Medium row spacing. Not often a perfect looking field. Round, small leaves. Tangled growth, visible pods. Beans have larger, heart-shaped leaves (mung bean). Peas grow taller. Lentils have small, thin leaves on offshoots. Peanut has small, round leaves close to the ground. Soybean has darker, pointer leaves, wider row spacing.\n"
#     "   {2: Rapeseed} - Scruffy light green, medium row width. Broad leaves. Bright yellow flowers, visible pods.\n"
#     "   {3: Maize or Sorghum or Millet} - Wide row spacing. Straight growth, leaves fold to the side. Millet has thinnest leaves, then maize, then sorghum. Tall growth. Millet thins out, goes yellow. Maize stays greener, darker brown. Sorghum has wide, floppy leaves, dark later stages.\n"
#     "   {4: Potato} - Large soil furrows, low-growing plants. Pointy or round leaves. Dark color throughout.\n"
#     "   {5: Rice} - Flooded fields with flood walls. Thin, vertical dark green leaves. Heads drop near harvest, crops fall over.\n"
#     "   {7: Cotton} - Tall early growth, three-pointed leaves. Flowers early, more prevalent later. Leaves lose color, go dark brown.\n"
#     "   {8: Sugarcane} - Thin, long leaves, tillers out. Leaves only at the top. Extremely tall, dense growth.\n"
#     "   {9: Fallow} - Bare soil that is likely used for agriculture but not at the moment.\n"
#     "   {10: Multiple crop types with boundary in the middle} - Clear divide in front of the image. Small, adjacent fields.\n"
#     "   {11: Too hard to tell} - Obstructed field, crop type unclear, or amiguous fields.\n"
#     "   {12: Not cropland} - Wild vegetation, urban areas, wastelands, lakes, building sites.\n"
#     "   {13: Other crop} - Identifiable crop not on the list.\n"
#     "   {14: Too early in growing stage} - Bare soil, small shoots, crop type unclear.\n"
#     "   {15: Image quality poor} - Blurry, overexposed, distorted, obscured by blur. \n"
#     "   {16: Ambiguous} - Too amiguous to determine what crop is growing at the field 20m in front."
# )

# Your Kharif Prompt
# prompt = (
#     "Please play the role of a crop classification expert. Which of the classes below best describes "
#     "the first field at the center of the input image? Look at the field and the plant shape and "
#     "think about what it could be. Don't be lazy, try your best guess to match the actual class. Make sure to really think about based on your knowledge of how the different fields would look and triple check your answer."
#     "Here are the classes: [ "
#     "{0: Wheat or Barley} - Field: Dense, uniform carpet, green to golden. Plant: Thin, grass-like stalks with narrow leaves. When mature, 'bearded' (awned) seed heads appear, bending the tops. "
#     "{1: Peas or Beans or Lentils} - Field: May appear somewhat patchy, but still shows some row structure. Not as uniform as wheat, but plants grow in clusters or bands. "
#     "Plant: Peas/Beans: Tendrils present, with compound leaves (multiple leaflets per stem). Beans tend to be taller and more upright, while peas may be bushier or vining. "
#     "Lentils: Very low-growing, forming a dense, mat-like cover, almost hugging the soil. "
#     "Pods (if visible): Small green pods hanging from stems in later growth stages. "
#     "Important to distinguish: "
#     "- Do not confuse with wheat (denser, uniform coverage). "
#     "- Not rapeseed, which grows taller and later develops bright yellow flowers. "
#     "- Not soybeans (more uniform coverage, larger leaves). "
#     "- Not fallow land (ensure visible planting rows). "
#     "{2: Rapeseed} - Field: Bright yellow flowers when in bloom. Dark green before flowering but still uniform in coverage. "
#     "Plant: Broad green leaves, tall thin stems, and four-petaled yellow flowers in the reproductive stage. "
#     "{3: Maize or Sorghum or Millet} - Field: Tall, well-spaced plants in rows. "
#     "Plant: Maize: Broad leaves, tassels at the top, and large ears (cobs) visible along stems. "
#     "Sorghum: Resembles maize but with thicker seed heads at the top, sometimes reddish. "
#     "Millet: Shorter than maize or sorghum, with small, clustered seed heads that vary by type. "
#     "{4: Potato} - Field: Low, bushy plants with visible mounded soil ridges between rows. Often dark green. "
#     "Plant: Thick stems, compound leaves, and occasionally small white or purple flowers. "
#     "{5: Rice} - Field: Flooded paddies, appearing watery or mirror-like in early stages. "
#     "Plant: Thin, bright green leaves early, then drooping golden seed heads when mature. "
#     "{6: Soybean or Peanut} - Field: Soybeans: Dense leafy canopy with a uniform green look. Peanuts: Lower-growing, more spaced, with soil visible between plants. "
#     "Plant: Soybeans: Trifoliate leaves (three leaflets per stem). Peanuts: Four-leaflet leaves and yellow flowers close to the ground. "
#     "{7: Cotton} - Field: Bushy, dark green plants before maturity. White cotton bolls become visible when mature. "
#     "Plant: Broad lobed leaves, showy flowers (white/pink/yellow), and green bolls that open into fluffy cotton. "
#     "{8: Sugarcane} - Field: Tall, dense, 'wall-like' plants, often lining roadsides. "
#     "Plant: Thick jointed stalks, long sharp leaves, and a woody, segmented cane structure. "
#     "{9: Fallow} - Field: Irregular, unmanaged, with bare soil, weeds, or leftover stubble. No clear planting pattern. "
#     "Plant: A mix of wild grasses, weeds, or dried plants. May have patchy green areas or completely dry brown soil. "
#     "{10: Multiple crop types with boundary in the middle} - Field: A clear visible division at the center of the image between two crop fields with different crops growing in each one."
#     "{12: Not cropland} - Field: Shurbs or unmanaged land, urban, forested, or barren land. No visible signs of organized agriculture. "
#     "{13: Other crop} - Field: Clearly cultivated, but does not match any of the listed crop categories. "
#     "{14: Too early in growing stage} - Field: Planted very recently, only seedlings visible, even an expert couldn't tell. "
#     "{15: Image quality poor} - Field: The image is blurry, distorted, or obstructed, preventing classification. "
#     "{16 : Grass} - Field: Appears as a uniform or patchy green coverage, with varying densities. Can be short and even like a lawn or taller with mixed textures. "
#     "]. Answer with the number and crop type, for example '{0: Wheat or Barley}'"
# )

import os
import time
import csv
import tqdm
import datetime
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models import SafetySetting
from google.cloud import storage
from PIL import Image
import io
import json
import random
from collections import defaultdict

prompt = (
    "1. Goal:\n"
    "   You are a crop classification expert. Your task is to analyze an input image and determine with high precision the crop type of the first field located at the center of the image, approximately 20 meters in front.\n"
    "\n"
    "2. Return Format:\n"
    "   Here are the classes: [ {0: Wheat or Barley}, {1: Legumes}, {2: Rapeseed}, {3: Maize or Sorghum or Millet}, {4: Potato}, {5: Rice}, {7: Cotton}, {8: Sugarcane}, {9: Fallow}, {10: Multiple crop types with boundary in the middle}, {11: Too hard to tell}, {12: Not cropland}, {13: Other crop}, {14: Too early in growing stage}, {15: Image quality poor}, {16: Ambiguous} ]\n"
    "   Answer with the number and crop type, for example: '{0: Wheat or Barley}'\n"
    "\n"
    "   - When unsure between a specific crop and a catch-all class like 'Not Possible', choose the catch-all class.\n"
     "   Focus on the field's appearance and the plant's shape to make your best guess. Prioritize high precision for crop classes over catch-all classes like 'Not Possible'.\n"
    "   Use the provided class descriptions as a guide, but remember that real-world fields may not perfectly match these descriptions. Use your judgment and definitely quadruple check the image before answering.\n"
    "   Consider the image's filename for the date to understand the Kharif growing season stage in India.\n"
    "\n"
    "   Classes:\n"
    "   {0: Wheat or Barley} - Close row spacing, wider leaves than rice. Floppy leaves, multiple tillers. Long single-piece heads.\n"
    "   {1: Legumes} - Medium row spacing. Not often a perfect looking field. Round, small leaves. Tangled growth, visible pods. Beans have larger, heart-shaped leaves (mung bean). Peas grow taller. Lentils have small, thin leaves on offshoots. Peanut has small, round leaves close to the ground. Soybean has darker, pointer leaves, wider row spacing.\n"
    "   {2: Rapeseed} - Scruffy light green, medium row width. Broad leaves. Bright yellow flowers, visible pods.\n"
    "   {3: Maize or Sorghum or Millet} - Wide row spacing. Straight growth, leaves fold to the side. Millet has thinnest leaves, then maize, then sorghum. Tall growth. Millet thins out, goes yellow. Maize stays greener, darker brown. Sorghum has wide, floppy leaves, dark later stages.\n"
    "   {4: Potato} - Large soil furrows, low-growing plants. Pointy or round leaves. Dark color throughout.\n"
    "   {5: Rice} - Flooded fields with flood walls. Thin, vertical dark green leaves. Heads drop near harvest, crops fall over.\n"
    "   {7: Cotton} - Tall early growth, three-pointed leaves. Flowers early, more prevalent later. Leaves lose color, go dark brown.\n"
    "   {8: Sugarcane} - Thin, long leaves, tillers out. Leaves only at the top. Extremely tall, dense growth.\n"
    "   {9: Fallow} - Bare soil that is likely used for agriculture but not at the moment.\n"
    "   {10: Multiple crop types with boundary in the middle} - Clear divide in front of the image. Small, adjacent fields.\n"
    "   {11: Too hard to tell} - Obstructed field, crop type unclear, or amiguous fields.\n"
    "   {12: Not cropland} - Wild vegetation, urban areas, wastelands, building sites.\n"
    "   {13: Other crop} - Identifiable crop not on the list.\n"
    "   {14: Too early in growing stage} - Bare soil, small shoots, crop type unclear.\n"
    "   {15: Image quality poor} - Blurry, overexposed, distorted, obscured by blur. \n"
    "   {16: Ambiguous} - Too amiguous to determine what crop is growing at the field 20m in front."
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/laguarta_jordi/sean7391/street-view-crop-type-82f4bc0e078d.json"

# Google Cloud Project and Bucket (REPLACE THESE WITH YOUR VALUES)
PROJECT_ID = "street-view-crop-type"  # Replace with your Google Cloud project ID
BUCKET_NAME = "india_croptype_streetview"  # Replace with your Google Cloud Storage bucket name

vertexai.init(project=PROJECT_ID, location="us-central1")


def crop_center(image_path, crop_percentage):
    """
    Crops the center portion of an image based on the specified percentage.
    
    Args:
        image_path (str): Path to the image file
        crop_percentage (int): Percentage of the center area to keep (10-100)
    
    Returns:
        bytes: The cropped image as bytes
    """
    # Ensure crop_percentage is between 10 and 100
    crop_percentage = max(10, min(100, crop_percentage))
    
    # Open the image
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size
        
        # Calculate crop dimensions
        crop_width = int(width * (crop_percentage / 100))
        crop_height = int(height * (crop_percentage / 100))
        
        # Calculate crop coordinates (center crop)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = (width + crop_width) // 2
        bottom = (height + crop_height) // 2
        
        # Crop image
        cropped_img = img.crop((left, top, right, bottom))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()


def upload_image_get_url(image_data, bucket_name, blob_name, expiration_time=3600):
    """Uploads an image to GCS and returns a signed URL."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Check if it's a file path or bytes
    if isinstance(image_data, str):
        blob.upload_from_filename(image_data)
    else:
        blob.upload_from_string(image_data, content_type='image/jpeg')

    url = blob.generate_signed_url(
        expiration=datetime.timedelta(seconds=expiration_time),
        method="GET"
    )
    return url

def upload_image_get_url_file(image_path, bucket_name, expiration_time=12000):
    """
    Uploads an image to Google Cloud Storage and returns a signed URL for public access.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_name = os.path.basename(image_path)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(image_path)

    url = blob.generate_signed_url(
        expiration=datetime.timedelta(seconds=expiration_time),
        method='GET'
    )
    return url, blob

def find_image_path(image_name, folder_paths):
    """
    Searches for the image in multiple folders and returns the first found path.
    """
    for folder_path in folder_paths:
        image_path = os.path.join(folder_path, image_name)
        if os.path.exists(image_path):
            return image_path
    return None

def csv_to_gemini(csv_file, output_jsonl, bucket_name, folder_paths, train_ratio=0.1, crop_percentage=100):
    """
    Reads a CSV file with image and label information, uploads each image to GCS, 
    and writes data in Gemini-style JSON lines for training and validation using stratified sampling.
    """
    class_mapping = {
        0: "Wheat or Barley",
        1: "Legumes",
        2: "Rapeseed",
        3: "Maize or Sorghum or Millet",
        4: "Potato",
        5: "Rice",
        7: "Cotton",
        8: "Sugarcane",
        9: "Fallow",
        10: "Multiple crop types with boundary in the middle",
        11: "Too hard to tell",
        12: "Not cropland",
        13: "Other crop",
        14: "Too early in growing stage",
        15: "Image quality poor",
        16: "Ambiguous"
    }

    # Dictionary to store data by class
    data_by_class = defaultdict(list)

    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row["image_name"]
            class_number = int(row["class_id"])
            label = class_mapping.get(class_number, "Unknown Class")

            local_image_path = find_image_path(image_name, folder_paths)
            if not local_image_path:
                continue  # Skip if image is not found
            
            #crop the image
            if crop_percentage < 100:
                cropped_image = crop_center(local_image_path, crop_percentage)
                cropped_blob_name = f"cropped_{crop_percentage}_{os.path.basename(local_image_path)}"
                gcs_url = upload_image_get_url(cropped_image, bucket_name, cropped_blob_name)
            else:
                gcs_url, _ = upload_image_get_url_file(local_image_path, bucket_name)

            gemini_format = {
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
                    },
                    {
                        "role": "model",
                        "parts": [
                            {"text": f"{class_number} : {label}"}
                        ]
                    }
                ]
            }
            data_by_class[class_number].append(gemini_format)

    train_data, val_data = [], []
    
    for class_number, items in data_by_class.items():
        random.shuffle(items)
        train_size = int(len(items) * train_ratio)
        train_data.extend(items[:train_size])
        val_data.extend(items[train_size:])

    random.shuffle(train_data)
    random.shuffle(val_data)

    train_output = os.path.splitext(output_jsonl)[0] + "_train.jsonl"
    with open(train_output, mode="w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    val_output = os.path.splitext(output_jsonl)[0] + "_val.jsonl"
    with open(val_output, mode="w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

def main():
    csv_file_path = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/labels1/cropLabelsBatchAllProcessed.csv" 
    output_jsonl_path = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatchAll/finetune-gemini-labels75/output.json"  
    bucket_name = "india_croptype_streetview"
    folder_paths = [
        '/home/laguarta_jordi/sean7391/streetview_highres/highrest_testset_Kharif_2023_1k_2401',
        '/home/laguarta_jordi/sean7391/streetview_highres/highrest_testset_Kharif_2023_2k_1902'
    ]
    crop_percentage = 75 # Change this to 25 to crop to 25% of the center.
    
    csv_to_gemini(
        csv_file=csv_file_path,
        output_jsonl=output_jsonl_path,
        bucket_name=bucket_name,
        folder_paths=folder_paths,
        train_ratio=0.2,  # 20% for training, 80% for validation.
        crop_percentage=crop_percentage
    )

if __name__ == "__main__":
    main()