import csv
import json
from google.cloud import storage
import os
import datetime
import random



def upload_image_get_url(image_path, bucket_name, expiration_time=3600):
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

def csv_to_json(csv_file, json_file, bucket_name, folder_path):

    train_ratio = 0.2
    # Mapping of class numbers to labels
    class_mapping = {
        0: "Wheat or Barley",
        1: "Peas or Beans or Lentils",
        2: "Rapeseed",
        3: "Maize or Sorghum or Millet",
        4: "Potato",
        5: "Rice",
        6: "Soybean or Peanut",
        7: "Cotton",
        8: "Sugarcane",
        9: "Fallow",
        10: "Multiple crop types with boundary in the middle",
        11: "Too hard to tell",
        12: "Not cropland",
        13: "Other crop",
        14: "Too early in growing stage",
        15: "Image quality poor"

    }

    # List to store JSON entries
    all_data = []

    # Read the CSV file
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Extract necessary information
            filename = row["image_name"]
            class_number = int(row["class_id"])
            label = class_mapping.get(class_number, "Unknown Class")
            
            image_path = os.path.join(folder_path, filename)
            # Generate a URL for the image
            img_url, blob = upload_image_get_url(image_path, bucket_name)
            
            prompt = (
            "Please play the role of a crop classification expert. Which of the classes below best describes "
            "the first field at the center of the input image? Look at the field and the plant shape and "
            "think about what it could be. Don't be lazy, try your best guess to match the actual class. "
            "Here are the classes: [ "
            "{0 : Wheat or Barley} - Field: Dense, uniform carpet, green to golden. Plant: Thin, grass-like stalks with narrow leaves. When mature, 'bearded' (awned) seed heads appear, bending the tops. "
            "{1 : Peas or Beans or Lentils} - Field: May appear patchy, but still shows some row structure. Not as uniform as wheat, but plants grow in clusters or bands. "
            "Plant: Peas/Beans: Tendrils present, with compound leaves (multiple leaflets per stem). Beans tend to be taller, while peas may be bushier or vining. "
            "Lentils: Very low-growing, forming a dense carpet-like mat, almost hugging the soil. Pods (if visible): Small green pods hanging from stems in later growth stages. "
            "{2 : Rapeseed} - Field: Bright yellow flowers when in bloom. Dark green before flowering but still uniform in coverage. "
            "Plant: Broad green leaves, tall thin stems, and four-petaled yellow flowers in the reproductive stage. "
            "{3 : Maize or Sorghum or Millet} - Field: Tall, well-spaced plants in rows. "
            "Plant: Maize: Broad leaves, tassels at the top, and large ears (cobs) visible along stems. "
            "Sorghum: Resembles maize but with thicker seed heads at the top, sometimes reddish. "
            "Millet: Shorter than maize or sorghum, with small, clustered seed heads that vary by type. "
            "{4 : Potato} - Field: Low, bushy plants with visible mounded soil ridges between rows. Often dark green. "
            "Plant: Thick stems, compound leaves, and occasionally small white or purple flowers. "
            "{5 : Rice} - Field: Flooded paddies, appearing watery or mirror-like in early stages. "
            "Plant: Thin, bright green leaves early, then drooping golden seed heads when mature. "
            "{6 : Soybean or Peanut} - Field: Soybeans: Dense leafy canopy with a uniform green look. Peanuts: Lower-growing, more spaced, with soil visible between plants. "
            "Plant: Soybeans: Trifoliate leaves (three leaflets per stem). Peanuts: Four-leaflet leaves and yellow flowers close to the ground. "
            "{7 : Cotton} - Field: Bushy, dark green plants before maturity. White cotton bolls become visible when mature. "
            "Plant: Broad lobed leaves, showy flowers (white/pink/yellow), and green bolls that open into fluffy cotton. "
            "{8 : Sugarcane} - Field: Tall, dense, 'wall-like' plants, often lining roadsides. "
            "Plant: Thick jointed stalks, long sharp leaves, and a woody, segmented cane structure. "
            "{9 : Fallow} - Field: Irregular, unmanaged, with bare soil, weeds, or leftover stubble. No clear planting pattern. "
            "Plant: A mix of wild grasses, weeds, or dried plants. May have patchy green areas or completely dry brown soil. "
            "{10 : Multiple crop types with boundary in the middle} - Field: A clear visible division at the center of the image between two crop fields with different crops growing in each one."
            "{11 : Too hard to tell} - Field: Obstructed view, unclear angle, or poor resolution makes it impossible to determine the crop type. "
            "{12 : Not cropland} - Field: Urban, industrial, forested, or barren land. No visible signs of organized agriculture. "
            "{13 : Other crop} - Field: Clearly cultivated, but does not match any of the listed crop categories. "
            "{14 : Too early in growing stage} - Field: Sparse, very small plants, making it difficult to distinguish between crop types. "
            "{15 : Image quality poor} - Field: The image is blurry, distorted, or obstructed, preventing classification. "
            "]. Answer with the number and crop type, for example '{0 : Wheat or Barley}'"
        )
            # # prompt = "Which of the classes below best describe the first field at the center of the input image? Don't be lazy, try your best guess to match the actual class.\n Here are the classes: [1 : Peas or Beans or Lentils,\n 3 : Maize or Sorghum or Millet,\n 5 : Rice,\n 6 : Soybean or Peanut,\n 7 : Cotton,\n 8 : Sugarcane,\n 9 : Fallow,\n 10 : Shrubs,\n 11 : Field is too far away,\n 12 : Not cropland,\n 13 : Other crop not in this list,\n 14 : Seedlings]. Answer with the number and crop type, for example, '0 : Wheat or Barley'.\n"
            # prompt = (
            #     "Please play the role of a crop classification expert. Which of the classes below best describe "
            #     "the first field at the center of the input image? Look at the field and the plant shape and think "
            #     "about what it could be. Don't be lazy, try your best guess to match the actual class.\n"
            #     "Here are the classes: [\n"
            #     "  {0 : Wheat or Barley},\n"
            #     "  {1 : Peas or Beans or Lentils},\n"
            #     "  {2 : Rapeseed},\n"
            #     "  {3 : Maize or Sorghum or Millet},\n"
            #     "  {4 : Potato},\n"
            #     "  {5 : Rice},\n"
            #     "  {6 : Soybean or Peanut},\n"
            #     "  {7 : Cotton},\n"
            #     "  {8 : Sugarcane},\n"
            #     "  {9 : Fallow},\n"
            #     "  {10 : Multiple crop types with boundary in the middle},\n"
            #     "  {11 : Too hard to tell},\n"
            #     "  {12 : Not cropland},\n"
            #     "  {13 : Other crop},\n"
            #     "  {14 : Too early in growing stage},\n"
            #     "  {15 : Image quality poor},\n"
            #     "]. Answer with the number and crop type for example '{0 : Wheat or Barley}'\n"
            # )
            
            # Create the JSON structure
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Play the role of a crop classification expert"
                    },
                    {
                        "role": "user",
                        "content": [
                        {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{img_url}"},
                            },
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": f"{class_number} : {label}"
                    }
                ]
            }

            # Add to the list
            all_data.append(entry)


    # Shuffle the data
    random.shuffle(all_data)

    # Split into train and validation sets
    train_size = int(len(all_data) * train_ratio)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:2*train_size]

    # Write train and validation data to their respective files
    with open(json_file[:-5]+'_train.jsonl', mode="w") as file:
        for entry in train_data:
            file.write(json.dumps(entry) + "\n")

    with open(json_file[:-5]+'_val.jsonl', mode="w") as file:
        for entry in train_data:
            file.write(json.dumps(entry) + "\n")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/laguarta_jordi/sean7391/street-view-crop-type-82f4bc0e078d.json"

# Specify the input CSV and output JSON file
csv_file_path = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/labels1/cropLabelsBatch1Processed.csv" 
json_file_path = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/finetune-GPT-4o-labels/output.json"  

bucket_name = "india_croptype_streetview"
season = 'kharif'
folder_path = '/home/laguarta_jordi/sean7391/streetview_highres/highrest_testset_Kharif_2023_1k_2401'

# Generate the JSON from the CSV
csv_to_json(csv_file_path, json_file_path, bucket_name, folder_path)