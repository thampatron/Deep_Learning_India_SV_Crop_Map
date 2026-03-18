import os
import asyncio
import datetime
import csv
import base64
from google.cloud import storage
from together import AsyncTogether, Together
from openai import OpenAI
import openai
import google.generativeai as genai
import requests
import anthropic


# Initialize the prompt

kharif_prompt = ("Please play the role of a crop classification expert. Which of the classes below best describe the first field at the center of the input image. Look at the field and the plant shape and think about what it could be. Don't be lazy, try your best guess to match the actual class. "
                "Here are the classes: [ "
                "1 : Peas or Beans or Lentils, "
                "3 : Maize or Sorghum or Millet, "
                "5 : Rice, "
                "6 : Soybean or Peanut, "
                "7 : Cotton, "
                "8 : Sugarcane, "
                "9 : Fallow, "
                "10 : Shrubs, "
                "11 : Field is too far away, "
            "12 : Not cropland, "
            "13 : Other crop not in this list, "
            "14 : Seedlings, "
            "]. Answer with the number and crop type for example '{0 : Wheat or Barley}'" 
)

rabi_prompt = (
        "Please play the role of a crop classification expert. Which of the classes below best describe the first field at the center of the input image. Look at the field and the plant shape and think about what it could be. Don't be lazy, try your best guess to match the actual class. "
            "Here are the classes: [ "
            "0 : Wheat or Barley, "
            "1 : Peas or Beans or Lentils, "
            "2 : Rapeseed, "
            "3 : Maize or Sorghum or Millet, "
            "4 : Potato, "
            "5 : Rice, "
            "6 : Soybean or Peanut, "
            "8 : Sugarcane, "
            "9 : Fallow, "
            "10 : Shrubs, "
            "11 : Field is too far away for even an expert human to tell, "
            "12 : Not cropland, "
            "13 : Other crop not in this list, "
            "14 : Seedlings, "
            "], and following is the image. Answer with the number and crop type for example '0 : Wheat or Barley'" 
)

finetune_prompt = {"role": "system", "content": "Play the role of a crop classification expert"}, {"role": "user", "content": "Which of the classes below best describe the first field at the center of the input image? Don't be lazy, try your best guess to match the actual class.\nHere are the classes: [\n1 : Peas or Beans or Lentils,\n3 : Maize or Sorghum or Millet,\n5 : Rice,\n6 : Soybean or Peanut,\n7 : Cotton,\n8 : Sugarcane,\n9 : Fallow,\n10 : Shrubs,\n11 : Field is too far away,\n12 : Not cropland,\n13 : Other crop not in this list,\n14 : Seedlings,\n]. Answer with the number and crop type, for example, '0 : Wheat or Barley'.\n"}
# kharif_prompt = (
#     "What is the crop type of the field at the center of this image? These are your choices: "
#     "(1) Peas or Beans or Lentils, "
#     "(3) Maize or Sorghum or Millet, (5) Rice, (6) Soybean or Peanut, "
#     "(7) Cotton, (8) Sugarcane, (9) Fallow, "
#     "(10) Shrubs, (11) Not possible to label such as: multiple crop types with boundary in the middle or too hard to tell or too early in growing stage, "
#     "(12) Not cropland, (13) Other crop not in this list."
#     "Answer in the form of the number in brackets eg. (11) if it is 'Not possible to label' or (9) for 'Fallow'."
# )

# rabi_prompt = (
#     "What is the crop type of the field at the center of this image? These are your choices: "
#     "(0) Wheat or Barley, (1) Peas or Beans or Lentils, "
#     "(2) Rapeseed, (3) Maize or Sorghum or Millet, (4) Potato, (5) Rice,"
#     "(8) Sugarcane, (9) Fallow, "
#     "(10) Shrubs, (11) Not possible to label such as: multiple crop types with boundary in the middle or too hard to tell or too early in growing stage, "
#     "(12) Not cropland, (13) Other crop not in this list."
#     "Answer in the form of the number in brackets eg. (11) if it is 'Not possible to label' or (9) for 'Fallow'."
# )

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

def encode_image_base64(image_path):
    """
    Encodes an image file to base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def query_claude(folder_path, bucket_name, csv_output_path, season):
    """
    Processes all images in a folder, queries the Claude API for each image, and saves the results to a CSV file.
    """
    prompt = kharif_prompt if season == 'kharif' else rabi_prompt
    results = []

    # Initialize the Anthropic Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found. Set it as an environment variable: ANTHROPIC_API_KEY")

    client = anthropic.Client(api_key=api_key)

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        print(f"Processing image: {image_path}")
        image_url, blob = upload_image_get_url(image_path, bucket_name)

        try:
            # Encode the image in base64
            image_data_base64 = encode_image_base64(image_path)

            # Prepare the message for Claude
            message_payload = {
                "model": "claude-3-5-sonnet-20241022",  # Replace with the appropriate Claude model
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",  # Update if your images are not JPEG
                                    "data": image_data_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            }

            # Query Claude API
            response = client.messages.create(**message_payload)

            # Extract the result text and determine the result number
            response_text = response.content[0].text
            print(f"Response from Claude: {response_text}")

            result_number = None
            if "{" in response_text and "}" in response_text:
                try:
                    # Extract content between curly braces
                    result_text = response_text[response_text.find("{")+1:response_text.find("}")]
                    # Get number before the colon
                    result_number = result_text.split(":")[0].strip()
                except:
                    pass

            print(f"Result for {image_path}: {result_number}")
            results.append((image_path, result_number))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        finally:
            # Delete the GCloud image link
            blob.delete()
            print(f"Deleted GCloud link for image: {image_path}")

    # Save results to CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    with open(csv_output_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Original Image Path", "Result Number"])
        csv_writer.writerows(results)

    print(f"Results saved to {csv_output_path}")

def query_gemini(folder_path, bucket_name, csv_output_path, season):
    """
    Processes all images in a folder, queries the Gemini API for each image, and saves the results to a CSV file.
    """
    prompt = kharif_prompt if season == 'kharif' else rabi_prompt
    results = []

    # Initialize the Gemini Generative Model
    model = genai.GenerativeModel("gemini-1.5-flash")

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        print(f"Processing image: {image_path}")
        image_url, blob = upload_image_get_url(image_path, bucket_name)

        try:
            # Upload the image file to Gemini
            uploaded_file = genai.upload_file(image_path)

            # Generate content using the uploaded image and prompt
            result = model.generate_content(
                [uploaded_file, "\n\n", prompt]
            )

            # Extract the result text and determine the result number
            response_text = result.text
            print(f"Response from Gemini: {response_text}")

            result_number = None
            if "(" in response_text and ")" in response_text:
                result_number = response_text.split("(")[-1].split(")")[0]

            print(f"Result for {image_path}: {result_number}")
            results.append((image_path, result_number))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        finally:
            # Delete the GCloud image link
            blob.delete()
            print(f"Deleted GCloud link for image: {image_path}")

    # Save results to CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    with open(csv_output_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Original Image Path", "Result Number"])
        csv_writer.writerows(results)

    print(f"Results saved to {csv_output_path}")

def query_chatGPT(folder_path, bucket_name, csv_output_path, season, finetuned=False):
    """
    Processes all images in a folder, queries the ChatGPT API for each image, and saves the results to a CSV file.
    """

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )
    print('Jobs ', client.fine_tuning.jobs.list(limit=10))

    prompt = kharif_prompt if season == 'kharif' else rabi_prompt
    results = []

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        print(f"Processing image: {image_path}")
        image_url, blob = upload_image_get_url(image_path, bucket_name)

        try:
            if finetuned:
                model_name = "ft:gpt-4o-2024-08-06:personal:try1:AbbGuBp4:ckpt-step-44"
                model_name = "ft:gpt-4o-2024-08-06:personal:longprompt:AzW4qys8"
            else:
                model_name = "gpt-4o"
            # ChatGPT API call
            response_data = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_url}"},
                            },
                        ],
                    }
                ],
            )

            # Extract the response content
            # print('Response data: ', response_data)
            response = response_data.choices[0].message.content
            print('Response', response)

            # Extract the result number in brackets (if available)
            result_number = None
            if ":" in response:
                try:
                    # Split and strip the response to isolate the number
                    result_number = response.split(":")[0].strip()
                    # Ensure the result_number is numeric
                    if not result_number.isdigit():
                        result_number = None
                except Exception as e:
                    print(f"Error parsing result number: {e}")

            # print(f"Result for {image_path}: {result_number}")
            results.append((image_path, result_number))

        finally:
            # Delete the GCloud image link
            blob.delete()
            print(f"Deleted GCloud link for image: {image_path}")


    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)


    # Save results to CSV
    with open(csv_output_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Original Image Path", "Result Number"])
        csv_writer.writerows(results)

    print(f"Results saved to {csv_output_path}")


def query_chatGPT(folder_path, bucket_name, csv_output_path, season, finetuned=False):
    """
    Processes all images in a folder, queries the ChatGPT API for each image, and saves the results to a CSV file.
    """

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )
    print('Jobs ', client.fine_tuning.jobs.list(limit=10))

    prompt = kharif_prompt if season == 'kharif' else rabi_prompt
    results = []

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        print(f"Processing image: {image_path}")
        image_url, blob = upload_image_get_url(image_path, bucket_name)

        try:
            if finetuned:
                model_name = "ft:gpt-4o-2024-08-06:personal:try1:AbbGuBp4:ckpt-step-44"
                model_name = "ft:gpt-4o-2024-08-06:personal:try1:AbbGvRlA"
            else:
                model_name = "gpt-4o"
            # ChatGPT API call
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_url}"},
                            },
                        ],
                    }
                ]

            print(messages)
            response_data = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            # Extract the response content
            # print('Response data: ', response_data)
            response = response_data.choices[0].message.content
            print('Response', response)

            # Extract the result number in brackets (if available)
            result_number = None
            if ":" in response:
                try:
                    # Split and strip the response to isolate the number
                    result_number = response.split(":")[0].strip()
                    # Ensure the result_number is numeric
                    if not result_number.isdigit():
                        result_number = None
                except Exception as e:
                    print(f"Error parsing result number: {e}")

            # print(f"Result for {image_path}: {result_number}")
            results.append((image_path, result_number))

        finally:
            # Delete the GCloud image link
            blob.delete()
            print(f"Deleted GCloud link for image: {image_path}")


    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)


    # Save results to CSV
    with open(csv_output_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Original Image Path", "Result Number"])
        csv_writer.writerows(results)

    print(f"Results saved to {csv_output_path}")

async def query_togetherAI(folder_path, bucket_name, csv_output_path, season):
    """
    Processes all images in a folder, queries the API for each image, and saves the results to a CSV file.
    """
    # Initialize the Together client
    client = Together()

    prompt = kharif_prompt if season == 'kharif' else rabi_prompt

    results = []
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        print(f"Processing image: {image_path}")
        image_url, blob = upload_image_get_url(image_path, bucket_name)

        try:
            # Non-streaming API call (set stream=False or remove the stream parameter)
            response_data = client.chat.completions.create(
                model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                stream=False,  # Disable streaming
            )

            # Extract the response content
            # print(response_data)
            response = response_data.choices[0].message.content
            print(response)

            # Extract the result number in brackets (if available)
            result_number = None
            if "(" in response and ")" in response:
                result_number = response.split("(")[-1].split(")")[0]

            print(f"Result for {image_path}: {result_number}")
            results.append((image_path, result_number))

        finally:
            # Delete the GCloud image link
            blob.delete()
            print(f"Deleted GCloud link for image: {image_path}")


    # Check if the CSV exists, and create it if not
    if not os.path.exists(csv_output_path):
        with open(csv_output_path, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Original Image Path", "Result Number"])

    # Save results to CSV
    with open(csv_output_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Original Image Path", "Result Number"])
        csv_writer.writerows(results)

    print(f"Results saved to {csv_output_path}")

def query_validation_gpt(filename):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:personal:try1:AbbGvRlA",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    )
    print(completion.choices[0].message)
# Main function
if __name__ == "__main__":

    # Note: Set GOOGLE_APPLICATION_CREDENTIALS environment variable before running
    # export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"

    # seasons = ['rabi', 'kharif']
    seasons = ['kharif']


    for season in seasons:

        folder_path = f"/home/sean7391/streetview_highres/RandomSample100/RandomSample100/{season}"  # Update with the path to your folder
        bucket_name = "india_croptype"
        csv_output_path = f"labels1/gpt-4o-finetuned44/random_100_results_{season}.csv"

        # query_gemini(folder_path, bucket_name, csv_output_path, season)

        # csv_output_path = f"labels1/claude/random_100_results_{season}.csv"

        # query_claude(folder_path, bucket_name, csv_output_path, season)

        query_chatGPT(folder_path, bucket_name, csv_output_path, season, finetuned=True)

        # query_validation(folder_path, bucket_name, csv_output_path, season, finetuned=True)

        # csv_output_path = f"labels1/llama/random_100_results_{season}.csv"

        # asyncio.run(query_togetherAI(folder_path, bucket_name, csv_output_path, season))
