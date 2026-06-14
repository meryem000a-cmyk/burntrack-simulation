import os
from google import genai
from google.genai import types

# Setup API Key
api_key = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"
client = genai.Client(api_key=api_key)

image_path = "datasets/african_flora/images/train/acacia_raw/4929878308.jpg"

print(f"Loading test image: {image_path}")
with open(image_path, "rb") as f:
    image_data = f.read()

prompt = "Look at this plant. Is the majority of the foliage dry, dead, and brown (cured), or is it green and alive? Answer with exactly one word: 'DRY' or 'LIVE'."

print("\n--- Testing Gemini 3.5 Flash ---")
try:
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=[
            types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
            prompt
        ]
    )
    print(f"Response: {response.text.strip()}")
except Exception as e:
    print(f"Gemini 3.5 Flash failed: {e}")

print("\n--- Testing Gemini 3.1 Flash Lite ---")
try:
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=[
            types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
            prompt
        ]
    )
    print(f"Response: {response.text.strip()}")
except Exception as e:
    print(f"Gemini 3.1 Flash Lite failed: {e}")
