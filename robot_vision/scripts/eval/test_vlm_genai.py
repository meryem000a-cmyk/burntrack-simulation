import os
from google import genai
from google.genai import types

# Initialize the GenAI client with API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"
client = genai.Client(api_key=api_key)

image_path = "datasets/african_flora/images/train/acacia_raw/4929878308.jpg"

print(f"\nLoading image {image_path}...")
with open(image_path, "rb") as f:
    image_data = f.read()

print(f"\nSending to Google GenAI (gemini-2.5-flash) for classification...")
prompt = "You are an expert botanist analyzing fuel loads for wildfire models. Look at this plant. Is the majority of the foliage dry, dead, and brown (cured), or is it green and alive? Answer with exactly one word: 'DRY' or 'LIVE'."

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_data,
                mime_type='image/jpeg',
            ),
            prompt
        ]
    )
    print("\n--- VLM Output ---")
    print(response.text.strip())
    print("------------------")
except Exception as e:
    print(f"Error generating response: {e}")
