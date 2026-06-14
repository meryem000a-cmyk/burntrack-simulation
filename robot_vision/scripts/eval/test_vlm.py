import os
import sys
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Initialize Vertex AI with the provided project
project_id = "project-fcd14d48-8db6-4f3d-b95"
location = "us-central1"

try:
    vertexai.init(project=project_id, location=location)
except Exception as e:
    print(f"Error initializing Vertex AI. Make sure you are authenticated (run 'gcloud auth application-default login'): {e}")
    sys.exit(1)

model_name = "gemini-1.5-flash"
image_path = "datasets/african_flora/images/train/acacia_raw/4929878308.jpg"

if not os.path.exists(image_path):
    print(f"Image not found at {image_path}")
    sys.exit(1)

print(f"\nLoading image {image_path}...")
with open(image_path, "rb") as f:
    image_data = f.read()
    
image_part = Part.from_data(data=image_data, mime_type="image/jpeg")

print(f"\nSending to Google Cloud Vertex AI ({model_name}) for classification...")
model = GenerativeModel(model_name=model_name)
prompt = "You are an expert botanist analyzing fuel loads for wildfire models. Look at this plant. Is the majority of the foliage dry, dead, and brown (cured), or is it green and alive? Answer with exactly one word: 'DRY' or 'LIVE'."

try:
    response = model.generate_content([image_part, prompt])
    print("\n--- VLM Output ---")
    print(response.text.strip())
    print("------------------")
except Exception as e:
    print(f"Error generating response. If authentication failed, please run 'gcloud auth application-default login' in your terminal. Error details: {e}")
