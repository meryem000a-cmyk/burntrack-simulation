import os
import sys
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Verify that the user has a Google Cloud Project set
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project_id:
    print("ERROR: Please set the GOOGLE_CLOUD_PROJECT environment variable.")
    print("Example: export GOOGLE_CLOUD_PROJECT='your-project-id'")
    sys.exit(1)

# Initialize Vertex AI
try:
    vertexai.init(project=project_id, location="us-central1")
except Exception as e:
    print(f"Error initializing Vertex AI. Make sure you are authenticated (gcloud auth application-default login): {e}")
    sys.exit(1)

model_name = "gemini-1.5-flash-001" # Extremely fast and cost-effective for classification
image_path = "datasets/african_flora/images/train/acacia_raw/4929878308.jpg"

if not os.path.exists(image_path):
    print(f"Image not found at {image_path}. Please make sure acquire_data.py has downloaded it.")
    sys.exit(1)

print(f"Loading image {image_path}...")
with open(image_path, "rb") as f:
    image_data = f.read()
    
image_part = Part.from_data(data=image_data, mime_type="image/jpeg")

# The VLM
model = GenerativeModel(model_name=model_name)

# Our classification prompt
prompt = "You are an expert botanist analyzing fuel loads for wildfire models. Look at this plant. Is the majority of the foliage dry, dead, and brown (cured), or is it green and alive? Answer with exactly one word: 'DRY' or 'LIVE'."

print(f"\nSending to Google Cloud Vertex AI ({model_name}) for classification...")
try:
    response = model.generate_content([image_part, prompt])
    print("\n--- VLM Output ---")
    print(response.text.strip())
    print("------------------")
except Exception as e:
    print(f"Error generating response: {e}")
