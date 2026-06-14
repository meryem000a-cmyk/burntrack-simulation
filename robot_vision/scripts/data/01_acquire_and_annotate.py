import os
import glob
import random
import shutil
import time
import concurrent.futures
import threading
from google import genai
from google.genai import types

# Force the valid API key found in your .env
api_key = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"
client = genai.Client(api_key=api_key)

# Configuration
MAX_WORKERS = 8 # Safe concurrency limit to prevent system-wide disk I/O lockup
TEST_SAMPLES = 10 # Total test samples to reserve

source_dir = "datasets/african_flora/images/train"
raw_dirs = glob.glob(os.path.join(source_dir, "*_raw"))

dataset_base = "datasets/yolo_flora"
img_dirs = {
    "train": os.path.join(dataset_base, "images/train"),
    "val": os.path.join(dataset_base, "images/val"),
}
lbl_dirs = {
    "train": os.path.join(dataset_base, "labels/train"),
    "val": os.path.join(dataset_base, "labels/val"),
}

for d in list(img_dirs.values()) + list(lbl_dirs.values()):
    os.makedirs(d, exist_ok=True)

# Generate class mapping automatically based on raw dirs
species_list = sorted([os.path.basename(d).replace("_raw", "") for d in raw_dirs])
class_map = {}
class_id = 0
for sp in species_list:
    class_map[f"{sp}_not_dry"] = class_id
    class_map[f"{sp}_dry"] = class_id + 1
    class_id += 2

print("Generated Class Mapping:")
for k, v in class_map.items():
    print(f"  {v}: {k}")

# Write data.yaml
yaml_content = f"path: {os.path.abspath(dataset_base)}\ntrain: images/train\nval: images/val\n\nnames:\n"
for k, v in class_map.items():
    yaml_content += f"  {v}: {k}\n"
with open("flora_data.yaml", "w") as f:
    f.write(yaml_content)

def annotate_with_gemini(img_path, retry_count=0):
    print(f"  Annotating {os.path.basename(img_path)}...")
    
    # Check if file is empty to prevent 400 errors
    if os.path.getsize(img_path) == 0:
        print("  Annotation error: File is empty (0 bytes).")
        try:
            os.remove(img_path)
        except Exception:
            pass
        return None
        
    with open(img_path, "rb") as f:
        image_data = f.read()
    
    prompt = "Look at this plant. Is the majority of the foliage dry, dead, and brown (cured), or is it green and alive? Answer with exactly one word: 'DRY' or 'LIVE'."
    # 1. Try Gemini 3.5 Flash first
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[
                types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
                prompt
            ]
        )
        ans = response.text.strip().upper()
        if "DRY" in ans:
            return "dry"
        elif "LIVE" in ans:
            return "not_dry"
        return "not_dry" # Fallback
    except Exception as flash_err:
        # If Gemini 3.5 Flash fails (e.g. quota limit, transient error), try Gemini 3.1 Flash Lite
        print(f"\n[!] Gemini 3.5 Flash failed: {flash_err}. Trying fallback Gemini 3.1 Flash Lite...", flush=True)
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
                    prompt
                ]
            )
            ans = response.text.strip().upper()
            if "DRY" in ans:
                return "dry"
            elif "LIVE" in ans:
                return "not_dry"
            return "not_dry" # Fallback
        except Exception as e:
            # If BOTH fail, handle retries or delete bad image
            if retry_count < 3:
                print(f"\n[!] Dual fallback failed: {e}. Retrying in 15s (Attempt {retry_count+1}/3)...", flush=True)
                time.sleep(15)
                return annotate_with_gemini(img_path, retry_count + 1)
            else:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str:
                    print(f"\n[!] API Quota Exceeded. Skipping {os.path.basename(img_path)} for now (retaining raw image).", flush=True)
                    return None
                print(f"\n[!] Image {os.path.basename(img_path)} failed after 3 retries. Deleting corrupted image...", flush=True)
                try:
                    os.remove(img_path)
                except Exception as del_err:
                    print(f"Failed to delete bad image: {del_err}", flush=True)
                return None

print("\n--- Starting Annotation Pipeline ---")

all_processed_images = []

for raw_dir in raw_dirs:
    species = os.path.basename(raw_dir).replace("_raw", "")
    images = glob.glob(os.path.join(raw_dir, "*.jpg"))
    if not images:
        continue
        
    print(f"\nDiscovered {species} ({len(images)} images)...")
    for img_path in images:
        all_processed_images.append((species, img_path))

# Split the dataset 85% train, 15% test
random.shuffle(all_processed_images)
split_idx = int(len(all_processed_images) * 0.85)
train_set = all_processed_images[:split_idx]
test_set = all_processed_images[split_idx:]

class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.lock = threading.Lock()
        
    def increment(self):
        with self.lock:
            self.current += 1
            if self.current % 10 == 0 or self.current == self.total:
                percent = (self.current / self.total) * 100
                print(f"Progress: {self.current}/{self.total} ({percent:.2f}%) processed...", flush=True)

def process_single_image(args):
    species, img_path, split_name, tracker = args
    img_name = f"{species}_{os.path.basename(img_path)}"
    dest_img_path = os.path.join(img_dirs[split_name], img_name)
    dest_lbl_path = os.path.join(lbl_dirs[split_name], img_name.replace(".jpg", ".txt"))
    
    # If both label and image exist, we are fully done. Skip!
    if os.path.exists(dest_lbl_path) and os.path.exists(dest_img_path):
        # Automatically clean up the original in raw directory to keep disk space minimal
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
        tracker.increment()
        return
        
    # If the label doesn't exist, we need to annotate it
    if not os.path.exists(dest_lbl_path):
        # Annotate using the original raw image path to prevent disk write thrashing before success
        dryness = annotate_with_gemini(img_path)
        if dryness is None:
            tracker.increment()
            return
            
        class_name = f"{species}_{dryness}"
        c_id = class_map[class_name]
        
        # Write the label file
        try:
            with open(dest_lbl_path, "w") as f:
                f.write(f"{c_id} 0.5 0.5 1.0 1.0\n")
        except Exception as e:
            print(f"Failed to write label for {img_name}: {e}", flush=True)
            tracker.increment()
            return

    # Copy the image if it is missing (only after we know it has a valid label)
    if not os.path.exists(dest_img_path):
        try:
            shutil.copy2(img_path, dest_img_path)
            # Delete original from raw directory after copying to save space
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            print(f"Failed to copy/cleanup {img_name}: {e}", flush=True)
            
    tracker.increment()

def process_and_move(dataset, split_name):
    tracker = ProgressTracker(len(dataset))
    tasks = [(species, img_path, split_name, tracker) for species, img_path in dataset]
    
    batch_size = 500
    print(f"Starting concurrent annotation with {MAX_WORKERS} workers in batches of {batch_size}...", flush=True)
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Force consumption of the executor map generator to cleanly close/discard all Future objects
            list(executor.map(process_single_image, batch))
        
        # Explicitly clean up thread references and trigger garbage collection to free RSS RAM back to OS
        import gc
        gc.collect()

print(f"\nAnnotating {len(train_set)} TRAIN images...")
process_and_move(train_set, "train")

print(f"\nAnnotating {len(test_set)} VAL/TEST images (these will NOT be trained on)...")
process_and_move(test_set, "val")

print("\nAnnotation Complete!")
