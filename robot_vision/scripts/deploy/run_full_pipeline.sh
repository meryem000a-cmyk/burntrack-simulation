#!/bin/bash
set -e

echo "=========================================================="
echo "    Pan-African Flora Vision Pipeline (Full Production)   "
echo "=========================================================="
echo "Note: This will process the entire 80GB dataset."
echo "      Annotation and training will take a significant amount of time."

# 1. Setup Environment
echo -e "\n[1/5] Setting up environment..."
source flora_env/bin/activate || { echo "Run 'python3 -m venv flora_env' first."; exit 1; }
pip install -q pyyaml ultralytics "google-genai" numpy opencv-python "tflite-runtime" || echo "Pip install finished."

# 2. VLM Annotation on Existing Dataset
echo -e "\n[2/5] Annotating the full dataset with Gemini 2.5 Flash..."
python3 01_acquire_and_annotate.py

# 3. Train Lightweight YOLO Model
echo -e "\n[3/5] Training YOLO11n on annotated data (100 epochs)..."
echo "      (15% of images are held out for testing)"
# Clean up old runs
rm -rf runs/detect/train
yolo task=detect mode=train model=yolo11n.pt data=flora_data.yaml epochs=100 imgsz=320 batch=32

# 4. Quantize to INT8
echo -e "\n[4/5] Quantizing model to INT8 TFLite for edge..."
yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True data=flora_data.yaml imgsz=320

# 5. Run Inference
echo -e "\n[5/5] Running inference on 10 UNSEEN test pictures..."
python3 02_run_inference.py

echo -e "\n=========================================================="
echo "                   PIPELINE COMPLETE                      "
echo "=========================================================="
