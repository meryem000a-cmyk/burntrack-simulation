#!/usr/bin/env bash
# run_gcp_pipeline.sh - GCP Vertex AI Training Pipeline Orchestration Script (Optimized & Cached)
# ============================================================================================
# This script downscales your 16GB dataset into a highly optimized 512px representation,
# zips and uploads it, and launches the NVIDIA A100 job. It uses smart caching to bypass
# downscaling, zipping, and uploading if the results are already completed!

PROJECT_ID="project-fcd14d48-8db6-4f3d-b95"
BUCKET_NAME="anwar-african-flora-302947"
RAW_DATASET_DIR="/home/anwar/Documents/Vision/datasets/yolo_flora"
OPTIMIZED_DATASET_DIR="/tmp/yolo_flora_optimized"
ZIP_PATH="/tmp/yolo_flora_optimized.zip"
FLORA_ENV_PYTHON="/home/anwar/Documents/Vision/flora_env/bin/python"
DOWNSCALER_SCRIPT="/home/anwar/Documents/Vision/downscale_dataset.py"
ORCHESTRATOR_SCRIPT="/home/anwar/Documents/Vision/gcp_vertex_train.py"

echo "=========================================================================="
echo "           GCP VERTEX AI FLORA TRAINING WORKFLOW (OPTIMIZED)"
echo "=========================================================================="
echo "Project ID:  $PROJECT_ID"
echo "Bucket Name: gs://$BUCKET_NAME"
echo "GPU Target:  NVIDIA A100 (40GB)"
echo "--------------------------------------------------------------------------"

# 1. Downscale the dataset using multi-core processing (Skipped if already processed)
if [ ! -d "$OPTIMIZED_DATASET_DIR" ]; then
    echo "[STEP 1] Downscaling 39,621 images to 512px to optimize bandwidth/storage..."
    $FLORA_ENV_PYTHON "$DOWNSCALER_SCRIPT"
    if [ $? -eq 0 ]; then
        echo "  ✅ Dataset successfully downscaled and optimized!"
    else
        echo "  ❌ Downscaling failed! Exiting."
        exit 1
    fi
else
    echo "[STEP 1] Optimized dataset directory already exists. Skipping downscaling!"
fi

# 2. Zip the optimized dataset (Skipped if zip already exists)
if [ ! -f "$ZIP_PATH" ]; then
    echo "\n[STEP 2] Zipping optimized dataset (~2.2GB)..."
    zip_start=$(date +%s)
    cd "$OPTIMIZED_DATASET_DIR" && zip -r -q "$ZIP_PATH" .
    zip_end=$(date +%s)
    echo "  ✅ Zipping complete! Duration: $((zip_end - zip_start)) seconds."
    echo "  * Final Zip Size: $(du -sh $ZIP_PATH | cut -f1)"
else
    echo "\n[STEP 2] Optimized zip already exists at $ZIP_PATH. Skipping zipping!"
fi

# 3. Upload to GCS (Skipped if already exists in GCS bucket)
echo "\n[STEP 3] Checking if dataset is already uploaded to GCS..."
gcloud storage ls "gs://$BUCKET_NAME/data/yolo_flora.zip" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "  * Dataset not found in GCS. Uploading now..."
    gcloud storage cp "$ZIP_PATH" "gs://$BUCKET_NAME/data/yolo_flora.zip"
    if [ $? -eq 0 ]; then
        echo "  ✅ Dataset successfully uploaded to GCS!"
    else
        echo "  ❌ Upload failed! Please check your internet connection or GCS bucket status."
        exit 1
    fi
else
    echo "  ✅ Dataset already exists in GCS bucket gs://$BUCKET_NAME/data/yolo_flora.zip! Skipping upload."
fi

# 4. Launch the Vertex AI Training Job
echo "\n[STEP 4] Launching your ConvNeXt-Base training job on Vertex AI (NVIDIA L4)..."
$FLORA_ENV_PYTHON $ORCHESTRATOR_SCRIPT \
    --project "$PROJECT_ID" \
    --bucket "$BUCKET_NAME" \
    --gpu "L4"

echo "\n=========================================================================="
echo "                          PIPELINE SUBMISSION COMPLETE"
echo "=========================================================================="
