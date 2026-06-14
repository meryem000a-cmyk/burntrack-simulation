#!/usr/bin/env python3
"""
gcp_vertex_train.py - GCP Vertex AI Training Orchestrator for ConvNeXt-Base Flora Classifier
========================================================================================
Runs your high-capacity model on Google Cloud's enterprise accelerators (NVIDIA A100 or L4)
using Vertex AI Custom Jobs.

COST PROFILE (For a 50-epoch training run):
  - NVIDIA L4 GPU (g2-standard-4)  -> ~$1.01 per hour. Run time: ~10 mins -> Cost: ~$0.17
  - NVIDIA A100 (a2-highgpu-1g)    -> ~$3.67 per hour. Run time: ~4 mins  -> Cost: ~$0.25
  * This is serverless: you only pay for the exact minutes the container runs!

PRE-REQUISITES on Google Cloud Platform:
  1. Create a GCP Project and enable billing.
  2. Enable the following APIs: Vertex AI API, Cloud Storage API, Container Registry API.
  3. Create a Cloud Storage Bucket (e.g., gs://african-flora-data).
"""

import os
import sys
import argparse
from google.cloud import storage
try:
    from google.cloud import aiplatform
except ImportError:
    print("[INFO] Installing google-cloud-aiplatform...")
    os.system("pip install -q google-cloud-aiplatform")
    from google.cloud import aiplatform

def parse_args():
    parser = argparse.ArgumentParser(description="Submit ConvNeXt training job to GCP Vertex AI")
    parser.add_argument("--project", type=str, required=True, help="Your GCP Project ID")
    parser.add_argument("--bucket", type=str, required=True, help="Your Cloud Storage Bucket name (without gs://)")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP Region (e.g., us-central1)")
    parser.add_argument("--gpu", type=str, default="L4", choices=["L4", "A100", "T4"], help="Target GPU accelerator")
    return parser.parse_args()

def upload_dataset_to_gcs(bucket_name, local_dir, gcs_destination):
    """Zips the local dataset and uploads it directly to your GCS Bucket."""
    import shutil
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    zip_name = "yolo_flora"
    zip_file = f"{zip_name}.zip"
    
    # 1. Zip dataset
    if not os.path.exists(zip_file):
        print(f"[STEP 1] Zipping {local_dir} into {zip_file} for high-speed GCS transfer...")
        shutil.make_archive(zip_name, 'zip', local_dir)
    else:
        print(f"[STEP 1] Zip archive {zip_file} already exists, skipping compression.")

    # 2. Upload to GCS
    print(f"[STEP 2] Uploading {zip_file} to gs://{bucket_name}/{gcs_destination}...")
    blob = bucket.blob(gcs_destination)
    blob.upload_from_filename(zip_file)
    print("  ✅ Upload complete!")
    return f"gs://{bucket_name}/{gcs_destination}"

def main():
    args = parse_args()
    
    # Map target GPU to GCP hardware configurations
    gpu_configs = {
        "L4": {
            "machine_type": "g2-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1
        },
        "A100": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1
        },
        "T4": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1
        }
    }
    
    cfg = gpu_configs[args.gpu]
    
    print("=" * 70)
    print("          GCP VERTEX AI TRAINING SUBMISSION ENGINE")
    print("=" * 70)
    print(f"GCP Project:      {args.project}")
    print(f"GCS Bucket:       gs://{args.bucket}")
    print(f"Selected GPU:     NVIDIA {args.gpu} (Machine: {cfg['machine_type']})")
    print("-" * 70)

    # 1. Upload local training script and zipped data
    local_data_dir = "/home/anwar/Documents/Vision/datasets/yolo_flora"
    
    # Check if we have the dataset locally
    if not os.path.exists(local_data_dir):
        # Fallback for Kaggle environment
        local_data_dir = "/kaggle/input/africanplantclass/yolo_flora"
        
    # Check if the optimized zip has already been uploaded by our pipeline script
    if os.path.exists("/tmp/yolo_flora_optimized"):
        print("[INFO] Optimized dataset detected! Bypassing local 16GB zipping and uploading stage.")
        print(f"       Will use the already uploaded optimized dataset gs://{args.bucket}/data/yolo_flora.zip.")
        gcs_data_uri = f"gs://{args.bucket}/data/yolo_flora.zip"
    elif os.path.exists(local_data_dir):
        gcs_data_uri = upload_dataset_to_gcs(args.bucket, local_data_dir, "data/yolo_flora.zip")
    else:
        print("[WARNING] Local dataset folder not found. Assumes data is already in GCS.")
        gcs_data_uri = f"gs://{args.bucket}/data/yolo_flora.zip"

    # 2. Initialize Vertex AI SDK
    print("\n[STEP 3] Connecting to Vertex AI API...")
    aiplatform.init(
        project=args.project, 
        location=args.location, 
        staging_bucket=f"gs://{args.bucket}/staging"
    )

    # 3. Create Custom Training Job
    # We use a SOTA Google pre-built PyTorch container image with CUDA pre-installed
    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"
    
    print(f"[STEP 4] Packaging 'kaggle_train_efficientnet.py' into training container...")
    job = aiplatform.CustomTrainingJob(
        display_name="african_flora_convnext_training",
        script_path="kaggle_train_efficientnet.py",
        container_uri=container_uri,
        requirements=["timm", "numpy", "pandas", "torchvision", "PyYAML"]
    )

    # 4. Submit Job
    print(f"\n[🚀 LAUNCH] Submitting Job to Google Cloud GPU Pool...")
    print("  * This job runs asynchronously on GCP. You will see progress logs below.")
    
    model = job.run(
        replica_count=1,
        machine_type=cfg["machine_type"],
        accelerator_type=cfg["accelerator_type"],
        accelerator_count=cfg["accelerator_count"],
        args=[
            "--data", gcs_data_uri,
            "--epochs", "60",
            "--batch", "32" if args.gpu != "T4" else "16"
        ]
    )

    print("\n" + "=" * 70)
    print("               VERTEX AI TRAINING JOB COMPLETE!")
    print("  Check your Cloud Storage Bucket output/ or Vertex AI dashboard to get weights.")
    print("=" * 70)

if __name__ == "__main__":
    main()
