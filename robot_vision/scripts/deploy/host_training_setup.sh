#!/bin/bash
# host_training_setup.sh
# Creates virtual environment and installs dependencies for data acquisition and training

echo "Setting up Python virtual environment..."
python3 -m venv flora_env
source flora_env/bin/activate

echo "Installing data processing libraries..."
pip install pygbif pyinaturalist pandas requests

echo "Installing PyTorch and Ultralytics..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics

echo ""
echo "=========================================================="
echo "Environment setup complete!"
echo "=========================================================="
echo "To acquire data, you can run:"
echo "  source flora_env/bin/activate"
echo "  python acquire_data.py"
echo ""
echo "To train the model (assuming dataset is ready), run:"
echo "  yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=150 imgsz=320 batch=32 device=0"
echo ""
echo "To export the model to INT8 TFLite format for the Raspberry Pi, run:"
echo "  yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True data=data.yaml imgsz=320"
echo "=========================================================="
