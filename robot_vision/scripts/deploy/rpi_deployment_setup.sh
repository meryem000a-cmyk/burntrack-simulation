#!/bin/bash
# rpi_deployment_setup.sh
# Installs dependencies for the Raspberry Pi 4 edge deployment

echo "Installing Raspberry Pi dependencies..."
python3 -m pip install tflite-runtime numpy opencv-python

echo ""
echo "=========================================================="
echo "Raspberry Pi environment setup complete!"
echo "=========================================================="
echo "To run the vision inference on the rover, make sure you have the exported .tflite model and run:"
echo "  python rover_vision.py"
echo "=========================================================="
