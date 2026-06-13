# Robot Vision

Plant species classification and dryness (curing state) detection pipeline for the **BurnTrack** project.

This module provides real-time botanical classification and Rothermel curing state assessment (dry vs. alive) on resource-constrained edge hardware (e.g., Raspberry Pi 4) to dynamically update Multi-Dimensional Cellular Automata (MD-CA) fire spread simulations.

## Reorganized Directory Layout

```text
robot_vision/
├── README.md               # This documentation file
├── requirements.txt        # Python library dependencies (ONNX Runtime, Pillow, NumPy, torchvision)
├── config/                 # YAML configs for dataset and training
│   ├── data.yaml
│   ├── flora_cls_data.yaml
│   ├── flora_data.yaml
│   └── quick_demo_data.yaml
├── docs/                   # Exhaustive research guides and edge deployment papers
│   └── Vision_guide.md     # In-depth guide on dataset curation, model training, and Edge TFLite deployment
├── notebooks/              # Jupyter notebooks for interactive training & Kaggle/Colab environments
│   ├── README.md
│   └── train_yolo11m_colab.ipynb
├── models/                 # Pre-trained, optimized production models
│   ├── yolo_fp16.onnx      # YOLO classifier exported in FP16 ONNX format (20MB)
│   └── cnn_fp32.onnx       # Dryness classifier CNN exported in FP32 ONNX format (1.5MB)
├── scripts/                # Python scripts for data preparation, training, evaluation, and inference
│   ├── infer.py            # Primary inference entry point (runs YOLO + Dryness CNN on an image)
│   ├── compress_and_benchmark.py
│   ├── train.sh
│   ├── data/               # Data acquisition and augmentation
│   ├── deploy/             # Edge deployment and JSON orchestration scripts
│   ├── eval/               # Validation and model benchmark routines
│   ├── kaggle/             # Kaggle-specific training notebooks/scripts
│   ├── label/              # Auto-labeling (Moondream VLM) and bounding box annotation scripts
│   ├── train/              # Training loop implementations
│   ├── utils/              # Resizing and helper utility scripts
│   └── vlm/                # Experimental VLM (Moondream/Flora) scripts
└── tests/                  # Automated unit tests
    └── test_inference.py   # Test suite verifying model presence and mock inference execution
```

## Setup & Installation

1. **Install dependencies** (virtual environment recommended):
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Using ONNX Runtime allows us to run inference without installing heavy frameworks like PyTorch or TensorFlow on the edge device.*

2. **Verify Installation**:
   Ensure everything is set up correctly by running the automated unit test suite:
   ```bash
   python3 -m unittest tests/test_inference.py
   ```

## Running Inference

To run the dual-model inference pipeline (YOLOv8/11 ONNX + Dryness CNN ONNX) on one or more images:

```bash
python3 scripts/infer.py path/to/image.jpg
```

### Output Format
The script will output the classified genus and dryness category (curing state) for the input:
```text
  my_image.jpg                        [acacia              , dry]
```

## Model Integration & Rothermel Coupling

- **Genus Classification**: Identifies the botanical family (e.g. *Acacia*, *Adansonia*, *Aloe*, etc.) to look up specific fuel parameters.
- **Dryness Classification**: Maps directly to the Rothermel **LHMC** (Live Herbaceous Moisture Content) parameter. "Dry" indicates curing state (LHMC < 98%, leading to fuel load transfer to dead 1-hour timelag fuels), which dramatically accelerates the cellular automata fire propagation model.

For details on the model training, dataset curation, and TFLite edge deployment parameters (limiting CPU thread allocation for rover control safety), consult the detailed guide in [docs/Vision_guide.md](docs/Vision_guide.md).
