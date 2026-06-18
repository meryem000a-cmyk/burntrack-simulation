"""
Configuration BurnTrack - Projet étudiant ingénieurs
Hyperparamètres et chemins centralisés
"""

import torch

# === DONNÉES ===
REAL_DATA_PATH = "data/african_ground_truth.csv"
SYNTH_DATA_PATH = "data/synthetic_dataset_balanced_v2.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# === FEATURES ===
CONDITION_COLS = ['wind_speed', 'humidity', 'slope', 'slope_deg', 'slope_pct', 
                  'temp_c', 'rh_percent']
ROS_MEASURED_COLS = ['ros_measured', 'ROS']
ROS_ROTHERMEL_COLS = ['ros_rothermel', 'ROS_r']

# === MODÈLE ===
N_FEATURES = 5
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.2

# === ENTRAÎNEMENT ===
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 20

# === LOSS PONDÉRÉE ===
WEIGHT_REAL = 3.0
WEIGHT_SYNTH = 1.0

# === DEVICE ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === REPRODUCTIBILITÉ ===
SEED = 42
