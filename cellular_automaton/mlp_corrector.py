"""
cellular_automaton/mlp_corrector.py
====================================
MLP Corrector Adapter pour l'Automate Cellulaire — BurnTrack.

Charge le modèle MLP v2 (PyTorch) et applique la correction delta_ros
à chaque cellule de la grille avant la simulation de propagation.

Architecture MLP: 8→64→32→1 (GELU, LayerNorm, Dropout)
Features: wind_speed_ms, rh_percent, slope_pct, ros_rothermel,
          h_dead_kj_kg, sigma_m2_m3, m_live_woody, mx_percent

Usage:
    from cellular_automaton.mlp_corrector import MLPCorrector
    from cellular_automaton.grid import Grid
    from cellular_automaton.simulation import FireSimulation

    grid = Grid.uniform(50, 50, fuel_code="GR2", moisture_1h=0.06)
    corrector = MLPCorrector("models/mlp_v2_best.pt", "models/mlp_v2_scaler.joblib")
    corrector.apply_to_grid(grid)

    sim = FireSimulation(grid)
    sim.ignite(25, 25)
    sim.run(steps=60, dt=1.0)
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .grid import Grid, Cell, CellState

warnings.filterwarnings("ignore")

# Features attendues par le MLP (ordre exact)
MLP_FEATURE_NAMES = [
    "wind_speed_ms",
    "rh_percent",
    "slope_pct",
    "ros_rothermel",
    "h_dead_kj_kg",
    "sigma_m2_m3",
    "m_live_woody",
    "mx_percent",
]


class MLPCorrector:
    """
    Correcteur ML pour l'automate cellulaire.

    Charge un modèle MLP PyTorch et applique la correction delta_ros
    à toutes les cellules de la grille en une seule passe batch.

    Args:
        model_path : Chemin vers le checkpoint PyTorch (.pt)
        scaler_path: Chemin vers le scaler joblib (.joblib)
        device     : 'cpu' ou 'cuda' (défaut: 'cpu')
    """

    def __init__(
        self,
        model_path: str = "models/mlp_v2_best.pt",
        scaler_path: str = "models/mlp_v2_scaler.joblib",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = device
        self.model = None
        self.scaler = None
        self._loaded = False

    def load(self):
        """Charge le modèle et le scaler."""
        import torch
        import joblib
        from burntrack.corrector.mlp_v2 import BurnTrackMLP

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle MLP non trouvé: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler non trouvé: {self.scaler_path}")

        self.model = BurnTrackMLP()
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.scaler = joblib.load(self.scaler_path)
        self._loaded = True

    def _get_fuel_properties(self, fuel_code: str, grid: Grid) -> dict:
        """Extrait les propriétés physiques du fuel model."""
        fuel = grid.get_fuel(fuel_code)
        if fuel is None:
            return {
                "h_dead_kj_kg": 18622.0,
                "sigma_m2_m3": 1500.0,
                "mx_percent": 20.0,
            }
        return {
            "h_dead_kj_kg": fuel.h_dead,
            "sigma_m2_m3": self._compute_weighted_sav(fuel),
            "mx_percent": fuel.mx,
        }

    def _compute_weighted_sav(self, fuel) -> float:
        """Calcule le SAV moyen pondéré du fuel model."""
        w_total = fuel.w_total
        if w_total <= 0:
            return fuel.sigma_1h

        w_dead = fuel.w_1h + fuel.w_10h + fuel.w_100h
        w_live = fuel.w_live_herb + fuel.w_live_woody

        if w_dead > 0:
            sigma_dead = (
                fuel.sigma_1h * fuel.w_1h +
                fuel.sigma_10h * fuel.w_10h +
                fuel.sigma_100h * fuel.w_100h
            ) / w_dead
        else:
            sigma_dead = fuel.sigma_1h

        if w_live > 0:
            sigma_live = (
                fuel.sigma_live_herb * fuel.w_live_herb +
                fuel.sigma_live_woody * fuel.w_live_woody
            ) / w_live
        else:
            sigma_live = fuel.sigma_live_herb if fuel.sigma_live_herb > 0 else fuel.sigma_1h

        sigma_total = (sigma_dead * w_dead + sigma_live * w_live) / w_total
        return sigma_total if sigma_total > 0 else fuel.sigma_1h

    def _compute_rothermel_ros(self, cell: Cell, grid: Grid) -> float:
        """Calcule le ROS Rothermel de base pour une cellule."""
        from burntrack.engine.rothermel import (
            RothermelEngine,
            FuelModel as RothFuelModel,
            MoistureInputs,
            EnvironmentalConditions,
        )

        fuel_raw = grid.get_fuel(cell.fuel_code)
        if fuel_raw is None:
            return 0.0

        fuel = RothFuelModel(
            name=fuel_raw.name,
            w_1h=fuel_raw.w_1h, w_10h=fuel_raw.w_10h, w_100h=fuel_raw.w_100h,
            w_live_herb=fuel_raw.w_live_herb, w_live_woody=fuel_raw.w_live_woody,
            sigma_1h=fuel_raw.sigma_1h, sigma_10h=fuel_raw.sigma_10h,
            sigma_100h=fuel_raw.sigma_100h, sigma_live_herb=fuel_raw.sigma_live_herb,
            sigma_live_woody=fuel_raw.sigma_live_woody,
            delta=fuel_raw.delta, mx=fuel_raw.mx,
            h_dead=fuel_raw.h_dead, h_live=fuel_raw.h_live,
            st=getattr(fuel_raw, "st", 0.0555),
            se=getattr(fuel_raw, "se", 0.01),
        )

        wind_prop_dir = (cell.wind_dir_deg + 180.0) % 360.0
        aspect = cell.aspect_deg
        angle_wind_slope = abs((wind_prop_dir - aspect + 360.0) % 360.0)
        if angle_wind_slope > 180.0:
            angle_wind_slope = 360.0 - angle_wind_slope

        conditions = EnvironmentalConditions(
            wind_speed=cell.wind_speed_ms,
            slope_pct=cell.slope_pct,
            angle_wind_slope=angle_wind_slope,
        )

        engine = RothermelEngine()
        output = engine.compute(fuel, cell.moisture, conditions)
        return output.ros

    def _build_features(self, cell: Cell, grid: Grid) -> np.ndarray:
        """Construit les 8 features pour une cellule."""
        fuel_props = self._get_fuel_properties(cell.fuel_code, grid)
        ros_rothermel = self._compute_rothermel_ros(cell, grid)

        return np.array([[
            cell.wind_speed_ms,
            getattr(cell, "rh_percent", 50.0),  # Cell n'a pas rh_percent, defaut 50%
            cell.slope_pct,
            ros_rothermel,
            fuel_props["h_dead_kj_kg"],
            fuel_props["sigma_m2_m3"],
            cell.moisture.m_live_woody,
            fuel_props["mx_percent"],
        ]], dtype=np.float32)

    def predict_delta_ros(self, cell: Cell, grid: Grid) -> float:
        """Prédit le delta_ros pour une cellule."""
        if not self._loaded:
            self.load()

        import torch

        x = self._build_features(cell, grid)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            delta = self.model(x_tensor)

        return float(delta.cpu().numpy()[0])

    def apply_to_grid(self, grid: Grid):
        """
        Applique la correction MLP à toutes les cellules de la grille.

        Passe batch unique pour efficacité. Ne modifie que les cellules
        UNBURNED (les cellules BURNING/BURNED ont déjà leur delta_ros).
        """
        if not self._loaded:
            self.load()

        import torch

        all_features = []
        cell_indices = []

        for i in range(grid.rows):
            for j in range(grid.cols):
                c = grid.cells[i][j]
                if c.state == CellState.FIREBREAK:
                    continue

                features = self._build_features(c, grid)
                all_features.append(features[0])
                cell_indices.append((i, j))

        if not all_features:
            return

        x = np.array(all_features, dtype=np.float32)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(x_tensor)

        deltas = predictions.cpu().numpy()

        for idx, (i, j) in enumerate(cell_indices):
            grid.cells[i][j].delta_ros = float(deltas[idx])

    def apply_to_grid_batch(self, grid: Grid, batch_size: int = 1024):
        """
        Version batch pour les très grandes grilles.
        Traite les cellules par paquets de batch_size.
        """
        if not self._loaded:
            self.load()

        import torch

        all_features = []
        cell_indices = []

        for i in range(grid.rows):
            for j in range(grid.cols):
                c = grid.cells[i][j]
                if c.state == CellState.FIREBREAK:
                    continue

                features = self._build_features(c, grid)
                all_features.append(features[0])
                cell_indices.append((i, j))

        if not all_features:
            return

        x_all = np.array(all_features, dtype=np.float32)
        x_scaled = self.scaler.transform(x_all)

        for start in range(0, len(x_scaled), batch_size):
            end = min(start + batch_size, len(x_scaled))
            x_batch = torch.tensor(x_scaled[start:end], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                preds = self.model(x_batch)

            deltas = preds.cpu().numpy()
            for idx in range(start, end):
                i, j = cell_indices[idx]
                grid.cells[i][j].delta_ros = float(deltas[idx - start])
