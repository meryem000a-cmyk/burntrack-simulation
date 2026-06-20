"""
cellular_automaton/rules.py
===========================
Règles de propagation du feu pour l'Automate Cellulaire — BurnTrack.

Implémente la discrétisation de la propagation basée sur le ROS (m/min),
la distance physique entre cellules et l'intensité du vent et de la pente.
Supporte également la simulation d'ensemble stochastique.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from .grid import Grid, Cell, CellState
from burntrack.engine.rothermel import (
    RothermelEngine, FuelModel as RothFuelModel, 
    MoistureInputs, EnvironmentalConditions, RothermelOutput
)


def _to_roth_fuel(fm) -> RothFuelModel:
    """Convertit un FuelModel global en format attendu par RothermelEngine."""
    return RothFuelModel(
        name=fm.code,
        w_1h=fm.w_1h,
        w_10h=fm.w_10h,
        w_100h=fm.w_100h,
        w_live_herb=fm.w_live_herb,
        w_live_woody=fm.w_live_woody,
        sigma_1h=fm.sigma_1h,
        sigma_10h=fm.sigma_10h,
        sigma_100h=fm.sigma_100h,
        sigma_live_herb=fm.sigma_live_herb,
        sigma_live_woody=fm.sigma_live_woody,
        delta=fm.delta,
        mx=fm.mx,
        h_dead=fm.h_dead,
        h_live=fm.h_live,
        st=getattr(fm, "st", 0.0555),
        se=getattr(fm, "se", 0.01),
    )


class PropagationRules:
    """
    Règles physiques régissant le passage du feu de cellule en cellule.
    """

    def __init__(self, use_corrector: bool = True):
        self.engine = RothermelEngine()
        self.use_corrector = use_corrector

    def compute_cell_ros(self, cell: Cell, grid: Grid) -> RothermelOutput:
        """Calcule la vitesse de propagation brute (Rothermel v3) pour une cellule."""
        fm_raw = grid.get_fuel(cell.fuel_code)
        if fm_raw is None:
            # Sécurité : ROS nul si pas de combustible
            return RothermelOutput(ros=0.0, flame_length=0.0, fireline_intensity=0.0,
                                   heat_per_unit_area=0.0, fuel_consumption=0.0,
                                   spread_direction=0.0, phi_w=0.0, phi_s=0.0, phi_eff=0.0)

        fuel = _to_roth_fuel(fm_raw)
        
        # Calcul de la direction relative vent/pente
        wind_prop_dir = (cell.wind_dir_deg + 180.0) % 360.0
        aspect = cell.aspect_deg
        angle_wind_slope = abs((wind_prop_dir - aspect + 360.0) % 360.0)
        if angle_wind_slope > 180.0:
            angle_wind_slope = 360.0 - angle_wind_slope

        conditions = EnvironmentalConditions(
            wind_speed=cell.wind_speed_ms,
            slope_pct=cell.slope_pct,
            angle_wind_slope=angle_wind_slope
        )

        return self.engine.compute(fuel, cell.moisture, conditions)

    def apply_step(self, grid: Grid, dt_min: float) -> List[Tuple[int, int]]:
        """
        Avance l'état de la grille d'un pas de temps dt_min.
        Retourne la liste des nouvelles cellules enflammées (i, j).
        """
        new_ignitions: List[Tuple[int, int]] = []
        ignited_set = set()
        to_extinguish: List[Tuple[int, int]] = []

        # --- Pass 1 ---
        for i in range(grid.rows):
            for j in range(grid.cols):
                src = grid.cells[i][j]

                if src.state != CellState.BURNING:
                    continue

                # Avancer le compteur de combustion
                if src.burn_elapsed + dt_min >= src.burn_duration:
                    src.burn_elapsed = src.burn_duration
                    to_extinguish.append((i, j))
                    continue   # ne propage plus depuis une cellule mourante
                src.burn_elapsed += dt_min

                # Fuel model de la source
                fm_raw = grid.get_fuel(src.fuel_code)
                if fm_raw is None:
                    continue
                fuel = _to_roth_fuel(fm_raw)

                # Propagation vers chaque voisin UNBURNED
                for ni, nj, dist, spread_dir in grid.neighbors(i, j):
                    tgt = grid.cells[ni][nj]
                    if tgt.state != CellState.UNBURNED:
                        continue
                    if (ni, nj) in ignited_set:
                        continue

                    # Calcul du ROS dans la direction du voisin
                    out_src = self.compute_cell_ros(src, grid)
                    ros_base = out_src.ros

                    # Ajout de la correction IA si activée
                    ros_final = ros_base
                    if self.use_corrector and src.delta_ros != 0.0:
                        ros_final = max(0.0, ros_base + src.delta_ros)

                    if ros_final <= 0.0:
                        continue

                    # Direction de propagation maximale vs direction du voisin
                    max_dir = out_src.spread_direction
                    theta = np.radians(spread_dir - max_dir)
                    # Formule d'ellipse simplifiée pour le ROS directionnel
                    ros_dir = ros_final * np.cos(theta)

                    if ros_dir <= 0.0:
                        continue

                    # Temps requis pour franchir la distance entre les centres des deux cellules
                    time_to_spread = dist / ros_dir  # en minutes

                    # Probabilité de propagation sur ce pas de temps (dt)
                    prob = 1.0 - np.exp(-dt_min / time_to_spread)

                    if np.random.rand() < prob:
                        new_ignitions.append((ni, nj))
                        ignited_set.add((ni, nj))

        # --- Pass 2 : Application des changements d'état ---
        for i, j in to_extinguish:
            grid.cells[i][j].state = CellState.BURNED

        for i, j in new_ignitions:
            cell = grid.cells[i][j]
            cell.state = CellState.BURNING
            cell.ignition_time = dt_min  # sera ajusté par le runner principal
            # Calcul de sa propre durée de combustion pour sa future extinction
            out = self.compute_cell_ros(cell, grid)
            cell.burn_duration = max(1.0, out.fuel_consumption / (out.ros + 1e-5))

        return new_ignitions


# ---------------------------------------------------------------------------
# CorrectorV3Adapter — ML correcteur de delta_ros
# ---------------------------------------------------------------------------

RF_FEATURE_NAMES = [
    "ros_rothermel", "temp_c", "rh_percent", "wind_speed_ms", "vpd_kpa",
    "slope_deg", "slope_pct", "angle_wind_slope",
    "w_total_kg_m2", "w_dead_kg_m2", "w_live_kg_m2", "delta_m", "sigma_m2_m3",
    "mx_percent", "h_dead_kj_kg", "phi_w", "phi_s", "phi_eff",
    "beta", "beta_opt", "beta_ratio", "gamma", "eta_M", "eta_S",
    "I_R_kW_m2", "xi", "tau_min", "ndvi", "ndwi", "lst_c", "dfmc_percent",
]


class CorrectorV3Adapter:
    """
    Adaptateur pour le Corrector V3 (ML).
    """

    def __init__(self, rules=None):
        self.rules = rules
        self.model = None
        self.scaler = None
        self.feature_names = list(RF_FEATURE_NAMES)
        self.env_context = {}

    def set_model(self, model, scaler=None, **env_context):
        self.model = model
        self.scaler = scaler
        self.env_context = env_context
        if scaler is not None and hasattr(scaler, 'n_features_in_'):
            self.feature_names = RF_FEATURE_NAMES[:scaler.n_features_in_]

    def _build_cell_features(self, cell: Cell, output=None) -> dict:
        from burntrack.corrector.features import compute_vpd, compute_dfmc

        temp_c = getattr(cell, 'temp_c', 25.0)
        rh_percent = getattr(cell, 'rh_percent', 50.0)
        vpd = compute_vpd(temp_c, rh_percent)
        dfmc = getattr(cell, 'dfmc', compute_dfmc(temp_c, vpd))

        wind_prop_dir = (getattr(cell, 'wind_dir_deg', 270.0) + 180.0) % 360.0
        aspect = getattr(cell, 'aspect_deg', 0.0)
        angle_wind_slope = abs((wind_prop_dir - aspect + 360.0) % 360.0)
        if angle_wind_slope > 180.0:
            angle_wind_slope = 360.0 - angle_wind_slope

        # Valeurs environnementales par défaut
        features = {
            "temp_c": temp_c,
            "rh_percent": rh_percent,
            "wind_speed_ms": getattr(cell, 'wind_speed_ms', 3.0),
            "vpd_kpa": vpd,
            "slope_deg": np.degrees(np.arctan(getattr(cell, 'slope_pct', 0.0) / 100.0)),
            "slope_pct": getattr(cell, 'slope_pct', 0.0),
            "angle_wind_slope": angle_wind_slope,
            "ndvi": self.env_context.get("ndvi", 0.3),
            "ndwi": self.env_context.get("ndwi", -0.1),
            "lst_c": self.env_context.get("lst_c", temp_c + 5.0),
            "dfmc_percent": dfmc,
        }

        # Valeurs physiques Rothermel
        if output:
            features.update({
                "ros_rothermel": output.ros,
                "phi_w": output.phi_w,
                "phi_s": output.phi_s,
                "phi_eff": output.phi_eff,
                "I_R_kW_m2": output.fireline_intensity,
            })

        return features

    def apply_to_grid(self, grid: Grid):
        if self.model is None or self.scaler is None:
            return

        import torch

        all_features = []
        cell_coords = []

        for i in range(grid.rows):
            for j in range(grid.cols):
                cell = grid.cell(i, j)
                if cell.state == CellState.FIREBREAK:
                    continue

                out = None
                if self.rules:
                    out = self.rules.compute_cell_ros(cell, grid)

                feat_dict = self._build_cell_features(cell, out)
                vector = [feat_dict.get(name, 0.0) for name in self.feature_names]
                all_features.append(vector)
                cell_coords.append((i, j))

        if not all_features:
            return

        x = np.array(all_features, dtype=np.float32)
        x_scaled = self.scaler.transform(x)

        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32)
            preds = self.model(x_t).numpy()

        for idx, (i, j) in enumerate(cell_coords):
            grid.cell(i, j).delta_ros = float(preds[idx][0])


# ---------------------------------------------------------------------------
# EnsembleSimulation — Simulation d'ensemble stochastique
# ---------------------------------------------------------------------------

@dataclass
class PerturbationConfig:
    """Paramètres stochastiques des distributions de perturbation."""
    wind_speed: Tuple[str, float, float] = ("normal", 0.0, 0.20)
    wind_dir: Tuple[str, float, float] = ("normal", 0.0, 15.0)
    moisture_1h: Tuple[str, float, float] = ("normal", 0.0, 0.02)
    fuel_load: Tuple[str, float, float] = ("normal", 0.0, 0.15)


class EnsembleSimulation:
    """
    Lance plusieurs simulations de propagation stochastiques en perturbant
    les conditions aux limites pour obtenir une carte de probabilité.
    """

    def __init__(self, grid: Grid, n_realizations: int = 50,
                 rules: Optional[PropagationRules] = None,
                 perturb: Optional[PerturbationConfig] = None,
                 seed: Optional[int] = None):
        self.base_grid = grid
        self.n = n_realizations
        self.rules = rules
        self.perturb = perturb if perturb else PerturbationConfig()
        self.rng = np.random.default_rng(seed)

    def _sample_perturbation(self) -> dict:
        def _sample(p):
            law, *args = p
            if law == "normal":
                return float(self.rng.normal(*args))
            elif law == "uniform":
                return float(self.rng.uniform(*args))
            elif law == "lognormal":
                return float(self.rng.lognormal(*args))
            elif law == "fixed":
                return float(args[0])
            return 0.0

        return {
            "delta_wind_speed": _sample(self.perturb.wind_speed),
            "delta_wind_dir": _sample(self.perturb.wind_dir),
            "delta_moisture": _sample(self.perturb.moisture_1h),
            "delta_fuel_load": _sample(self.perturb.fuel_load),
        }

    def _apply_perturbation(self, grid: Grid, delta: dict) -> Grid:
        """Copie la grille et applique les perturbations."""
        import copy

        g = copy.deepcopy(grid)

        for i in range(g.rows):
            for j in range(g.cols):
                c = g.cells[i][j]

                # Vent
                c.wind_speed_ms = max(0.0, c.wind_speed_ms * (1.0 + delta["delta_wind_speed"]))
                c.wind_dir_deg = (c.wind_dir_deg + delta["delta_wind_dir"]) % 360.0

                # Humidite
                c.moisture.m_1h = float(np.clip(c.moisture.m_1h + delta["delta_moisture"], 0.01, 0.50))
                c.moisture.m_10h = float(np.clip(c.moisture.m_10h + delta["delta_moisture"] * 0.8, 0.02, 0.45))
                c.moisture.m_100h = float(np.clip(c.moisture.m_100h + delta["delta_moisture"] * 0.6, 0.03, 0.40))

        return g

    def run(
        self,
        steps: int,
        dt: float = 1.0,
        ignite_at: Optional[Tuple[int, int]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Execute N simulations et retourne la carte de probabilite.
        """
        from .simulation import FireSimulation

        rows, cols = self.base_grid.rows, self.base_grid.cols
        burn_count = np.zeros((rows, cols), dtype=np.float64)

        for k in range(self.n):
            delta = self._sample_perturbation()
            grid = self._apply_perturbation(self.base_grid, delta)

            sim = FireSimulation(
                grid,
                rules=PropagationRules() if self.rules is None else self.rules,
                seed=int(self.rng.integers(0, 2**31)),
            )

            if ignite_at is not None:
                sim.ignite(*ignite_at)

            sim.run(steps, dt, verbose=False, stop_if_extinct=True)

            burn_count += (grid.state_array() >= 2).astype(np.float64)

            if verbose and (k + 1) % max(1, self.n // 10) == 0:
                pct = (k + 1) / self.n * 100
                print(f"[Ensemble] {k+1:4d}/{self.n} ({pct:.0f}%)")

        return burn_count / self.n

    def run_with_ignitions(
        self,
        steps: int,
        ignition_points: List[Tuple[int, int]],
        dt: float = 1.0,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Execute l'ensemble avec plusieurs points d'ignition simultanes.
        """
        from .simulation import FireSimulation

        rows, cols = self.base_grid.rows, self.base_grid.cols
        burn_count = np.zeros((rows, cols), dtype=np.float64)

        for k in range(self.n):
            delta = self._sample_perturbation()
            grid = self._apply_perturbation(self.base_grid, delta)

            sim = FireSimulation(
                grid,
                rules=PropagationRules() if self.rules is None else self.rules,
                seed=int(self.rng.integers(0, 2**31)),
            )

            for i, j in ignition_points:
                sim.ignite(i, j)

            sim.run(steps, dt, verbose=False, stop_if_extinct=True)

            burn_count += (grid.state_array() >= 2).astype(np.float64)

            if verbose and (k + 1) % max(1, self.n // 10) == 0:
                pct = (k + 1) / self.n * 100
                print(f"[Ensemble] {k+1:4d}/{self.n} ({pct:.0f}%)")

        return burn_count / self.n
