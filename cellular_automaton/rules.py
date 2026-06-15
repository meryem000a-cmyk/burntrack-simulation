"""
cellular_automaton/rules.py
============================
Regles de propagation du feu -- BurnTrack.

Architecture :
  - PropagationRules appelle RothermelEngine pour chaque cellule source.
  - Le ROS (m/min) est pondère directionnellement vers chaque voisin.
  - Probabilite d'ignition par pas de temps : loi exponentielle p = 1-exp(-dt/t_ign).
  - Extinction : duree de combustion = max(tau * facteur, min_burn_min).

Note sur tau Rothermel :
  Le temps de residence tau est tres court pour les herbes fines (0.1-0.3 min).
  Le parametre min_burn_min garantit qu'une cellule reste BURNING assez longtemps
  pour propager le feu meme avec un grand pas de temps (dt = 1 min).
"""

import numpy as np
import sys, os
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from burntrack.engine.rothermel import (
    RothermelEngine,
    FuelModel as RothFuelModel,
    MoistureInputs,
    EnvironmentalConditions,
    RothermelOutput,
)
from burntrack.engine.fuel_models import FuelModel as FMFuelModel

from .grid import Grid, Cell, CellState


# ---------------------------------------------------------------------------
# Conversion FuelModel
# ---------------------------------------------------------------------------

def _to_roth_fuel(fm: FMFuelModel) -> RothFuelModel:
    """Adapte un FuelModel de fuel_models.py au format de RothermelEngine."""
    return RothFuelModel(
        name=fm.name,
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


# ---------------------------------------------------------------------------
# Utilitaires angulaires
# ---------------------------------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Difference angulaire absolue entre deux azimuts (0-180 deg)."""
    d = abs((b - a + 360.0) % 360.0)
    return d if d <= 180.0 else 360.0 - d


def _max_spread_direction(wind_dir_deg: float, slope_pct: float,
                           aspect_deg: float) -> float:
    """
    Direction de propagation maximale (vent + pente combines).

    Pour une pente forte, le feu monte le versant (aspect_deg).
    Sinon il suit le vent (wind_dir_deg + 180).
    Interpolation circulaire ponderee.
    """
    wind_prop = (wind_dir_deg + 180.0) % 360.0
    if slope_pct < 5.0:
        return wind_prop
    w_slope = min(slope_pct / 30.0, 1.0)
    w_wind  = 1.0 - w_slope
    ax = w_wind * np.cos(np.radians(wind_prop)) + w_slope * np.cos(np.radians(aspect_deg))
    ay = w_wind * np.sin(np.radians(wind_prop)) + w_slope * np.sin(np.radians(aspect_deg))
    return float(np.degrees(np.arctan2(ay, ax)) % 360.0)


# ---------------------------------------------------------------------------
# PropagationRules
# ---------------------------------------------------------------------------

class PropagationRules:
    """
    Regles de propagation du feu basees sur Rothermel v3.

    Args:
        stochastic            : True  -> probabiliste (recommande).
                                False -> deterministe (p > 50%).
        burn_duration_factor  : Multiplie tau Rothermel pour la duree de combustion.
        min_burn_min          : Plancher de duree de combustion (min).
                                Essentiel pour les herbes fines (tau ~ 0.1 min).
        min_ros_m_min         : ROS minimal sous lequel la propagation est ignoree.
        directional_exponent  : Puissance du cosinus de ponderation directionnelle.
                                2.0 = propagation assez directionnelle (recommande).
        back_fire_fraction    : Fraction du ROS max conservee dans la direction opposee.
                                0.15 = 15% (feu de recul physiquement realiste).
    """

    def __init__(
        self,
        stochastic: bool = True,
        burn_duration_factor: float = 4.0,
        min_burn_min: float = 5.0,
        min_ros_m_min: float = 0.01,
        directional_exponent: float = 2.0,
        back_fire_fraction: float = 0.15,
    ):
        self.engine               = RothermelEngine()
        self.stochastic           = stochastic
        self.burn_duration_factor = burn_duration_factor
        self.min_burn_min         = min_burn_min
        self.min_ros              = min_ros_m_min
        self.dir_exp              = directional_exponent
        self.back_fire            = back_fire_fraction

    # ------------------------------------------------------------------
    # ROS directionnel
    # ------------------------------------------------------------------

    def _directional_ros(self, ros_max: float, max_spread_dir: float,
                          spread_dir: float) -> float:
        """
        Ponderation directionnelle du ROS maximal.

            ROS(theta) = ROS_max * [back_fire + (1-back_fire) * cos^n(delta/2)]

        Garantit back_fire * ROS_max dans la direction opposee (feu de recul).
        """
        delta    = _angular_diff(max_spread_dir, spread_dir)
        cos_half = max(0.0, np.cos(np.radians(delta / 2.0)))
        weight   = self.back_fire + (1.0 - self.back_fire) * (cos_half ** self.dir_exp)
        return ros_max * weight

    def compute_cell_ros(self, src: Cell, fuel: RothFuelModel,
                          spread_dir: float) -> float:
        """
        ROS de la cellule source vers la direction spread_dir.

        1. Appelle Rothermel avec les conditions de la source.
        2. Ponderation directionnelle selon l'ecart au vent+pente.

        Args:
            src        : Cellule BURNING source
            fuel       : FuelModel converti pour RothermelEngine
            spread_dir : Azimut vers le voisin cible (0=Nord, sens horaire)

        Returns:
            ROS directionnel en m/min
        """
        wind_prop_dir    = (src.wind_dir_deg + 180.0) % 360.0
        angle_wind_slope = _angular_diff(wind_prop_dir, src.aspect_deg)

        conditions = EnvironmentalConditions(
            wind_speed=src.wind_speed_ms,
            slope_pct=src.slope_pct,
            angle_wind_slope=angle_wind_slope,
        )
        output = self.engine.compute(fuel, src.moisture, conditions)

        if output.ros < self.min_ros:
            return 0.0

        max_dir = _max_spread_direction(src.wind_dir_deg, src.slope_pct, src.aspect_deg)
        return self._directional_ros(output.ros, max_dir, spread_dir)

    # ------------------------------------------------------------------
    # Probabilite d'ignition
    # ------------------------------------------------------------------

    def ignition_probability(self, ros_m_min: float, distance_m: float,
                              dt_min: float) -> float:
        """
        Probabilite d'ignition par pas de temps (loi exponentielle).

            t_ign = distance / ROS
            p     = 1 - exp(-dt / t_ign)

        Returns:
            float dans [0, 1]
        """
        if ros_m_min < self.min_ros:
            return 0.0
        t_ign = distance_m / ros_m_min
        return float(np.clip(1.0 - np.exp(-dt_min / t_ign), 0.0, 1.0))

    def _should_ignite(self, ros: float, dist: float, dt: float) -> bool:
        p = self.ignition_probability(ros, dist, dt)
        return bool(np.random.random() < p) if self.stochastic else p > 0.5

    # ------------------------------------------------------------------
    # Duree de combustion
    # ------------------------------------------------------------------

    def _burn_duration(self, tgt: Cell, grid: Grid) -> float:
        """
        Duree de combustion d'une cellule ciblee (min).

        = max(tau_Rothermel * burn_duration_factor, min_burn_min)

        Pour les herbes fines (tau ~ 0.1 min), min_burn_min domine.
        Pour les combustibles lourds (tau ~ 5 min), le facteur domine.
        """
        fm_raw = grid.get_fuel(tgt.fuel_code)
        if fm_raw is None:
            return self.min_burn_min
        fuel = _to_roth_fuel(fm_raw)
        cond = EnvironmentalConditions(
            wind_speed=tgt.wind_speed_ms,
            slope_pct=tgt.slope_pct,
            angle_wind_slope=0.0,
        )
        out = self.engine.compute(fuel, tgt.moisture, cond)
        tau = out.residence_time if out.residence_time > 0 else 1.0
        return max(tau * self.burn_duration_factor, self.min_burn_min)

    # ------------------------------------------------------------------
    # Pas de temps principal
    # ------------------------------------------------------------------

    def apply_step(self, grid: Grid, dt_min: float, current_time: float) -> int:
        """
        Applique un pas de temps a toute la grille (two-pass).

        Pass 1 : identifier nouvelles ignitions et extinctions.
        Pass 2 : appliquer les changements d'etat.

        Args:
            grid         : La grille
            dt_min       : Duree du pas de temps (minutes)
            current_time : Temps courant (min depuis t=0)

        Returns:
            Nombre de nouvelles ignitions ce pas
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
                src.burn_elapsed += dt_min
                if src.burn_elapsed >= src.burn_duration:
                    to_extinguish.append((i, j))
                    continue   # ne propage plus depuis une cellule mourante

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

                    ros = self.compute_cell_ros(src, fuel, spread_dir)
                    if self._should_ignite(ros, dist, dt_min):
                        ignited_set.add((ni, nj))
                        new_ignitions.append((ni, nj))

        # --- Pass 2 : extinctions ---
        for i, j in to_extinguish:
            grid.cells[i][j].state = CellState.BURNED

        # --- Pass 2 : nouvelles ignitions ---
        for i, j in new_ignitions:
            c = grid.cells[i][j]
            c.state         = CellState.BURNING
            c.ignition_time = current_time
            c.burn_elapsed  = 0.0
            c.burn_duration = self._burn_duration(c, grid)

        return len(new_ignitions)


# ---------------------------------------------------------------------------
# CorrectorV3Adapter — ML correcteur de delta_ros
# ---------------------------------------------------------------------------

class CorrectorV3Adapter:
    """
    Adaptateur pour le Corrector V3 (ML).

    Calcule un delta_ros multiplicatif a partir des caracteristiques
    de la cellule, des sorties Rothermel et du contexte environnemental.

    Usage:
        adapter = CorrectorV3Adapter(model, temp_c=35.0, rh_percent=20.0)
        delta = adapter.predict_delta_ros(cell, output)

    Formule :
        ROS_corrige = ROS * (1 + delta_ros)
    """

    def __init__(self, model, **env_context):
        self.model = model
        self.env_context = env_context

    def predict_delta_ros(
        self, cell: Cell, output: RothermelOutput
    ) -> float:
        """
        Predir delta_ros pour une cellule source en combustion.

        Construit le vecteur de features a partir de l'etat de la cellule
        et des sorties Rothermel, puis interroge le modele ML.
        """
        if self.model is None:
            return 0.0

        row = {
            "ros_rothermel": output.ros,
            "phi_w": output.phi_w,
            "phi_s": output.phi_s,
            "phi_eff": output.phi_eff,
            "beta": output.beta,
            "beta_opt": output.beta_opt,
            "gamma": output.gamma,
            "eta_M": output.eta_M,
            "eta_S": output.eta_S,
            "I_R_kW_m2": output.reaction_intensity,
            "xi": output.xi,
            "tau_min": output.residence_time,
            "fireline_intensity": output.fireline_intensity,
            "wind_speed_ms": cell.wind_speed_ms,
            "slope_pct": cell.slope_pct,
            "aspect_deg": cell.aspect_deg,
            "m_1h": cell.moisture.m_1h,
            "m_10h": cell.moisture.m_10h,
            "m_100h": cell.moisture.m_100h,
            "m_live_herb": cell.moisture.m_live_herb,
            "m_live_woody": cell.moisture.m_live_woody,
            "fuel_model_code": cell.fuel_code,
        }
        row.update(self.env_context)

        result = self.model.predict(row)
        return float(result.get("delta_ros", 0.0))


# ---------------------------------------------------------------------------
# EnsembleSimulation — carte de brulage probabiliste
# ---------------------------------------------------------------------------

class PerturbationConfig:
    """
    Configuration des perturbations pour le mode ensemble.

    Chaque parametre est defini comme (loi, *args) :
        - ("gauss", mu, sigma)  : tirage gaussien
        - ("uniform", low, high): tirage uniforme
        - ("fixed", value)      : valeur fixe (pas de perturbation)
        - ("lognormal", mu, sigma): tirage log-normal
    """

    def __init__(
        self,
        wind_speed: Tuple = ("gauss", 0.0, 0.2),
        wind_dir: Tuple = ("gauss", 0.0, 15.0),
        moisture_1h: Tuple = ("gauss", 0.0, 0.02),
        fuel_load: Tuple = ("lognormal", 0.0, 0.1),
    ):
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.moisture_1h = moisture_1h
        self.fuel_load = fuel_load


class EnsembleSimulation:
    """
    Simulation d'ensemble — genere des cartes de brulage probabilistes.

    Cree N realisations en perturbant les entrees (vent, humidite,
    combustible), execute une simulation complete pour chaque tirage,
    puis agrege les resultats en carte de probabilite de brulage.

    Usage:
        ens = EnsembleSimulation(base_grid, n_ensemble=100)
        prob_map = ens.run(steps=120, dt=1.0, ignite_at=(25, 25))
    """

    def __init__(
        self,
        base_grid: Grid,
        n_ensemble: int = 50,
        perturbation: Optional[PerturbationConfig] = None,
        rules: Optional[PropagationRules] = None,
        seed: Optional[int] = None,
    ):
        self.base_grid = base_grid
        self.n = n_ensemble
        self.perturb = perturbation or PerturbationConfig()
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def _sample_perturbation(self) -> dict:
        """Tire un jeu de perturbations pour une realisation."""

        def _sample(param: Tuple) -> float:
            law, *args = param
            if law == "gauss":
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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        steps: int,
        dt: float = 1.0,
        ignite_at: Optional[Tuple[int, int]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Execute N simulations et retourne la carte de probabilite.

        Args:
            steps     : Nombre de pas de temps par realisation
            dt        : Pas de temps (min)
            ignite_at : Point d'ignition (row, col). None = utilise allumettes
                        existantes dans base_grid.
            verbose   : Affiche la progression

        Returns:
            np.ndarray (rows, cols) — probabilite de brulage dans [0, 1]
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

            for pt in ignition_points:
                sim.ignite(*pt)

            sim.run(steps, dt, verbose=False, stop_if_extinct=True)
            burn_count += (grid.state_array() >= 2).astype(np.float64)

            if verbose and (k + 1) % max(1, self.n // 10) == 0:
                pct = (k + 1) / self.n * 100
                print(f"[Ensemble] {k+1:4d}/{self.n} ({pct:.0f}%)")

        return burn_count / self.n

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------

    @staticmethod
    def prob_summary(prob_map: np.ndarray) -> dict:
        """Resume statistique d'une carte de probabilite."""
        return {
            "shape": prob_map.shape,
            "mean_p": float(np.mean(prob_map)),
            "median_p": float(np.median(prob_map)),
            "p90": float(np.percentile(prob_map, 90)),
            "p10": float(np.percentile(prob_map, 10)),
            "high_risk_pct": float(np.mean(prob_map > 0.5) * 100),
            "burned_area_pct": float(np.mean(prob_map > 0.05) * 100),
        }