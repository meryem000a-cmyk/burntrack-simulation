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
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rothermel.rothermel_engine_v3 import (
    RothermelEngine,
    FuelModel as RothFuelModel,
    MoistureInputs,
    EnvironmentalConditions,
    RothermelOutput,
)
from rothermel.fuel_models import FuelModel as FMFuelModel

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
