"""
cellular_automaton/grid.py
==========================
Grille de l'Automate Cellulaire — BurnTrack.

Chaque cellule contient son état (UNBURNED/BURNING/BURNED/FIREBREAK),
ses propriétés de combustible, d'humidité et d'environnement.
La grille sert de support à la simulation de propagation du feu.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Generator, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from burntrack.engine.fuel_models import FuelModel, ALL_FUEL_MODELS
from burntrack.engine.rothermel import MoistureInputs


class CellState(IntEnum):
    """États possibles d'une cellule."""
    UNBURNED  = 0  # Végétation intacte — non brûlée
    BURNING   = 1  # En cours de combustion
    BURNED    = 2  # Cendres — combustible épuisé
    FIREBREAK = 3  # Coupe-feu : eau, route, roche — ne brûle pas


@dataclass
class Cell:
    """
    Cellule élémentaire de la grille.

    Attributes:
        state          : État courant de la cellule
        fuel_code      : Code du fuel model (clé dans ALL_FUEL_MODELS)
        moisture       : Humidités des combustibles
        slope_pct      : Pente (%)
        aspect_deg     : Orientation de la pente (degrés, 0=Nord, sens horaire)
        elevation_m    : Altitude (m)
        wind_speed_ms  : Vitesse du vent local (m/s)
        wind_dir_deg   : Direction d'où vient le vent (degrés, convention météo)
        ignition_time  : Temps d'ignition en minutes depuis t=0, None si non brûlé
        burn_duration  : Durée de combustion (min) — calculée par Rothermel à l'ignition
        burn_elapsed   : Temps écoulé en combustion (min)
        rh_percent     : Humidité relative de l'air (%) — requis pour le MLP
        temp_c         : Température de l'air (°C) — requis pour le MLP
    """
    state: CellState = CellState.UNBURNED
    fuel_code: str = "GR2"
    moisture: MoistureInputs = field(default_factory=MoistureInputs)
    slope_pct: float = 0.0
    aspect_deg: float = 0.0          # 0=Nord, 90=Est, 180=Sud, 270=Ouest
    elevation_m: float = 0.0
    wind_speed_ms: float = 3.0
    wind_dir_deg: float = 270.0      # Vent d'Ouest par défaut (convention : d'où vient le vent)
    ignition_time: Optional[float] = None
    burn_duration: float = 10.0      # sera recalculé à l'ignition
    burn_elapsed: float = 0.0
    delta_ros: float = 0.0           # correction IA du ROS
    rh_percent: float = 50.0         # humidité relative (%) — pour MLP
    temp_c: float = 25.0             # température (°C) — pour MLP
    ignition_buffer: float = 0.0     # accumulateur d'énergie thermique sub-grid
    ignition_threshold: float = 1.0  # seuil d'ignition (stochastique ou fixe)


class Grid:
    """
    Grille 2D de l'automate cellulaire.

    Args:
        rows      : Nombre de lignes (axe Nord-Sud)
        cols      : Nombre de colonnes (axe Est-Ouest)
        cell_size : Résolution spatiale en mètres (côté d'une cellule carrée)
    """

    # Voisinage de Moore (8 directions)
    # (delta_row, delta_col, angle_vers_voisin_en_degrés)
    NEIGHBOR_OFFSETS: List[Tuple[int, int, float]] = [
        (-1,  0,   0.0),   # Nord
        (-1,  1,  45.0),   # Nord-Est
        ( 0,  1,  90.0),   # Est
        ( 1,  1, 135.0),   # Sud-Est
        ( 1,  0, 180.0),   # Sud
        ( 1, -1, 225.0),   # Sud-Ouest
        ( 0, -1, 270.0),   # Ouest
        (-1, -1, 315.0),   # Nord-Ouest
    ]
    DIAG_SCALE = float(np.sqrt(2))

    def __init__(self, rows: int, cols: int, cell_size: float = 30.0):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.cells: List[List[Cell]] = [
            [Cell() for _ in range(cols)] for _ in range(rows)
        ]
        self._fuel_cache: Dict[str, FuelModel] = dict(ALL_FUEL_MODELS)

    def cell(self, row: int, col: int) -> Cell:
        return self.cells[row][col]

    def set_state(self, row: int, col: int, state: CellState):
        self.cells[row][col].state = state

    def get_fuel(self, code: str) -> Optional[FuelModel]:
        """Retourne le FuelModel correspondant au code, ou None si inconnu."""
        return self._fuel_cache.get(code)

    def neighbors(
        self, row: int, col: int
    ) -> Generator[Tuple[int, int, float, float], None, None]:
        """
        Génère les voisins valides de (row, col).
        """
        for di, dj, angle in self.NEIGHBOR_OFFSETS:
            ni, nj = row + di, col + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                is_diag = (di != 0 and dj != 0)
                dist = self.cell_size * (self.DIAG_SCALE if is_diag else 1.0)
                yield ni, nj, dist, angle

    def state_array(self) -> np.ndarray:
        """Matrice numpy des états entiers — pour visualisation et export."""
        arr = np.zeros((self.rows, self.cols), dtype=np.int8)
        for i in range(self.rows):
            for j in range(self.cols):
                arr[i, j] = int(self.cells[i][j].state)
        return arr

    def burning_count(self) -> int:
        return sum(
            1
            for i in range(self.rows)
            for j in range(self.cols)
            if self.cells[i][j].state == CellState.BURNING
        )

    def burned_fraction(self) -> float:
        """Fraction de la grille brûlée ou en combustion (hors FIREBREAK)."""
        total = sum(
            1
            for i in range(self.rows)
            for j in range(self.cols)
            if self.cells[i][j].state != CellState.FIREBREAK
        )
        if total == 0:
            return 0.0
        burned = sum(
            1
            for i in range(self.rows)
            for j in range(self.cols)
            if self.cells[i][j].state in (CellState.BURNING, CellState.BURNED)
        )
        return burned / total

    @classmethod
    def uniform(
        cls,
        rows: int,
        cols: int,
        cell_size: float = 30.0,
        fuel_code: str = "GR2",
        moisture_1h: float = 0.06,
        wind_speed_ms: float = 3.0,
        wind_dir_deg: float = 270.0,
        slope_pct: float = 0.0,
        aspect_deg: float = 0.0,
    ) -> "Grid":
        """
        Grille uniforme — terrain homogène pour tests et benchmarks.
        """
        g = cls(rows, cols, cell_size)
        for i in range(rows):
            for j in range(cols):
                c = g.cells[i][j]
                c.fuel_code = fuel_code
                c.moisture = MoistureInputs(
                    m_1h=moisture_1h,
                    m_10h=moisture_1h + 0.01,
                    m_100h=moisture_1h + 0.02,
                    m_live_herb=min(moisture_1h * 6, 1.0),
                    m_live_woody=min(moisture_1h * 8, 1.0),
                )
                c.wind_speed_ms = wind_speed_ms
                c.wind_dir_deg = wind_dir_deg
                c.slope_pct = slope_pct
                c.aspect_deg = aspect_deg
        return g

    @classmethod
    def from_arrays(
        cls,
        fuel_codes: np.ndarray,
        slope_pct: np.ndarray,
        aspect_deg: np.ndarray,
        elevation_m: np.ndarray,
        wind_speed: np.ndarray,
        wind_dir: np.ndarray,
        moisture_1h: np.ndarray,
        cell_size: float = 30.0,
        rh_percent: np.ndarray = None,
        temp_c: np.ndarray = None,
    ) -> "Grid":
        """
        Crée une grille à partir d'arrays numpy (données raster GIS / ERA5).
        """
        rows, cols = fuel_codes.shape
        g = cls(rows, cols, cell_size)
        for i in range(rows):
            for j in range(cols):
                c = g.cells[i][j]
                c.fuel_code = str(fuel_codes[i, j])
                c.slope_pct = float(slope_pct[i, j])
                c.aspect_deg = float(aspect_deg[i, j])
                c.elevation_m = float(elevation_m[i, j])
                c.wind_speed_ms = float(wind_speed[i, j])
                c.wind_dir_deg = float(wind_dir[i, j])
                m = float(moisture_1h[i, j])
                c.moisture = MoistureInputs(
                    m_1h=m,
                    m_10h=m + 0.01,
                    m_100h=m + 0.02,
                    m_live_herb=min(m * 6, 1.0),
                    m_live_woody=min(m * 8, 1.0),
                )
                if rh_percent is not None:
                    c.rh_percent = float(rh_percent[i, j])
                if temp_c is not None:
                    c.temp_c = float(temp_c[i, j])
        return g

    def add_firebreak(
        self,
        row_start: int, col_start: int,
        row_end: int,   col_end: int,
    ):
        """
        Trace un coupe-feu (cellules FIREBREAK) entre deux points (Bresenham).
        """
        r0, c0, r1, c1 = row_start, col_start, row_end, col_end
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        while True:
            if 0 <= r0 < self.rows and 0 <= c0 < self.cols:
                self.cells[r0][c0].state = CellState.FIREBREAK
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
