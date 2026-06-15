"""
robot_nav/waypoints.py
=======================
Planification de waypoints pour la mission de survey en foret -- BurnTrack.

Composants
----------
Waypoint
    Point de mesure : position grille, priorite, label, visite ou non.

WaypointPlanner
    Genere une liste ordonnee de waypoints depuis une carte NDVI ou de secheresse.
    Algorithme :
      1. Score de priorite = (1 - NDVI) normalise  (sec = score eleve = prioritaire)
      2. Selection des top-N cellules hors zones brulees / coupe-feu
      3. Tri greedy nearest-neighbor depuis la position du robot
         (approximation du TSP en O(N^2) -- largement suffisant pour N < 100)
      4. Filtrage dynamique : retire les waypoints que le feu va atteindre
         avant que le robot puisse y arriver

GPSGrid
    Conversion bidirectionnelle (lat, lon) <-> (row, col) en grille CA.
    Preset Bouskoura inclus : GPSGrid.bouskoura(rows, cols, cell_size=30.0)

Usage
-----
    import numpy as np
    from cellular_automaton import Grid
    from robot_nav.waypoints import WaypointPlanner, GPSGrid

    # Depuis une vraie image NDVI (array 2D, valeurs -1 a 1)
    ndvi = np.load("bouskoura_ndvi.npy")
    grid = Grid.from_arrays(...)
    planner = WaypointPlanner.from_ndvi(ndvi, grid, n_waypoints=25)

    # Ou depuis la secheresse synthetique (pour tests sans donnees satellite)
    planner = WaypointPlanner.from_synthetic(grid, n_waypoints=20, seed=42)

    # Ordonner depuis la position du robot
    ordered = planner.greedy_tour(start=(39, 2))

    # Filtrer les waypoints inaccessibles (feu trop proche)
    from robot_nav.planner import RiskMap, PropagationRules
    risk = RiskMap()
    arrival = risk.build(grid, PropagationRules())
    safe = planner.filter_reachable(arrival, current_time=5.0, safety_margin=10.0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellular_automaton.grid import Grid, CellState

INF = float("inf")
Pos = Tuple[int, int]


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """Point de mesure pour le robot."""
    position:  Pos
    priority:  float = 1.0    # 0 = faible, 1 = critique (zone tres seche)
    label:     str   = ""     # ex: "zone_seche_NE", "lisiere_nord"
    visited:   bool  = False
    reachable: bool  = True   # mis a False si le feu coupe l acces

    def __repr__(self):
        v = "V" if self.visited else " "
        r = "X" if not self.reachable else " "
        return f"WP({self.position}, p={self.priority:.2f} [{v}{r}] {self.label})"


# ---------------------------------------------------------------------------
# WaypointPlanner
# ---------------------------------------------------------------------------

class WaypointPlanner:
    """
    Genere et ordonne les waypoints de la mission de survey.

    Attributes:
        waypoints : Liste ordonnee des waypoints (apres appel a greedy_tour)
    """

    def __init__(self, waypoints: List[Waypoint]):
        self.waypoints = waypoints

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------

    @classmethod
    def from_ndvi(
        cls,
        ndvi: np.ndarray,
        grid: Grid,
        n_waypoints: int = 25,
        min_spacing: int = 3,
        border_margin: int = 1,
    ) -> "WaypointPlanner":
        """
        Genere des waypoints depuis une carte NDVI reelle (Sentinel-2 ou autre).

        Priorite = secheresse = (ndvi_max - ndvi) normalise.
        Les zones les plus seches (NDVI bas) sont les plus prioritaires.

        Args:
            ndvi           : Array (rows, cols) de valeurs NDVI dans [-1, 1].
                             Peut etre None (utilise from_synthetic a la place).
            grid           : Grille CA -- meme dimensions que ndvi.
            n_waypoints    : Nombre de waypoints a selectionner.
            min_spacing    : Espacement minimal entre waypoints (cellules).
                             Evite de concentrer tous les WP dans une meme zone.
            border_margin  : Nombre de cellules a exclure en bordure de grille.
        """
        assert ndvi.shape == (grid.rows, grid.cols), \
            f"NDVI shape {ndvi.shape} != grid shape ({grid.rows}, {grid.cols})"

        # Score de priorite : secheresse normalisee
        ndvi_min = np.nanmin(ndvi)
        ndvi_max = np.nanmax(ndvi)
        span = ndvi_max - ndvi_min if ndvi_max > ndvi_min else 1.0
        dryness = (ndvi_max - ndvi) / span   # 0 = humide, 1 = sec

        return cls._select_waypoints(dryness, grid, n_waypoints,
                                     min_spacing, border_margin)

    @classmethod
    def from_synthetic(
        cls,
        grid: Grid,
        n_waypoints: int = 20,
        min_spacing: int = 3,
        border_margin: int = 1,
        seed: Optional[int] = None,
    ) -> "WaypointPlanner":
        """
        Genere des waypoints avec une carte de secheresse synthetique.

        Utile pour les tests sans donnees satellite reelles.
        La secheresse est simulee par une combinaison de gradients et de bruit
        -- reproduit la variabilite spatiale typique d une foret mediterraneenne.
        """
        rng = np.random.default_rng(seed)
        rows, cols = grid.rows, grid.cols

        # Gradient NE -> SO (zone NE plus seche) + patches aleatoires
        r_idx, c_idx = np.mgrid[0:rows, 0:cols]
        gradient = (r_idx / rows * 0.3 + (cols - c_idx) / cols * 0.3)
        noise    = rng.random((rows, cols)) * 0.4
        dryness  = np.clip(gradient + noise, 0.0, 1.0)

        return cls._select_waypoints(dryness, grid, n_waypoints,
                                     min_spacing, border_margin)

    @classmethod
    def _select_waypoints(
        cls,
        dryness: np.ndarray,
        grid: Grid,
        n_waypoints: int,
        min_spacing: int,
        border_margin: int,
    ) -> "WaypointPlanner":
        """Selectione les top-N cellules par score de secheresse avec espacement minimal."""
        rows, cols = grid.rows, grid.cols

        # Trier toutes les cellules par score decroissant
        flat_order = np.argsort(dryness.ravel())[::-1]
        selected:   List[Waypoint] = []
        occupied:   List[Pos]      = []

        for flat_idx in flat_order:
            if len(selected) >= n_waypoints:
                break
            r, c = divmod(int(flat_idx), cols)

            # Exclure les bordures
            if r < border_margin or r >= rows - border_margin:
                continue
            if c < border_margin or c >= cols - border_margin:
                continue

            # Exclure les coupe-feu et cellules deja brulees
            cell = grid.cells[r][c]
            if cell.state in (CellState.FIREBREAK, CellState.BURNED, CellState.BURNING):
                continue

            # Verifier l espacement minimal (evite les clusters)
            too_close = any(
                abs(r - pr) <= min_spacing and abs(c - pc) <= min_spacing
                for pr, pc in occupied
            )
            if too_close:
                continue

            priority = float(dryness[r, c])
            label    = _priority_label(priority)
            selected.append(Waypoint(position=(r, c), priority=priority, label=label))
            occupied.append((r, c))

        return cls(selected)

    # ------------------------------------------------------------------
    # Ordonnancement greedy (nearest-neighbor TSP)
    # ------------------------------------------------------------------

    def greedy_tour(self, start: Pos) -> List[Waypoint]:
        """
        Ordonne les waypoints par tournee greedy nearest-neighbor depuis `start`.

        A chaque etape, choisit le waypoint non visite le plus proche
        (distance euclidienne en cellules) en preferant les plus prioritaires.

        Score de selection = distance_euclidienne / (priority + 0.1)
        -- les zones tres seches valent le detour.

        Args:
            start : Position courante du robot (row, col)

        Returns:
            Liste ordonnee de Waypoints (meme liste que self.waypoints, reordonnee).
        """
        remaining = list(self.waypoints)
        ordered:  List[Waypoint] = []
        current = start

        while remaining:
            # Score = distance normalisee par priorite (bas = bon)
            def score(wp: Waypoint) -> float:
                dr = wp.position[0] - current[0]
                dc = wp.position[1] - current[1]
                dist = np.sqrt(dr*dr + dc*dc)
                return dist / (wp.priority + 0.1)

            best = min(remaining, key=score)
            ordered.append(best)
            remaining.remove(best)
            current = best.position

        self.waypoints = ordered
        return ordered

    # ------------------------------------------------------------------
    # Filtrage dynamique (mis a jour a chaque pas de simulation)
    # ------------------------------------------------------------------

    def filter_reachable(
        self,
        arrival_time: Dict[Pos, float],
        current_time: float,
        safety_margin: float = 10.0,
    ) -> List[Waypoint]:
        """
        Marque comme inaccessibles les waypoints que le feu atteindra
        avant que le robot puisse y arriver (avec marge de securite).

        A appeler apres chaque mise a jour du RiskMap.

        Args:
            arrival_time   : Sortie de RiskMap.build()
            current_time   : Temps courant de simulation (min)
            safety_margin  : Marge de securite (min)

        Returns:
            Liste des waypoints encore accessibles et non visites.
        """
        accessible = []
        for wp in self.waypoints:
            if wp.visited:
                continue
            t_fire = arrival_time.get(wp.position, INF)
            wp.reachable = (t_fire - current_time) >= safety_margin
            if wp.reachable:
                accessible.append(wp)
        return accessible

    def mark_visited(self, position: Pos):
        """Marque un waypoint comme visite (appele par RobotNavigator)."""
        for wp in self.waypoints:
            if wp.position == position and not wp.visited:
                wp.visited = True
                break

    def next_unvisited(self) -> Optional[Waypoint]:
        """Prochain waypoint accessible non visite (dans l ordre du tour)."""
        for wp in self.waypoints:
            if not wp.visited and wp.reachable:
                return wp
        return None

    def coverage_fraction(self) -> float:
        """Fraction de waypoints visites (0.0 -> 1.0)."""
        if not self.waypoints:
            return 1.0
        visited = sum(1 for wp in self.waypoints if wp.visited)
        return visited / len(self.waypoints)

    def summary(self) -> str:
        total    = len(self.waypoints)
        visited  = sum(1 for wp in self.waypoints if wp.visited)
        blocked  = sum(1 for wp in self.waypoints if not wp.reachable)
        return (f"{visited}/{total} visites, {blocked} bloques par le feu, "
                f"couverture {self.coverage_fraction()*100:.0f}%")


# ---------------------------------------------------------------------------
# Utilitaire label
# ---------------------------------------------------------------------------

def _priority_label(priority: float) -> str:
    if priority > 0.8:
        return "critique"
    if priority > 0.6:
        return "eleve"
    if priority > 0.4:
        return "modere"
    return "faible"


# ---------------------------------------------------------------------------
# GPSGrid -- conversion lat/lon <-> (row, col)
# ---------------------------------------------------------------------------

class GPSGrid:
    """
    Conversion bidirectionnelle entre coordonnees GPS (lat, lon)
    et position dans la grille CA (row, col).

    Convention :
        row 0, col 0 = coin Nord-Ouest de la grille (lat_max, lon_min)
        row augmente vers le Sud, col augmente vers l Est.

    Args:
        lat_nw     : Latitude du coin Nord-Ouest (degres decimaux)
        lon_nw     : Longitude du coin Nord-Ouest (degres decimaux)
        cell_size  : Taille d une cellule en metres
        rows, cols : Dimensions de la grille
    """

    # Constantes de conversion approchees (valides pour le Maroc)
    METERS_PER_DEG_LAT = 111_320.0          # ~constant
    # METERS_PER_DEG_LON varie avec la latitude (cos(lat))

    def __init__(self, lat_nw: float, lon_nw: float,
                 cell_size: float, rows: int, cols: int):
        self.lat_nw    = lat_nw
        self.lon_nw    = lon_nw
        self.cell_size = cell_size
        self.rows      = rows
        self.cols      = cols
        # Resolution en degres par cellule
        self._dlat = cell_size / self.METERS_PER_DEG_LAT
        self._dlon = cell_size / (self.METERS_PER_DEG_LAT * np.cos(np.radians(lat_nw)))

    def latlon_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convertit (lat, lon) en (row, col) dans la grille.
        Retourne (-1,-1) si hors grille.
        """
        row = int((self.lat_nw - lat) / self._dlat)
        col = int((lon - self.lon_nw) / self._dlon)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return (-1, -1)

    def cell_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convertit (row, col) en (lat, lon) -- centre de la cellule."""
        lat = self.lat_nw - (row + 0.5) * self._dlat
        lon = self.lon_nw + (col + 0.5) * self._dlon
        return (lat, lon)

    def path_to_gps(self, path: List[Pos]) -> List[Tuple[float, float]]:
        """Convertit un chemin (liste de (row,col)) en liste de (lat,lon)."""
        return [self.cell_to_latlon(r, c) for r, c in path]

    @classmethod
    def bouskoura(cls, rows: int, cols: int,
                  cell_size: float = 30.0) -> "GPSGrid":
        """
        Preset GPS pour la foret de Bouskoura, Casablanca, Maroc.

        Foret de Bouskoura : ~33.37 N, 7.65 W (coin Nord-Ouest approximatif)
        Surface couverte selon rows*cols*cell_size.

        Usage:
            gps = GPSGrid.bouskoura(rows=200, cols=167, cell_size=30.0)
            row, col = gps.latlon_to_cell(33.35, -7.62)
            lat, lon = gps.cell_to_latlon(10, 20)
        """
        return cls(
            lat_nw=33.38,
            lon_nw=-7.66,
            cell_size=cell_size,
            rows=rows,
            cols=cols,
        )

    def __repr__(self):
        lat_se, lon_se = self.cell_to_latlon(self.rows - 1, self.cols - 1)
        return (f"GPSGrid(NW=({self.lat_nw:.4f}N, {self.lon_nw:.4f}E) "
                f"SE=({lat_se:.4f}N, {lon_se:.4f}E) "
                f"{self.rows}x{self.cols} @ {self.cell_size}m/cell)")
