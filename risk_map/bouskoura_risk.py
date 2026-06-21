"""
risk_map/bouskoura_risk.py
==========================
Carte de risque d'éclosion pour la forêt de Bouskoura (Casablanca, Maroc).

Methodology (d'après Assali et al., 2016 — Chefchaouen-Ouazzane):
  Risk = w1*Flammabilité + w2*Topoclimat + w3*NDVI_risk + w4*Pression_humaine

Pipeline:
  1. Polygone de la forêt (coordonnées OSM approximatives)
  2. Grille rasterisée — cellules hors forêt → FIREBREAK
  3. Score de risque 0-1 par cellule
  4. Retourne (Grid, risk_array, meta)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shapely.geometry import Point, Polygon, MultiPolygon
from cellular_automaton.grid import Grid, Cell, CellState
from burntrack.engine.rothermel import MoistureInputs

# ---------------------------------------------------------------------------
# Bouskoura forest polygon (approximatif, basé sur données cartographiques)
# Forêt de Bouskoura — ~1800 ha, eucalyptus + pins, SE de Casablanca
# ---------------------------------------------------------------------------
BOUSKOURA_POLYGON_LATLON = [
    (33.400, -7.653),  # NW  (lisière nord-ouest, près du golf)
    (33.402, -7.636),  # N   (le long de P3020)
    (33.397, -7.622),  # NE  (angle nord-est — Route de Bouskoura à Ouled Ali)
    (33.380, -7.621),  # E   (limite est — longe Route de Bouskoura à Ouled Ali)
    (33.363, -7.625),  # SE
    (33.352, -7.637),  # S   (près de RP3040)
    (33.351, -7.650),  # SW
    (33.362, -7.661),  # W
    (33.381, -7.661),  # NW-W
]

# Grid parameters
CELL_SIZE_M   = 40.0   # 40m resolution → manageable grid
WIND_DIR_DOM  = 270.0  # vent dominant d'Ouest (typique Maroc atlantique)
WIND_SPEED_MS = 5.0
MOISTURE_1H   = 0.06

# Fuel model assignment by zone (simplifié Bouskoura)
# Bouskoura = plantation d'eucalyptus + pins + matorral en sous-bois
FUEL_ZONES = [
    # (polygon_vertices_latlon, fuel_code, description)
    # Zone centrale : eucalyptus dense
    ([(33.385,-7.655),(33.385,-7.630),(33.365,-7.630),(33.365,-7.655)],
     "TU1", "Eucalyptus dense"),
    # Zone nord : pins maritimes
    ([(33.390,-7.660),(33.390,-7.635),(33.378,-7.635),(33.378,-7.660)],
     "TU2", "Pins maritimes"),
]

# Flammabilité par fuel model (0-1), adapté Assali et al.
# Codes doivent exister dans ALL_FUEL_MODELS de burntrack.engine.fuel_models
FLAMMABILITY = {
    "AF_MAQUIS":      0.90,   # matorral sec — très inflammable (Bouskoura sous-bois)
    "SH5":            0.80,   # arbustes moyens
    "SH7":            0.85,   # arbustes denses
    "AF_CHENE_LIEGE": 0.65,   # chêne-liège / eucalyptus dense
    "AF_CEDRE":       0.60,   # pins / cèdre
    "GR4":            0.75,   # pelouse sèche en lisière
    "GR2":            0.50,   # pelouse courte (hors forêt)
    "default":        0.60,
}


def _latlon_to_grid(lat, lon, lat_nw, lon_nw, cell_size_m):
    """Convertit lat/lon en indices (row, col) de grille."""
    # 1 degré lat ≈ 111 000 m
    # 1 degré lon ≈ 111 000 * cos(lat) m
    lat_m_per_deg = 111_000.0
    lon_m_per_deg = 111_000.0 * np.cos(np.radians(lat_nw))
    row = int((lat_nw - lat) * lat_m_per_deg / cell_size_m)
    col = int((lon - lon_nw) * lon_m_per_deg / cell_size_m)
    return row, col


def _grid_to_latlon(row, col, lat_nw, lon_nw, cell_size_m):
    lat_m_per_deg = 111_000.0
    lon_m_per_deg = 111_000.0 * np.cos(np.radians(lat_nw))
    lat = lat_nw - row * cell_size_m / lat_m_per_deg
    lon = lon_nw + col * cell_size_m / lon_m_per_deg
    return lat, lon


def _make_forest_mask(rows, cols, lat_nw, lon_nw, cell_size_m, poly):
    """Rasterise le polygone forêt → masque booléen (True = forêt).
    Vectorisé via matplotlib.path — ~100x plus rapide que Shapely point-by-point.
    """
    from matplotlib.path import Path

    lat_m = 111_000.0
    lon_m = 111_000.0 * np.cos(np.radians(lat_nw))

    row_centers = lat_nw - (np.arange(rows) + 0.5) * cell_size_m / lat_m
    col_centers = lon_nw + (np.arange(cols) + 0.5) * cell_size_m / lon_m

    lon_grid, lat_grid = np.meshgrid(col_centers, row_centers)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    poly_coords = np.array(poly.exterior.coords)
    path = Path(poly_coords)
    mask = path.contains_points(points).reshape(rows, cols)
    return mask


def _synthetic_ndvi(rows, cols, mask, rng):
    """
    NDVI synthétique spatialement corrélé.
    Forêt dense au centre (NDVI 0.5-0.7), plus sec aux bords (0.2-0.4).
    """
    from scipy.ndimage import gaussian_filter

    # Champ aléatoire de base
    base = rng.uniform(0.0, 1.0, (rows, cols))
    # Lissage spatial pour corrélation spatiale
    smooth = gaussian_filter(base, sigma=4.0)
    # Normaliser 0-1
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)

    # Distance au centre de la forêt
    cx, cy = rows / 2, cols / 2
    dist = np.sqrt(((np.arange(rows)[:, None] - cx) / rows) ** 2 +
                   ((np.arange(cols)[None, :] - cy) / cols) ** 2)
    dist_norm = dist / (dist.max() + 1e-9)

    # NDVI = haute valeur au centre, plus sec aux bords
    ndvi = 0.65 - 0.30 * dist_norm + 0.15 * smooth
    ndvi = np.clip(ndvi, 0.1, 0.85)
    ndvi[~mask] = 0.1   # hors forêt = très sec
    return ndvi


def _synthetic_slope(rows, cols, rng):
    """Pente synthétique (%) — terrain légèrement ondulé."""
    from scipy.ndimage import gaussian_filter
    base = rng.uniform(0, 1, (rows, cols))
    smooth = gaussian_filter(base, sigma=6.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    return smooth * 25.0   # 0-25% slope


def _synthetic_aspect(rows, cols, rng):
    """Aspect synthétique (degrés) — variation douce."""
    from scipy.ndimage import gaussian_filter
    base = rng.uniform(0, 360, (rows, cols))
    # Smooth circularly
    sin_base = gaussian_filter(np.sin(np.radians(base)), sigma=5.0)
    cos_base = gaussian_filter(np.cos(np.radians(base)), sigma=5.0)
    return np.degrees(np.arctan2(sin_base, cos_base)) % 360.0


def compute_risk_score(ndvi, slope_pct, aspect_deg, fuel_code,
                        wind_dir_deg=WIND_DIR_DOM, is_edge=False):
    """
    Score de risque d'éclosion 0-1 pour une cellule.

    Adapté de Assali et al. (2016) :
      Risk = 0.35*Flam + 0.25*Topoclimat + 0.25*NDVI_risk + 0.15*Pression_humaine
    """
    # 1. Flammabilité du modèle de combustible
    flam = FLAMMABILITY.get(fuel_code, FLAMMABILITY["default"])

    # 2. Topoclimat = f(pente, exposition vs vent dominant)
    # Pente : plus c'est pentu, plus le feu monte vite
    slope_factor = min(1.0, slope_pct / 30.0)

    # Exposition : versant sec (S, SW) + dans le sens du vent dominant → risque max
    wind_spread_dir = (wind_dir_deg + 180.0) % 360.0   # direction vers laquelle souffle le vent
    angle_diff = abs((aspect_deg - wind_spread_dir + 360.0) % 360.0)
    if angle_diff > 180.0:
        angle_diff = 360.0 - angle_diff
    aspect_factor = 1.0 - angle_diff / 180.0   # 1 si exposition = direction vent

    topoclim = 0.5 * slope_factor + 0.5 * aspect_factor

    # 3. Risque NDVI (végétation sèche = risque élevé)
    ndvi_risk = 1.0 - np.clip(ndvi, 0.0, 1.0)

    # 4. Pression humaine (lisière de forêt = plus exposée)
    human = 0.8 if is_edge else 0.2

    # Score final pondéré
    score = (0.35 * flam + 0.25 * topoclim + 0.25 * ndvi_risk + 0.15 * human)
    return float(np.clip(score, 0.0, 1.0))


def build_bouskoura_grid(cell_size_m=CELL_SIZE_M, seed=42):
    """
    Construit la grille CA de la forêt de Bouskoura avec carte de risque.

    Returns:
        grid       : Grid CA avec cellules FIREBREAK hors forêt
        risk_array : np.ndarray (rows, cols) de scores 0-1
        ndvi_array : np.ndarray (rows, cols)
        meta       : dict avec lat_nw, lon_nw, cell_size_m, forest_cells
    """
    rng = np.random.default_rng(seed)

    # Polygone forêt (lon, lat pour Shapely)
    poly_coords = [(lon, lat) for lat, lon in BOUSKOURA_POLYGON_LATLON]
    forest_poly = Polygon(poly_coords)

    # Bounding box de la grille
    lons = [c[0] for c in poly_coords]
    lats = [c[1] for c in poly_coords]
    margin_deg = 0.005   # ~500m de marge autour du polygone
    lat_nw = max(lats) + margin_deg
    lon_nw = min(lons) - margin_deg
    lat_se = min(lats) - margin_deg
    lon_se = max(lons) + margin_deg

    # Dimensions de la grille
    lat_m = 111_000.0
    lon_m = 111_000.0 * np.cos(np.radians(lat_nw))
    rows = max(20, int((lat_nw - lat_se) * lat_m / cell_size_m) + 1)
    cols = max(20, int((lon_se - lon_nw) * lon_m / cell_size_m) + 1)

    print(f"[BouskouraGrid] Grille {rows}×{cols} à {cell_size_m}m/cellule")

    # Masque forêt
    print("[BouskouraGrid] Rasterisation du polygone forêt...")
    forest_mask = _make_forest_mask(rows, cols, lat_nw, lon_nw, cell_size_m, forest_poly)
    n_forest = forest_mask.sum()
    print(f"[BouskouraGrid] {n_forest} cellules forêt ({n_forest * cell_size_m**2 / 10000:.0f} ha)")

    # Terrains synthétiques
    ndvi   = _synthetic_ndvi(rows, cols, forest_mask, rng)
    slope  = _synthetic_slope(rows, cols, rng)
    aspect = _synthetic_aspect(rows, cols, rng)

    # Détection des cellules de lisière (bord de forêt)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(forest_mask, iterations=2)
    edge_mask = forest_mask & ~eroded

    # Assignation fuel model selon NDVI
    def ndvi_to_fuel(n, is_forest):
        if not is_forest:
            return "GR2"
        if n > 0.55:
            return "AF_CHENE_LIEGE"   # eucalyptus/chêne-liège dense
        elif n > 0.40:
            return "AF_CEDRE"         # pins / couvert intermédiaire
        elif n > 0.25:
            return "SH7"              # matorral mixte dense
        else:
            return "AF_MAQUIS"        # matorral sec → très inflammable

    # Construction de la grille CA
    grid = Grid(rows, cols, cell_size_m)
    risk_array = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            c = grid.cells[i][j]
            in_forest = bool(forest_mask[i, j])

            if not in_forest:
                c.state    = CellState.FIREBREAK
                c.fuel_code = "GR2"
                risk_array[i, j] = 0.0
                continue

            fuel = ndvi_to_fuel(ndvi[i, j], True)
            c.fuel_code    = fuel
            c.slope_pct    = float(slope[i, j])
            c.aspect_deg   = float(aspect[i, j])
            c.wind_speed_ms = WIND_SPEED_MS
            c.wind_dir_deg  = WIND_DIR_DOM
            c.rh_percent    = 35.0   # été marocain sec
            c.temp_c        = 32.0
            m1h = MOISTURE_1H * (0.5 + ndvi[i, j])  # NDVI élevé = plus humide
            c.moisture = MoistureInputs(
                m_1h=float(np.clip(m1h, 0.02, 0.20)),
                m_10h=float(np.clip(m1h + 0.01, 0.03, 0.25)),
                m_100h=float(np.clip(m1h + 0.02, 0.04, 0.30)),
                m_live_herb=float(np.clip(m1h * 5, 0.30, 1.20)),
                m_live_woody=float(np.clip(m1h * 7, 0.50, 1.50)),
            )

            risk_array[i, j] = compute_risk_score(
                ndvi=ndvi[i, j],
                slope_pct=slope[i, j],
                aspect_deg=aspect[i, j],
                fuel_code=fuel,
                wind_dir_deg=WIND_DIR_DOM,
                is_edge=bool(edge_mask[i, j]),
            )

    meta = {
        "lat_nw": lat_nw,
        "lon_nw": lon_nw,
        "lat_se": lat_se,
        "lon_se": lon_se,
        "cell_size_m": cell_size_m,
        "rows": rows,
        "cols": cols,
        "forest_cells": int(n_forest),
        "forest_ha": float(n_forest * cell_size_m**2 / 10000),
    }

    return grid, risk_array, ndvi, meta
