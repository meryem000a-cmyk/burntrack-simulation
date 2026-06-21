"""
risk_map/ignition_selector.py
=============================
Sélection des points d'ignition prioritaires à partir de la carte de risque.

Algorithme :
  1. Trier toutes les cellules forêt par score de risque décroissant
  2. Sélectionner itérativement — rejeter si trop proche d'un point déjà sélectionné
  3. Retourne une liste de dicts avec position, score, coordonnées GPS
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def select_ignition_points(
    risk_array: np.ndarray,
    forest_mask: Optional[np.ndarray] = None,
    n_points: int = 15,
    min_cell_dist: int = 5,
    lat_nw: float = 33.40,
    lon_nw: float = -7.68,
    cell_size_m: float = 40.0,
) -> List[Dict]:
    """
    Sélectionne les N points d'ignition les plus risqués avec espacement minimum.

    Args:
        risk_array    : Grille de scores de risque (0-1)
        forest_mask   : Masque booléen des cellules forêt (None = toutes)
        n_points      : Nombre de points à sélectionner
        min_cell_dist : Distance minimum entre deux points (en cellules)
        lat_nw/lon_nw : Coin NW de la grille (pour conversion GPS)
        cell_size_m   : Taille d'une cellule en mètres

    Returns:
        Liste de dicts triée par risque décroissant :
        [{row, col, risk_score, lat, lon, rank, label}, ...]
    """
    rows, cols = risk_array.shape

    # Candidats : cellules forêt avec score > 0
    candidates = []
    for i in range(rows):
        for j in range(cols):
            if forest_mask is not None and not forest_mask[i, j]:
                continue
            score = float(risk_array[i, j])
            if score > 0.0:
                candidates.append((score, i, j))

    # Trier par score décroissant
    candidates.sort(reverse=True)

    # Sélection avec contrainte de distance
    selected = []
    for score, i, j in candidates:
        if len(selected) >= n_points:
            break
        # Vérifier distance minimum avec tous les points déjà sélectionnés
        too_close = False
        for pt in selected:
            dist = np.sqrt((i - pt["row"]) ** 2 + (j - pt["col"]) ** 2)
            if dist < min_cell_dist:
                too_close = True
                break
        if too_close:
            continue

        # Conversion GPS
        lat_m_per_deg = 111_000.0
        lon_m_per_deg = 111_000.0 * np.cos(np.radians(lat_nw))
        lat = lat_nw - (i + 0.5) * cell_size_m / lat_m_per_deg
        lon = lon_nw + (j + 0.5) * cell_size_m / lon_m_per_deg

        selected.append({
            "rank":       len(selected) + 1,
            "row":        int(i),
            "col":        int(j),
            "risk_score": round(score, 4),
            "lat":        round(lat, 6),
            "lon":        round(lon, 6),
            "label":      f"P{len(selected)+1}",
        })

    return selected


def risk_zones(risk_array: np.ndarray, forest_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Classifie la grille en zones de risque (adapté Assali et al. 2016).

    Returns:
        {"très_faible": %, "faible": %, "moyen": %, "élevé": %, "thresholds": [...]}
    """
    vals = risk_array.copy()
    if forest_mask is not None:
        vals = vals[forest_mask]
    vals = vals[vals > 0]

    if len(vals) == 0:
        return {}

    thresholds = [0.25, 0.45, 0.65]
    labels = ["très_faible", "faible", "moyen", "élevé"]
    bins = [0.0] + thresholds + [1.01]

    result = {"thresholds": thresholds}
    total = len(vals)
    for k, label in enumerate(labels):
        count = int(((vals >= bins[k]) & (vals < bins[k+1])).sum())
        result[label] = round(count / total * 100, 1)

    return result
