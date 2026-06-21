"""
robot_nav/robot_server.py
==========================
Serveur de communication Robot ↔ PC — BurnTrack.

Le Raspberry Pi du robot envoie du JSON via HTTP POST.
Ce serveur reçoit les données, met à jour la grille CA,
recalcule la carte de risque, et renvoie le prochain waypoint GPS.

Lancement (depuis la racine du repo) :
    python robot_nav/robot_server.py [--host 0.0.0.0] [--port 5000]

API :
    POST /telemetry      — données capteurs du robot → retourne prochain waypoint
    GET  /status         — état courant de la simulation
    GET  /waypoints      — liste complète des waypoints
    POST /ignite         — allumer un feu manuellement (test)
    POST /reset          — réinitialiser la simulation
    GET  /health         — ping

Format telemetry (JSON envoyé par le robot) :
    {
        "lat": 33.375,
        "lon": -7.655,
        "wind_speed_ms": 4.2,
        "wind_dir_deg": 265.0,
        "rh_percent": 28.0,
        "temp_c": 37.5,
        "moisture_1h": 0.05,
        "slope_pct": 3.0,
        "fuel_code": "AF_MAQUIS_SEC",   (optionnel — détecté par la caméra)
        "ndvi": 0.22,                    (optionnel — depuis image)
        "co_ppm": 12.0,                  (optionnel — capteur fumée)
        "timestamp": "2024-08-15T14:32:00"
    }

Réponse /telemetry :
    {
        "status": "ok",
        "next_waypoint": {"lat": 33.371, "lon": -7.648, "priority": 0.87, "label": "critique"},
        "mission_complete": false,
        "robot_position": [row, col],
        "fire_distance_min": 12.4,       (minutes avant que le feu arrive)
        "danger_level": "low"            ("low" / "medium" / "high" / "critical")
    }
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("ERREUR: Flask non installé. Lancez: pip install flask")
    sys.exit(1)

from cellular_automaton.grid import Grid, CellState
from cellular_automaton.rules import PropagationRules
from cellular_automaton.simulation import FireSimulation
from cellular_automaton.mlp_corrector import MLPCorrector
from robot_nav.planner import RobotNavigator, RiskMap
from robot_nav.waypoints import WaypointPlanner, GPSGrid
from burntrack.engine.rothermel import MoistureInputs
from burntrack.engine.fuel_models import ALL_FUEL_MODELS

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("robot_server")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Etat global de la simulation (partagé entre les requêtes)
# ---------------------------------------------------------------------------

class SimulationState:
    def __init__(self):
        self.grid: Optional[Grid] = None
        self.sim: Optional[FireSimulation] = None
        self.rules: Optional[PropagationRules] = None
        self.gps: Optional[GPSGrid] = None
        self.wp_planner: Optional[WaypointPlanner] = None
        self.navigator: Optional[RobotNavigator] = None
        self.risk_map: RiskMap = RiskMap()
        self.arrival_time: Dict = {}
        self.last_robot_pos: Optional[tuple] = None
        self.mission_complete: bool = False
        self.step_count: int = 0
        self.telemetry_log: list = []   # en mémoire avant flush sur disque

    def is_ready(self) -> bool:
        return self.grid is not None and self.sim is not None

STATE = SimulationState()

# ---------------------------------------------------------------------------
# Chemin du log telemetry
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent.parent / "data" / "robot_telemetry"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = LOG_DIR / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

def _append_telemetry(record: dict):
    """Sauvegarde un enregistrement telemetry sur disque (JSONL)."""
    STATE.telemetry_log.append(record)
    with open(_log_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ---------------------------------------------------------------------------
# Initialisation de la simulation
# ---------------------------------------------------------------------------

def init_simulation(
    rows: int = 50,
    cols: int = 50,
    cell_size: float = 30.0,
    fuel_code: str = "AF_MAQUIS_SEC",
    moisture_1h: float = 0.06,
    wind_speed_ms: float = 4.0,
    wind_dir_deg: float = 270.0,
    robot_start: tuple = (47, 2),
    n_waypoints: int = 20,
    model_path: str = "models/corrector_v3_best.pt",
    scaler_path: str = "models/mlp_v2_scaler.joblib",
):
    """Initialise la grille, la simulation et le navigateur."""
    fuel = fuel_code if fuel_code in ALL_FUEL_MODELS else "SH5"

    STATE.grid = Grid.uniform(
        rows, cols, cell_size=cell_size, fuel_code=fuel,
        moisture_1h=moisture_1h, wind_speed_ms=wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
    )

    # Coupe-feux type Bouskoura
    STATE.grid.add_firebreak(rows // 2, 0, rows // 2, cols - 1)
    STATE.grid.add_firebreak(0, cols // 3, rows - 1, cols // 3)
    STATE.grid.add_firebreak(0, 2 * cols // 3, rows - 1, 2 * cols // 3)

    # Correcteur MLP
    try:
        corrector = MLPCorrector(model_path=model_path, scaler_path=scaler_path)
        corrector.apply_to_grid(STATE.grid)
        log.info("Correcteur MLP chargé et appliqué.")
    except FileNotFoundError as e:
        log.warning(f"Correcteur MLP non trouvé ({e}) — Rothermel brut.")

    # GPS
    STATE.gps = GPSGrid.bouskoura(rows, cols, cell_size)

    # Waypoints depuis sécheresse synthétique (remplaçable par NDVI réel)
    STATE.wp_planner = WaypointPlanner.from_synthetic(
        STATE.grid, n_waypoints=n_waypoints, seed=42
    )
    STATE.wp_planner.greedy_tour(start=robot_start)

    # Règles + simulation
    STATE.rules = PropagationRules(stochastic=True, burn_duration_factor=4.0, min_burn_min=5.0)
    STATE.sim = FireSimulation(STATE.grid, rules=STATE.rules, seed=7)

    # Navigateur
    STATE.navigator = RobotNavigator(
        position=robot_start,
        safety_margin_min=8.0,
        replan_every=5,
    )
    STATE.navigator.set_waypoints(STATE.wp_planner)

    STATE.step_count = 0
    STATE.mission_complete = False
    STATE.last_robot_pos = robot_start
    log.info(f"Simulation initialisée : {rows}×{cols} @ {cell_size}m, robot en {robot_start}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ready": STATE.is_ready()})


@app.route("/status", methods=["GET"])
def status():
    if not STATE.is_ready():
        return jsonify({"error": "Simulation non initialisée"}), 503

    burning = STATE.grid.burning_count()
    burned_frac = STATE.grid.burned_fraction()
    robot_pos = STATE.navigator.position if STATE.navigator else None
    next_wp = None
    if STATE.wp_planner:
        wp = STATE.wp_planner.next_unvisited()
        if wp and STATE.gps:
            lat, lon = STATE.gps.cell_to_latlon(*wp.position)
            next_wp = {"lat": round(lat, 6), "lon": round(lon, 6),
                       "priority": round(wp.priority, 3), "label": wp.label}

    return jsonify({
        "time_min": round(STATE.sim.current_time, 1),
        "step": STATE.step_count,
        "burning_cells": burning,
        "burned_fraction": round(burned_frac, 4),
        "robot_position": list(robot_pos) if robot_pos else None,
        "next_waypoint": next_wp,
        "mission_complete": STATE.mission_complete,
        "waypoints_summary": STATE.wp_planner.summary() if STATE.wp_planner else "",
    })


@app.route("/waypoints", methods=["GET"])
def waypoints():
    if not STATE.wp_planner or not STATE.gps:
        return jsonify({"error": "Simulation non initialisée"}), 503

    wps = []
    for wp in STATE.wp_planner.waypoints:
        lat, lon = STATE.gps.cell_to_latlon(*wp.position)
        wps.append({
            "position": list(wp.position),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "priority": round(wp.priority, 3),
            "label": wp.label,
            "visited": wp.visited,
            "reachable": wp.reachable,
        })

    return jsonify({"waypoints": wps, "total": len(wps)})


@app.route("/ignite", methods=["POST"])
def ignite():
    """Allume un feu manuellement (test ou déclenchement réel)."""
    if not STATE.is_ready():
        return jsonify({"error": "Simulation non initialisée"}), 503

    data = request.get_json(silent=True) or {}
    row = data.get("row")
    col = data.get("col")

    # Depuis GPS
    if row is None and "lat" in data and "lon" in data and STATE.gps:
        row, col = STATE.gps.latlon_to_cell(data["lat"], data["lon"])
        if row == -1:
            return jsonify({"error": "Coordonnées GPS hors grille"}), 400

    if row is None or col is None:
        return jsonify({"error": "Fournir row/col ou lat/lon"}), 400

    STATE.sim.ignite(int(row), int(col))
    log.info(f"Feu allumé en ({row}, {col})")
    return jsonify({"status": "ok", "ignited_at": [row, col]})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    init_simulation(**{k: v for k, v in data.items() if k in [
        "rows", "cols", "cell_size", "fuel_code", "moisture_1h",
        "wind_speed_ms", "wind_dir_deg", "n_waypoints",
    ]})
    return jsonify({"status": "ok", "message": "Simulation réinitialisée"})


@app.route("/telemetry", methods=["POST"])
def telemetry():
    """
    Endpoint principal : reçoit les données du robot, met à jour la grille,
    fait avancer la simulation, et retourne le prochain waypoint.
    """
    if not STATE.is_ready():
        return jsonify({"error": "Simulation non initialisée. Appeler /reset d'abord."}), 503

    data: Dict[str, Any] = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON invalide ou manquant"}), 400

    # ---- 1. Convertir GPS → cellule ----
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat/lon requis"}), 400

    row, col = STATE.gps.latlon_to_cell(lat, lon)
    if row == -1:
        return jsonify({"error": f"Position GPS ({lat}, {lon}) hors de la grille"}), 400

    STATE.last_robot_pos = (row, col)

    # ---- 2. Mettre à jour la cellule avec les données terrain ----
    cell = STATE.grid.cells[row][col]

    if "wind_speed_ms" in data:
        cell.wind_speed_ms = float(data["wind_speed_ms"])
    if "wind_dir_deg" in data:
        cell.wind_dir_deg = float(data["wind_dir_deg"])
    if "rh_percent" in data:
        cell.rh_percent = float(data["rh_percent"])
    if "temp_c" in data:
        cell.temp_c = float(data["temp_c"])
    if "moisture_1h" in data:
        m = float(data["moisture_1h"])
        cell.moisture = MoistureInputs(
            m_1h=m, m_10h=m + 0.01, m_100h=m + 0.02,
            m_live_herb=min(m * 6, 1.0), m_live_woody=min(m * 8, 1.0),
        )
    if "slope_pct" in data:
        cell.slope_pct = float(data["slope_pct"])
    if "fuel_code" in data and data["fuel_code"] in ALL_FUEL_MODELS:
        cell.fuel_code = data["fuel_code"]

    # ---- 3. Sauvegarder telemetry ----
    record = {
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "lat": lat, "lon": lon,
        "row": row, "col": col,
        **{k: data[k] for k in [
            "wind_speed_ms", "wind_dir_deg", "rh_percent", "temp_c",
            "moisture_1h", "slope_pct", "fuel_code", "ndvi", "co_ppm"
        ] if k in data},
    }
    _append_telemetry(record)

    # ---- 4. Avancer la simulation d'un pas ----
    STATE.sim.step(dt=1.0)
    STATE.step_count += 1

    # ---- 5. Mettre à jour la carte de risque ----
    STATE.arrival_time = STATE.risk_map.build(STATE.grid, STATE.rules)

    # Filtrer les waypoints inaccessibles
    if STATE.wp_planner:
        STATE.wp_planner.filter_reachable(
            STATE.arrival_time, STATE.sim.current_time, safety_margin=8.0
        )

    # ---- 6. Faire bouger le navigateur ----
    nav_status = "idle"
    if STATE.navigator and not STATE.mission_complete:
        STATE.navigator.position = (row, col)   # sync position GPS
        nav_status = STATE.navigator.step(STATE.grid, STATE.rules, STATE.sim.current_time)
        if nav_status == "mission_complete":
            STATE.mission_complete = True

    # ---- 7. Prochain waypoint ----
    next_wp = None
    fire_distance_min = float("inf")
    danger_level = "low"

    if STATE.wp_planner and STATE.gps:
        wp = STATE.wp_planner.next_unvisited()
        if wp:
            lat_wp, lon_wp = STATE.gps.cell_to_latlon(*wp.position)
            next_wp = {
                "lat": round(lat_wp, 6),
                "lon": round(lon_wp, 6),
                "priority": round(wp.priority, 3),
                "label": wp.label,
            }

    # Distance feu au robot
    fire_distance_min = STATE.arrival_time.get((row, col), float("inf"))
    if fire_distance_min - STATE.sim.current_time < 5:
        danger_level = "critical"
    elif fire_distance_min - STATE.sim.current_time < 10:
        danger_level = "high"
    elif fire_distance_min - STATE.sim.current_time < 20:
        danger_level = "medium"
    else:
        danger_level = "low"

    return jsonify({
        "status": "ok",
        "next_waypoint": next_wp,
        "mission_complete": STATE.mission_complete,
        "nav_status": nav_status,
        "robot_position": [row, col],
        "fire_distance_min": round(fire_distance_min - STATE.sim.current_time, 1)
            if fire_distance_min != float("inf") else None,
        "danger_level": danger_level,
        "sim_time_min": round(STATE.sim.current_time, 1),
        "burning_cells": STATE.grid.burning_count(),
        "burned_fraction": round(STATE.grid.burned_fraction(), 4),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BurnTrack Robot Communication Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--cols", type=int, default=50)
    parser.add_argument("--cell-size", type=float, default=30.0)
    parser.add_argument("--fuel", default="AF_MAQUIS_SEC")
    args = parser.parse_args()

    init_simulation(
        rows=args.rows,
        cols=args.cols,
        cell_size=args.cell_size,
        fuel_code=args.fuel,
    )

    print("=" * 55)
    print("  BurnTrack — Robot Communication Server")
    print("=" * 55)
    print(f"  Endpoint telemetry : POST http://{args.host}:{args.port}/telemetry")
    print(f"  Statut             : GET  http://{args.host}:{args.port}/status")
    print(f"  Waypoints          : GET  http://{args.host}:{args.port}/waypoints")
    print(f"  Allumer feu        : POST http://{args.host}:{args.port}/ignite")
    print(f"  Reset              : POST http://{args.host}:{args.port}/reset")
    print(f"  Log telemetry      : {_log_file}")
    print("=" * 55)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
