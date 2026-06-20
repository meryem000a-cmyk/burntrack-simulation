"""
visualization/server.py
=======================
Serveur WebSocket pour la visualisation en temps reel de l'automate cellulaire BurnTrack.

Usage:
    python visualization/server.py [--host 0.0.0.0] [--port 8765]

Puis ouvrir http://localhost:8765 dans un navigateur.
"""

import asyncio
import json
import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional, Set
from scipy.interpolate import griddata

try:
    import websockets
except ImportError:
    print("ERREUR: 'websockets' n'est pas installe. Lancez: pip install websockets")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cellular_automaton.grid import Grid, Cell, CellState
from cellular_automaton.rules import PropagationRules, _to_roth_fuel, _max_spread_direction, _angular_diff
from cellular_automaton.simulation import FireSimulation
from burntrack.engine.rothermel import RothermelEngine, EnvironmentalConditions, RothermelOutput
from burntrack.engine.fuel_models import ALL_FUEL_MODELS


class SimulationServer:
    """Serveur WebSocket gerant les simulations CA en temps reel."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.grid: Optional[Grid] = None
        self.sim: Optional[FireSimulation] = None
        self.rules: Optional[PropagationRules] = None
        self.running = False
        self.speed = 1.0
        self.dt = 1.0
        self.max_steps = 300
        self.step_count = 0
        self.ws_clients: Set = set()
        self.stats_history: Dict = {
            "time": [], "burning": [], "fraction": [], "ignitions": [],
            "ros_avg": [], "intensity_avg": [], "flame_avg": [],
        }

    # ------------------------------------------------------------------
    # Configuration de la grille
    # ------------------------------------------------------------------

    def configure_grid(self, params: dict) -> dict:
        rows = params.get("rows", 50)
        cols = params.get("cols", 50)
        cell_size = params.get("cell_size_m", 30.0)
        fuel_code = params.get("fuel_code", "GR4")
        moisture_1h = params.get("moisture_1h", 0.06)
        wind_speed_ms = params.get("wind_speed_ms", 5.0)
        wind_dir_deg = params.get("wind_dir_deg", 270.0)
        slope_pct = params.get("slope_pct", 0.0)
        aspect_deg = params.get("aspect_deg", 0.0)

        self.grid = Grid.uniform(
            rows, cols, cell_size, fuel_code, moisture_1h,
            wind_speed_ms, wind_dir_deg, slope_pct, aspect_deg,
        )
        self.rules = PropagationRules(stochastic=True)
        self.sim = FireSimulation(self.grid, rules=self.rules, seed=42)
        self.step_count = 0
        self.stats_history = {
            "time": [], "burning": [], "fraction": [], "ignitions": [],
            "ros_avg": [], "intensity_avg": [], "flame_avg": [],
        }
        return {"status": "ok", "rows": rows, "cols": cols, "cell_size_m": cell_size}

    def configure_from_real_data(self, params: dict) -> dict:
        """Configure la grille a partir de donnees reelles (CSV ground truth)."""
        import pandas as pd

        region = params.get("region", "kruger")
        n_rows = params.get("rows", 40)
        n_cols = params.get("cols", 40)
        cell_size = params.get("cell_size_m", 10.0)

        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "african_ground_truth.csv")
        if not os.path.exists(csv_path):
            return {"status": "error", "message": f"Fichier non trouve: {csv_path}"}

        df = pd.read_csv(csv_path)
        if region != "all":
            df = df[df["region"] == region]

        if len(df) < 10:
            return {"status": "error", "message": f"Pas assez de donnees pour la region '{region}' ({len(df)} points)"}

        # Centrer la grille sur les donnees
        lat_center = df["latitude"].mean()
        lon_center = df["longitude"].mean()
        lat_range = df["latitude"].max() - df["latitude"].min()
        lon_range = df["longitude"].max() - df["longitude"].min()

        # Creer les axes de la grille
        lat_min = lat_center - lat_range * 0.6
        lat_max = lat_center + lat_range * 0.6
        lon_min = lon_center - lon_range * 0.6
        lon_max = lon_center + lon_range * 0.6

        grid_lats = np.linspace(lat_min, lat_max, n_rows)
        grid_lons = np.linspace(lon_min, lon_max, n_cols)

        # Extraire les points d'observation
        obs_lats = df["latitude"].values
        obs_lons = df["longitude"].values

        # Interpoler les donnees environnementales
        def interp_field(col, default=0.0):
            if col in df.columns:
                vals = df[col].values
                return griddata(
                    np.column_stack([obs_lats, obs_lons]),
                    vals,
                    np.column_stack([np.repeat(grid_lats, n_cols), np.tile(grid_lons, n_rows)]).reshape(-1, 2),
                    method='linear',
                    fill_value=default,
                ).reshape(n_rows, n_cols)
            return np.full((n_rows, n_cols), default)

        slope_grid = interp_field("slope_pct", 0.0)
        wind_speed_grid = interp_field("wind_speed_ms", 3.0)
        wind_dir_grid = interp_field("wind_dir", 270.0)
        elevation_grid = interp_field("elevation_m", 500.0)
        moisture_grid = interp_field("m_1h", 0.06)
        aspect_grid = interp_field("slope_aspect_deg", 0.0)
        rh_grid = interp_field("rh_percent", 40.0)
        temp_grid = interp_field("temp_c", 25.0)

        # Determiner le carburant dominant par zone
        fuel_codes = df["fuel_model_code"].values
        unique_fuels = list(set(fuel_codes))
        fuel_grid = np.full((n_rows, n_cols), unique_fuels[0] if unique_fuels else "AF_MIOMBO", dtype=object)

        # Assigner le carburant le plus proche a chaque cellule
        from scipy.spatial import cKDTree
        obs_points = np.column_stack([obs_lats, obs_lons])
        tree = cKDTree(obs_points)
        grid_points = np.column_stack([np.repeat(grid_lats, n_cols), np.tile(grid_lons, n_rows)])
        _, idxs = tree.query(grid_points)
        fuel_grid = fuel_codes[idxs].reshape(n_rows, n_cols)

        # Creer la grille heterogene
        self.grid = Grid(n_rows, n_cols, cell_size)
        self._real_lat_axis = grid_lats.tolist()
        self._real_lon_axis = grid_lons.tolist()
        self._real_region = region

        for i in range(n_rows):
            for j in range(n_cols):
                c = self.grid.cells[i][j]
                c.fuel_code = str(fuel_grid[i, j])
                c.slope_pct = float(np.clip(slope_grid[i, j], 0, 80))
                c.aspect_deg = float(aspect_grid[i, j])
                c.elevation_m = float(elevation_grid[i, j])
                c.wind_speed_ms = float(np.clip(wind_speed_grid[i, j], 0, 30))
                c.wind_dir_deg = float(wind_dir_grid[i, j])
                c.rh_percent = float(rh_grid[i, j])
                c.temp_c = float(temp_grid[i, j])
                m1h = float(np.clip(moisture_grid[i, j], 0.01, 0.5))
                from burntrack.engine.rothermel import MoistureInputs
                c.moisture = MoistureInputs(
                    m_1h=m1h, m_10h=m1h + 0.01, m_100h=m1h + 0.02,
                    m_live_herb=min(m1h * 6, 1.0), m_live_woody=min(m1h * 8, 1.0),
                )

        self.rules = PropagationRules(stochastic=True)
        self.sim = FireSimulation(self.grid, rules=self.rules, seed=42)
        self.step_count = 0
        self.stats_history = {
            "time": [], "burning": [], "fraction": [], "ignitions": [],
            "ros_avg": [], "intensity_avg": [], "flame_avg": [],
        }

        # Calculer les stats du terrain
        avg_slope = float(np.mean(slope_grid))
        avg_wind = float(np.mean(wind_speed_grid))
        avg_moisture = float(np.mean(moisture_grid))

        return {
            "status": "ok",
            "rows": n_rows, "cols": n_cols, "cell_size_m": cell_size,
            "region": region,
            "center_lat": round(lat_center, 4),
            "center_lon": round(lon_center, 4),
            "avg_slope_pct": round(avg_slope, 1),
            "avg_wind_ms": round(avg_wind, 1),
            "avg_moisture": round(avg_moisture, 3),
            "fuel_models": unique_fuels[:10],
            "n_observations": len(df),
        }

    # ------------------------------------------------------------------
    # Generation des donnees par cellule
    # ------------------------------------------------------------------

    def _compute_ros_grid(self) -> List[List[float]]:
        """Calcule le ROS maximal pour chaque cellule (pour heatmap)."""
        if self.grid is None:
            return []
        rows, cols = self.grid.rows, self.grid.cols
        ros_grid = [[0.0] * cols for _ in range(rows)]
        engine = RothermelEngine()

        for i in range(rows):
            for j in range(cols):
                c = self.grid.cells[i][j]
                if c.state != CellState.BURNING:
                    continue
                fm_raw = self.grid.get_fuel(c.fuel_code)
                if fm_raw is None:
                    continue
                fuel = _to_roth_fuel(fm_raw)
                wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                conditions = EnvironmentalConditions(
                    wind_speed=c.wind_speed_ms,
                    slope_pct=c.slope_pct,
                    angle_wind_slope=angle_wind_slope,
                )
                try:
                    output = engine.compute(fuel, c.moisture, conditions)
                    ros_eff = max(output.ros + getattr(c, 'delta_ros', 0.0), 0.0)
                    max_dir = _max_spread_direction(c.wind_dir_deg, c.slope_pct, c.aspect_deg)
                    # ROS max (direction du vent+pente)
                    ros_grid[i][j] = round(ros_eff, 3)
                except Exception:
                    ros_grid[i][j] = 0.0
        return ros_grid

    def _compute_intensity_grid(self) -> List[List[float]]:
        """Calcule l'intensite de Byram pour chaque cellule BURNING."""
        if self.grid is None:
            return []
        rows, cols = self.grid.rows, self.grid.cols
        grid = [[0.0] * cols for _ in range(rows)]
        engine = RothermelEngine()

        for i in range(rows):
            for j in range(cols):
                c = self.grid.cells[i][j]
                if c.state != CellState.BURNING:
                    continue
                fm_raw = self.grid.get_fuel(c.fuel_code)
                if fm_raw is None:
                    continue
                fuel = _to_roth_fuel(fm_raw)
                wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                conditions = EnvironmentalConditions(
                    wind_speed=c.wind_speed_ms,
                    slope_pct=c.slope_pct,
                    angle_wind_slope=angle_wind_slope,
                )
                try:
                    output = engine.compute(fuel, c.moisture, conditions)
                    grid[i][j] = round(output.fireline_intensity, 2)
                except Exception:
                    grid[i][j] = 0.0
        return grid

    def _compute_flame_grid(self) -> List[List[float]]:
        """Calcule la longueur de flamme pour chaque cellule BURNING."""
        if self.grid is None:
            return []
        rows, cols = self.grid.rows, self.grid.cols
        grid = [[0.0] * cols for _ in range(rows)]
        engine = RothermelEngine()

        for i in range(rows):
            for j in range(cols):
                c = self.grid.cells[i][j]
                if c.state != CellState.BURNING:
                    continue
                fm_raw = self.grid.get_fuel(c.fuel_code)
                if fm_raw is None:
                    continue
                fuel = _to_roth_fuel(fm_raw)
                wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                conditions = EnvironmentalConditions(
                    wind_speed=c.wind_speed_ms,
                    slope_pct=c.slope_pct,
                    angle_wind_slope=angle_wind_slope,
                )
                try:
                    output = engine.compute(fuel, c.moisture, conditions)
                    grid[i][j] = round(output.flame_length, 3)
                except Exception:
                    grid[i][j] = 0.0
        return grid

    def _compute_env_grids(self) -> dict:
        """Retourne les grilles environnementales pour les overlays."""
        if self.grid is None:
            return {}
        rows, cols = self.grid.rows, self.grid.cols
        return {
            "slope_pct": [[round(self.grid.cells[i][j].slope_pct, 1) for j in range(cols)] for i in range(rows)],
            "wind_speed": [[round(self.grid.cells[i][j].wind_speed_ms, 1) for j in range(cols)] for i in range(rows)],
            "wind_dir": [[round(self.grid.cells[i][j].wind_dir_deg, 0) for j in range(cols)] for i in range(rows)],
            "moisture_1h": [[round(self.grid.cells[i][j].moisture.m_1h, 3) for j in range(cols)] for i in range(rows)],
            "elevation_m": [[round(self.grid.cells[i][j].elevation_m, 1) for j in range(cols)] for i in range(rows)],
            "fuel_codes": [[self.grid.cells[i][j].fuel_code for j in range(cols)] for i in range(rows)],
        }

    def _compute_risk_grids(self) -> dict:
        """Calcule les cartes de risque pour chaque cellule."""
        if self.grid is None:
            return {}
        rows, cols = self.grid.rows, self.grid.cols
        engine = RothermelEngine()

        fire_risk = [[0.0] * cols for _ in range(rows)]
        terrain_risk = [[0.0] * cols for _ in range(rows)]
        fuel_risk = [[0.0] * cols for _ in range(rows)]
        wind_risk = [[0.0] * cols for _ in range(rows)]
        moisture_risk = [[0.0] * cols for _ in range(rows)]
        combined_risk = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                c = self.grid.cells[i][j]
                if c.state == 3:  # FIREBREAK
                    continue

                # === TERRAIN RISK (pente + aspect) ===
                # Pente forte = risque eleve (le feu monte plus vite)
                slope_factor = min(c.slope_pct / 50.0, 1.0)
                # Aspect sud = plus seche + plus de soleil = risque plus eleve
                aspect_factor = max(0, np.cos(np.radians(c.aspect_deg - 180))) * 0.3
                terrain_risk[i][j] = round(min(1.0, slope_factor * 0.7 + aspect_factor), 3)

                # === FUEL RISK (carburant) ===
                fm_raw = self.grid.get_fuel(c.fuel_code)
                if fm_raw is not None:
                    # Charge de carburant totale
                    load_factor = min(fm_raw.w_total / 5.0, 1.0)
                    # Profondeur du lit (plus profond = plus de carburant)
                    depth_factor = min(fm_raw.delta / 1.0, 1.0)
                    # SAV (surface area volume ratio) - herbes fines = propagation rapide
                    sav_factor = min(fm_raw.sigma_1h / 5000.0, 1.0)
                    fuel_risk[i][j] = round(min(1.0, load_factor * 0.4 + depth_factor * 0.3 + sav_factor * 0.3), 3)

                # === WIND RISK (vent) ===
                wind_factor = min(c.wind_speed_ms / 15.0, 1.0)
                wind_risk[i][j] = round(wind_factor, 3)

                # === MOISTURE RISK (humidite - inversé: plus sec = plus risque) ===
                m1h = c.moisture.m_1h
                # 0% = risque max, 50% = risque min
                moisture_factor = max(0, 1.0 - m1h * 2.0)
                moisture_risk[i][j] = round(moisture_factor, 3)

                # === FIRE RISK (ROS Rothermel) ===
                if fm_raw is not None:
                    try:
                        fuel = _to_roth_fuel(fm_raw)
                        wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                        angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                        conditions = EnvironmentalConditions(
                            wind_speed=c.wind_speed_ms,
                            slope_pct=c.slope_pct,
                            angle_wind_slope=angle_wind_slope,
                        )
                        output = engine.compute(fuel, c.moisture, conditions)
                        ros = max(output.ros + getattr(c, 'delta_ros', 0.0), 0.0)
                        # Normaliser le ROS (0-15 m/min = risque 0-1)
                        fire_risk[i][j] = round(min(1.0, ros / 15.0), 3)
                    except Exception:
                        fire_risk[i][j] = 0.0

                # === COMBINED RISK (score composite) ===
                # Poids: terrain 20%, fuel 25%, wind 20%, moisture 15%, fire 20%
                combined_risk[i][j] = round(
                    terrain_risk[i][j] * 0.20 +
                    fuel_risk[i][j] * 0.25 +
                    wind_risk[i][j] * 0.20 +
                    moisture_risk[i][j] * 0.15 +
                    fire_risk[i][j] * 0.20,
                    3
                )

        return {
            "fire_risk": fire_risk,
            "terrain_risk": terrain_risk,
            "fuel_risk": fuel_risk,
            "wind_risk": wind_risk,
            "moisture_risk": moisture_risk,
            "combined_risk": combined_risk,
        }

    def _cell_detail(self, row: int, col: int) -> dict:
        """Details complets d'une cellule."""
        if self.grid is None:
            return {}
        c = self.grid.cells[row][col]
        fm_raw = self.grid.get_fuel(c.fuel_code)
        data = {
            "row": row, "col": col,
            "state": c.state.name,
            "fuel_code": c.fuel_code,
            "slope_pct": round(c.slope_pct, 1),
            "aspect_deg": round(c.aspect_deg, 1),
            "elevation_m": round(c.elevation_m, 1),
            "wind_speed_ms": round(c.wind_speed_ms, 2),
            "wind_dir_deg": round(c.wind_dir_deg, 1),
            "moisture_1h": round(c.moisture.m_1h, 3),
            "moisture_10h": round(c.moisture.m_10h, 3),
            "moisture_100h": round(c.moisture.m_100h, 3),
            "moisture_live_herb": round(c.moisture.m_live_herb, 3),
            "moisture_live_woody": round(c.moisture.m_live_woody, 3),
            "rh_percent": round(c.rh_percent, 1),
            "temp_c": round(c.temp_c, 1),
            "ignition_time": round(c.ignition_time, 2) if c.ignition_time is not None else None,
            "burn_duration": round(c.burn_duration, 2),
            "burn_elapsed": round(c.burn_elapsed, 2),
            "delta_ros": round(c.delta_ros, 4),
        }
        if fm_raw is not None:
            data["fuel_name"] = fm_raw.name
            data["fuel_w_total"] = round(fm_raw.w_total, 4)
            data["fuel_w_dead"] = round(fm_raw.w_dead, 4)
            data["fuel_w_live"] = round(fm_raw.w_live, 4)
            data["fuel_delta_m"] = round(fm_raw.delta, 3)
            data["fuel_sigma"] = round(fm_raw.sigma_1h, 1)
            data["fuel_mx"] = round(fm_raw.mx, 1)

        # Calculer le ROS et l'intensite pour les cellules en feu
        if c.state == CellState.BURNING and fm_raw is not None:
            try:
                engine = RothermelEngine()
                fuel = _to_roth_fuel(fm_raw)
                wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                conditions = EnvironmentalConditions(
                    wind_speed=c.wind_speed_ms,
                    slope_pct=c.slope_pct,
                    angle_wind_slope=angle_wind_slope,
                )
                output = engine.compute(fuel, c.moisture, conditions)
                data["ros_rothermel"] = round(output.ros, 3)
                data["fireline_intensity"] = round(output.fireline_intensity, 2)
                data["flame_length"] = round(output.flame_length, 3)
                data["reaction_intensity"] = round(output.reaction_intensity, 2)
                data["residence_time"] = round(output.residence_time, 3)
                data["fuel_consumption"] = round(output.fuel_consumption, 4)
                data["phi_w"] = round(output.phi_w, 4)
                data["phi_s"] = round(output.phi_s, 4)
                data["phi_eff"] = round(output.phi_eff, 4)
                data["heat_per_unit_area"] = round(output.heat_per_unit_area, 2)
                data["spread_direction"] = round(output.spread_direction, 1)
            except Exception:
                pass

        return data

    def _build_frame(self) -> dict:
        """Construit une frame complete a envoyer au client."""
        grid_arr = self.sim.snapshot().tolist()

        # Stats courantes
        burning = self.grid.burning_count() if self.grid else 0
        fraction = self.grid.burned_fraction() if self.grid else 0.0
        new_ign = self.sim.stats.new_ignitions[-1] if self.sim.stats.new_ignitions else 0
        ros_avg = 0.0
        intensity_avg = 0.0
        flame_avg = 0.0

        # Calculer les moyennes pour les cellules en feu
        burning_cells = []
        if self.grid:
            for i in range(self.grid.rows):
                for j in range(self.grid.cols):
                    if self.grid.cells[i][j].state == CellState.BURNING:
                        burning_cells.append(self.grid.cells[i][j])

        if burning_cells:
            engine = RothermelEngine()
            ros_vals, int_vals, fl_vals = [], [], []
            for c in burning_cells[:50]:  # Limiter le calcul pour la performance
                fm_raw = self.grid.get_fuel(c.fuel_code)
                if fm_raw is None:
                    continue
                try:
                    fuel = _to_roth_fuel(fm_raw)
                    wind_prop_dir = (c.wind_dir_deg + 180.0) % 360.0
                    angle_wind_slope = _angular_diff(wind_prop_dir, c.aspect_deg)
                    conditions = EnvironmentalConditions(
                        wind_speed=c.wind_speed_ms, slope_pct=c.slope_pct,
                        angle_wind_slope=angle_wind_slope,
                    )
                    output = engine.compute(fuel, c.moisture, conditions)
                    ros_vals.append(output.ros)
                    int_vals.append(output.fireline_intensity)
                    fl_vals.append(output.flame_length)
                except Exception:
                    pass
            if ros_vals:
                ros_avg = round(float(np.mean(ros_vals)), 3)
                intensity_avg = round(float(np.mean(int_vals)), 2)
                flame_avg = round(float(np.mean(fl_vals)), 3)

        # Enregistrer l'historique
        self.stats_history["time"].append(round(self.sim.current_time, 1))
        self.stats_history["burning"].append(burning)
        self.stats_history["fraction"].append(round(fraction, 5))
        self.stats_history["ignitions"].append(new_ign)
        self.stats_history["ros_avg"].append(ros_avg)
        self.stats_history["intensity_avg"].append(intensity_avg)
        self.stats_history["flame_avg"].append(flame_avg)

        return {
            "type": "frame",
            "time_min": round(self.sim.current_time, 1),
            "step": self.sim.step_count,
            "grid": grid_arr,
            "ros_grid": self._compute_ros_grid(),
            "intensity_grid": self._compute_intensity_grid(),
            "flame_length_grid": self._compute_flame_grid(),
            "stats": {
                "burning": burning,
                "burned_frac": round(fraction, 5),
                "new_ignitions": new_ign,
                "ros_avg": ros_avg,
                "intensity_avg": intensity_avg,
                "flame_avg": flame_avg,
            },
        }

    # ------------------------------------------------------------------
    # Boucle de simulation
    # ------------------------------------------------------------------

     async def simulation_loop(self):
        """Boucle principale : avance la simulation et broadcast les frames."""
        if self.grid and self.grid.burning_count() == 0:
            self.running = False
            await self._broadcast({"type": "simulation_end", "reason": "extinct"})
            return
        while self.running and self.sim:
            if self.grid and self.grid.burning_count() == 0:
                self.running = False
                await self._broadcast({"type": "simulation_end", "reason": "extinct"})
                break
            if self.step_count >= self.max_steps:
                self.running = False
                await self._broadcast({"type": "simulation_end", "reason": "max_steps"})
                break
            # Executer un pas
            n_new = self.sim.step(self.dt)
            self.step_count += 1
            # Envoyer la frame
            frame = self._build_frame()
            await self._broadcast(frame)
            # Envoyer l'historique complet periodiquement
            if self.step_count % 5 == 0 or self.step_count == 1:
                await self._broadcast({
                    "type": "stats_history",
                    **self.stats_history,
                })
            # Attendre selon la vitesse
            delay = max(0.02, 0.5 / self.speed)
            await asyncio.sleep(delay)
        # Envoyer les stats finales
        if self.sim:
            await self._broadcast({
                "type": "stats_history",
                **self.stats_history,
            })
            await self._broadcast({
                "type": "simulation_end",
                "reason": "complete",
                "summary": self.sim.stats.summary(),
            })

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    async def _broadcast(self, msg: dict):
        if not self.ws_clients:
            return
        data = json.dumps(msg, default=str)
        await asyncio.gather(
            *[ws.send(data) for ws in self.ws_clients],
            return_exceptions=True,
        )

    async def handler(self, websocket):
        """Handler principal pour une connexion WebSocket."""
        self.ws_clients.add(websocket)
        print(f"[WS] Client connecte ({len(self.ws_clients)} total)")

        # Envoyer l'etat courant
        await websocket.send(json.dumps({
            "type": "ready",
            "fuel_models": list(ALL_FUEL_MODELS.keys()),
            "grid": {
                "rows": self.grid.rows if self.grid else 50,
                "cols": self.grid.cols if self.grid else 50,
                "cell_size_m": self.grid.cell_size if self.grid else 30.0,
            } if self.grid else None,
        }))

        try:
            async for message in websocket:
                msg = json.loads(message)
                await self._process_command(websocket, msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ws_clients.discard(websocket)
            print(f"[WS] Client deconnecte ({len(self.ws_clients)} total)")

    async def _process_command(self, websocket, msg: dict):
        cmd = msg.get("cmd", "")

        if cmd == "configure":
            # Arreter la simulation en cours
            self.running = False
            result = self.configure_grid(msg)
            await websocket.send(json.dumps({"type": "configured", **result}))

            # Envoyer la frame initiale
            if self.sim:
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))
                await websocket.send(json.dumps({
                    "type": "env_grids",
                    **self._compute_env_grids(),
                }, default=str))

        elif cmd == "ignite":
            row, col = msg.get("row", 25), msg.get("col", 25)
            if self.sim:
                self.sim.ignite(row, col)
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))

        elif cmd == "ignite_multiple":
            points = msg.get("points", [])
            if self.sim and points:
                for p in points:
                    self.sim.ignite(p["row"], p["col"])
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))

        elif cmd == "play":
            if self.sim and not self.running:
                self.running = True
                self.speed = msg.get("speed", 1.0)
                asyncio.create_task(self.simulation_loop())

        elif cmd == "pause":
            self.running = False

        elif cmd == "step":
            if self.sim:
                n_new = self.sim.step(self.dt)
                self.step_count += 1
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))

        elif cmd == "reset":
            if self.sim:
                self.running = False
                self.sim.reset()
                self.step_count = 0
                self.stats_history = {
                    "time": [], "burning": [], "fraction": [], "ignitions": [],
                    "ros_avg": [], "intensity_avg": [], "flame_avg": [],
                }
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))

        elif cmd == "set_speed":
            self.speed = msg.get("speed", 1.0)

        elif cmd == "set_dt":
            self.dt = msg.get("dt", 1.0)

        elif cmd == "set_max_steps":
            self.max_steps = msg.get("max_steps", 300)

        elif cmd == "query_cell":
            row, col = msg.get("row", 0), msg.get("col", 0)
            if self.grid and 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
                detail = self._cell_detail(row, col)
                await websocket.send(json.dumps({
                    "type": "cell_detail", **detail,
                }, default=str))

        elif cmd == "load_real_data":
            self.running = False
            result = self.configure_from_real_data(msg)
            await websocket.send(json.dumps({"type": "configured", **result}, default=str))
            if result["status"] == "ok" and self.sim:
                frame = self._build_frame()
                await websocket.send(json.dumps(frame, default=str))
                env_data = self._compute_env_grids()
                risk_data = self._compute_risk_grids()
                await websocket.send(json.dumps({
                    "type": "env_grids",
                    **env_data,
                    **risk_data,
                }, default=str))

        elif cmd == "get_env":
            env_data = self._compute_env_grids()
            risk_data = self._compute_risk_grids()
            await websocket.send(json.dumps({
                "type": "env_grids",
                **env_data,
                **risk_data,
            }, default=str))

        elif cmd == "get_risk":
            risk_data = self._compute_risk_grids()
            await websocket.send(json.dumps({
                "type": "risk_grids",
                **risk_data,
            }, default=str))

        elif cmd == "get_stats":
            await websocket.send(json.dumps({
                "type": "stats_history",
                **self.stats_history,
            }))

        elif cmd == "get_real_regions":
            import pandas as pd
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "african_ground_truth.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                regions = df["region"].value_counts().to_dict()
                await websocket.send(json.dumps({
                    "type": "real_regions",
                    "regions": regions,
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "real_regions",
                    "regions": {},
                }))


def main():
    parser = argparse.ArgumentParser(description="BurnTrack WebSocket Visualization Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    args = parser.parse_args()

    server = SimulationServer(host=args.host, port=args.port)

    # Configurer une grille par defaut
    server.configure_grid({
        "rows": 50, "cols": 50, "cell_size_m": 30.0,
        "fuel_code": "GR4", "moisture_1h": 0.06,
        "wind_speed_ms": 5.0, "wind_dir_deg": 270.0, "slope_pct": 0.0,
    })

    print(f"=== BurnTrack Visualization Server ===")
    print(f"WebSocket: ws://{args.host}:{args.port}")
    print(f"HTTP:      http://localhost:{args.port}")
    print(f"Grille:    {server.grid.rows}x{server.grid.cols} ({server.grid.cell_size}m)")
    print(f"Fuel:      GR4 | Vent: 5.0 m/s Ouest | Humidite: 6%")
    print()

    # Servir aussi le fichier HTML via HTTP
    html_path = os.path.join(os.path.dirname(__file__), "index.html")

    async def serve_http(reader, writer):
        try:
            data = await reader.read(4096)
            request_line = data.decode().split('\r\n')[0]
            method, path, _ = request_line.split(' ', 2)

            if path == '/' or path == '/index.html':
                content_type = 'text/html; charset=utf-8'
                with open(html_path, 'rb') as f:
                    body = f.read()
            elif path == '/favicon.ico':
                writer.write(b"HTTP/1.1 204 No Content\r\n\r\n")
                await writer.drain()
                writer.close()
                return
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(response)
            await writer.drain()
            writer.close()
        except Exception as e:
            try:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                await writer.drain()
                writer.close()
            except Exception:
                pass

    async def run_servers():
        # WebSocket server
        ws_server = await websockets.serve(server.handler, args.host, args.port)
        print(f"WebSocket actif sur ws://{args.host}:{args.port}")

        # HTTP server sur port+1
        http_port = args.port + 1
        http_server = await asyncio.start_server(serve_http, args.host, http_port)
        print(f"HTML accessible sur http://localhost:{http_port}")
        print()
        print("Ouvrez le navigateur sur http://localhost:8766")

        await asyncio.gather(
            ws_server.wait_closed(),
            http_server.serve_forever(),
        )

    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        print("\nArret du serveur.")


if __name__ == "__main__":
    main()
