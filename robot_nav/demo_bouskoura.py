"""
demo_bouskoura.py
==================
Demo complete du pipeline BurnTrack sur terrain type Bouskoura.

Lance depuis la racine du repo :
    python robot_nav/demo_bouskoura.py

Ce que ca fait :
  1. Grille 50x50 (1.5km x 1.5km a 30m/cellule)
  2. Chemins forestiers de Bouskoura -> FIREBREAK
  3. Carte NDVI synthetique -> zone NE plus seche = prioritaire
  4. Feu allume coin Nord-Ouest (vent d Ouest)
  5. Robot demarre coin Sud-Ouest, visite les 20 waypoints les plus secs
     en evitant le feu via D* Lite + re-planification toutes les 5 etapes
  6. Affiche la couverture finale et les waypoints visites
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cellular_automaton import Grid, FireSimulation, PropagationRules
from burntrack.utils.simulation import create_rules, run_adaptive
from robot_nav import RobotNavigator, WaypointPlanner, GPSGrid
from rothermel.fuel_models import ALL_FUEL_MODELS

# ---------------------------------------------------------------------------
# 1. Grille type Bouskoura
# ---------------------------------------------------------------------------
ROWS, COLS = 50, 50
CELL_SIZE  = 30.0   # metres

# Fuel : maquis sec nord-africain si disponible, sinon SH5 (arbustes)
FUEL = "AF_MAQUIS_SEC" if "AF_MAQUIS_SEC" in ALL_FUEL_MODELS else "SH5"
print(f"Fuel model : {FUEL}")

grid = Grid.uniform(ROWS, COLS, cell_size=CELL_SIZE, fuel_code=FUEL,
                    moisture_1h=0.06, wind_speed_ms=4.0, wind_dir_deg=270.0)

# Chemins forestiers (axes principaux de Bouskoura)
grid.add_firebreak(25, 0,  25, COLS-1)   # route horizontale centrale
grid.add_firebreak(0,  16, ROWS-1, 16)   # piste verticale Ouest
grid.add_firebreak(0,  33, ROWS-1, 33)   # piste verticale Est

print(f"Grille : {ROWS}x{COLS} ({ROWS*CELL_SIZE/1000:.1f}km x {COLS*CELL_SIZE/1000:.1f}km)")
print(f"Coupe-feux : ligne 25 horizontale + colonnes 16 et 33 verticales")

# ---------------------------------------------------------------------------
# 2. GPS Bouskoura
# ---------------------------------------------------------------------------
gps = GPSGrid.bouskoura(ROWS, COLS, CELL_SIZE)
print(f"\n{gps}")

# ---------------------------------------------------------------------------
# 3. NDVI synthetique (zone NE plus seche)
# ---------------------------------------------------------------------------
np.random.seed(42)
r_idx, c_idx = np.mgrid[0:ROWS, 0:COLS]
# Gradient : coin NE plus sec (faible NDVI)
ndvi = 0.7 - 0.4 * (r_idx / ROWS) - 0.2 * ((COLS - c_idx) / COLS)
ndvi += np.random.randn(ROWS, COLS) * 0.08
ndvi = np.clip(ndvi, -1.0, 1.0)
print(f"\nNDVI : min={ndvi.min():.2f}, max={ndvi.max():.2f}, mean={ndvi.mean():.2f}")

# ---------------------------------------------------------------------------
# 4. Waypoints depuis NDVI
# ---------------------------------------------------------------------------
wp_planner = WaypointPlanner.from_ndvi(ndvi, grid, n_waypoints=20, min_spacing=4)
START_POS  = (ROWS-3, 2)                # coin Sud-Ouest
wp_planner.greedy_tour(start=START_POS)

print(f"\n{len(wp_planner.waypoints)} waypoints selectionnes (zones les plus seches) :")
for i, wp in enumerate(wp_planner.waypoints[:5]):
    lat, lon = gps.cell_to_latlon(*wp.position)
    print(f"  {i+1}. {wp.position} | priorite={wp.priority:.2f} | {wp.label} | "
          f"GPS=({lat:.4f}N, {lon:.4f}W)")
print(f"  ... ({len(wp_planner.waypoints)-5} autres)")

# ---------------------------------------------------------------------------
# 5. Simulation feu + robot
# ---------------------------------------------------------------------------
rules = create_rules(stochastic=True, burn_duration_factor=4.0, min_burn_min=5.0)
sim   = FireSimulation(grid, rules, seed=7)
sim.ignite(2, 2)   # coin Nord-Ouest -- vent pousse vers l Est

robot = RobotNavigator(
    position=START_POS,
    safety_margin_min=8.0,
    replan_every=5,
)
robot.set_waypoints(wp_planner)

print(f"\nRobot depart : {robot.position}")
print(f"Premier objectif : {robot.goal}")
print(f"\n{'t(min)':>7} | {'feu':>5} | {'brule':>6} | {'robot':>10} | {'status':<20} | {'couverture'}")
print("-" * 75)

t0 = time.time()
for step in range(300):
    run_adaptive(sim, grid)

    # Robot se deplace toutes les 2 min (vitesse ~15 m/min = rover lent)
    if step % 2 == 0:
        status = robot.step(grid, rules, sim.current_time)

        if step % 20 == 0 or status != "navigating":
            print(f"{sim.current_time:7.1f} | "
                  f"{grid.burning_count():5d} | "
                  f"{grid.burned_fraction()*100:5.1f}% | "
                  f"{str(robot.position):>10} | "
                  f"{status:<20} | "
                  f"{wp_planner.summary()}")

        if status not in ("navigating",):
            break

# ---------------------------------------------------------------------------
# 6. Resultats
# ---------------------------------------------------------------------------
wall = time.time() - t0
print(f"\n{'='*60}")
print(f"Simulation terminee en {sim.current_time:.1f} min ({wall:.1f}s reel)")
print(f"\nRobot :")
print(f"  {robot.coverage_summary()}")
print(f"\nWaypoints visites :")
for log in robot.waypoints_log:
    print(f"  {log}")

print(f"\nZone brulee finale : {grid.burned_fraction()*100:.1f}%")
print(f"Cellules en feu restantes : {grid.burning_count()}")
