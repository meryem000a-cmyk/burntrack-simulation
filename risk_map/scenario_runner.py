"""
risk_map/scenario_runner.py
===========================
Simule le feu depuis chaque point d'ignition prioritaire et classe les scénarios
par superficie brûlée potentielle → zones prioritaires de surveillance.
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cellular_automaton.grid import Grid, CellState
from cellular_automaton.rules import PropagationRules, EnsembleSimulation, PerturbationConfig
from cellular_automaton.simulation import FireSimulation


def run_single_scenario(
    grid: Grid,
    ignition_point: Dict,
    steps: int = 60,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Lance une simulation depuis un point d'ignition et collecte les résultats.

    Returns dict with: burned_ha, burned_pct, peak_burning, time_to_10pct,
                       spread_direction, ignition_point, step_history
    """
    g = copy.deepcopy(grid)
    rules = PropagationRules(stochastic=True)
    sim = FireSimulation(g, rules=rules, seed=seed)

    row, col = ignition_point["row"], ignition_point["col"]
    if g.cells[row][col].state == CellState.FIREBREAK:
        return {**ignition_point, "burned_ha": 0.0, "burned_pct": 0.0,
                "error": "FIREBREAK", "valid": False}

    sim.ignite(row, col)

    # Compter cellules non-FIREBREAK comme base
    total_burnable = sum(
        1 for i in range(g.rows) for j in range(g.cols)
        if g.cells[i][j].state != CellState.FIREBREAK
    )

    time_to_10pct = None
    step_history  = []

    for step in range(steps):
        n_new = sim.step(dt)
        burning = g.burning_count()
        fraction = g.burned_fraction()

        step_history.append({
            "t": round(sim.current_time, 1),
            "burning": burning,
            "fraction": round(fraction, 4),
            "new": n_new,
        })

        if time_to_10pct is None and fraction >= 0.10:
            time_to_10pct = round(sim.current_time, 1)

        if burning == 0:
            break

    # Superficie brûlée finale
    cell_area_ha = (g.cell_size ** 2) / 10_000.0
    burned_cells = sum(
        1 for i in range(g.rows) for j in range(g.cols)
        if g.cells[i][j].state in (CellState.BURNING, CellState.BURNED)
    )
    burned_ha  = burned_cells * cell_area_ha
    burned_pct = burned_cells / max(total_burnable, 1) * 100.0

    # Direction de propagation dominante (centroïde des cellules brûlées vs ignition)
    burned_rows, burned_cols = [], []
    for i in range(g.rows):
        for j in range(g.cols):
            if g.cells[i][j].state in (CellState.BURNING, CellState.BURNED):
                burned_rows.append(i)
                burned_cols.append(j)

    spread_dir = None
    if burned_rows:
        cr = np.mean(burned_rows) - row
        cc = np.mean(burned_cols) - col
        if abs(cr) > 0.5 or abs(cc) > 0.5:
            spread_dir = round(float(np.degrees(np.arctan2(cc, -cr)) % 360), 1)

    return {
        **ignition_point,
        "burned_ha":        round(burned_ha, 2),
        "burned_pct":       round(burned_pct, 2),
        "burned_cells":     burned_cells,
        "peak_burning":     max((s["burning"] for s in step_history), default=0),
        "sim_duration_min": round(sim.current_time, 1),
        "time_to_10pct":    time_to_10pct,
        "spread_direction": spread_dir,
        "step_history":     step_history,
        "valid":            True,
    }


def run_all_scenarios(
    grid: Grid,
    ignition_points: List[Dict],
    steps: int = 60,
    dt: float = 1.0,
    verbose: bool = True,
) -> List[Dict]:
    """
    Lance une simulation pour chaque point d'ignition et classe par superficie brûlée.

    Returns liste triée par burned_ha décroissant, avec danger_level ajouté.
    """
    results = []
    n = len(ignition_points)

    for k, pt in enumerate(ignition_points):
        if verbose:
            print(f"[Scénario {k+1}/{n}] Ignition en ({pt['row']},{pt['col']}) "
                  f"risque={pt['risk_score']:.3f}...", end=" ", flush=True)
        result = run_single_scenario(grid, pt, steps=steps, dt=dt, seed=42 + k)
        if verbose:
            if result.get("valid"):
                print(f"→ {result['burned_ha']:.1f} ha brûlés ({result['burned_pct']:.1f}%)")
            else:
                print(f"→ INVALIDE ({result.get('error','?')})")
        results.append(result)

    # Trier par superficie brûlée décroissante
    results.sort(key=lambda x: x.get("burned_ha", 0.0), reverse=True)

    # Renuméroter le rang de danger + label
    max_burned = max((r.get("burned_ha", 0) for r in results), default=1.0)
    for k, r in enumerate(results):
        r["scenario_rank"] = k + 1
        burned = r.get("burned_ha", 0)
        if burned >= max_burned * 0.75:
            r["danger_level"] = "critique"
        elif burned >= max_burned * 0.45:
            r["danger_level"] = "élevé"
        elif burned >= max_burned * 0.20:
            r["danger_level"] = "moyen"
        else:
            r["danger_level"] = "faible"

    return results


def run_ensemble_scenario(
    grid: Grid,
    ignition_point: Dict,
    n_realizations: int = 30,
    steps: int = 60,
    dt: float = 1.0,
    seed: int = 42,
) -> Dict:
    """
    Lance un ensemble stochastique depuis un point d'ignition.

    Returns: prob_map (rows×cols float), mean_burned_ha, std_burned_ha
    """
    g = copy.deepcopy(grid)
    row, col = ignition_point["row"], ignition_point["col"]

    perturb = PerturbationConfig(
        wind_speed=("normal", 0.0, 0.25),
        wind_dir=("normal", 0.0, 20.0),
        moisture_1h=("normal", 0.0, 0.015),
        fuel_load=("normal", 0.0, 0.10),
    )
    ens = EnsembleSimulation(
        g, n_realizations=n_realizations,
        perturb=perturb, seed=seed,
    )

    prob_map = ens.run(steps=steps, dt=dt, ignite_at=(row, col), verbose=False)

    cell_area_ha = (g.cell_size ** 2) / 10_000.0
    burned_areas = prob_map * n_realizations * cell_area_ha
    mean_burned  = float(prob_map.sum() * cell_area_ha)

    return {
        **ignition_point,
        "prob_map":       prob_map.tolist(),
        "mean_burned_ha": round(mean_burned, 2),
        "n_realizations": n_realizations,
    }
