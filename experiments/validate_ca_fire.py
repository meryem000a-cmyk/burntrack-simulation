"""
Validate Rothermel + Cellular Automaton against known fire behaviour.
Uses long thin grid (30x500) to avoid CFL boundary effects.
Focus: does the fire burn realistically? ROS comparison with calibration factor.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellular_automaton import Grid, FireSimulation, PropagationRules
from cellular_automaton.grid import CellState
from burntrack.engine.rothermel import RothermelEngine, MoistureInputs, EnvironmentalConditions
from burntrack.engine.fuel_models import ALL_FUEL_MODELS

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
CELL = 30.0


def measure_head_col(grid):
    maxc = -1
    for i in range(grid.rows):
        for j in range(grid.cols):
            if grid.cells[i][j].state in (CellState.BURNING, CellState.BURNED):
                if j > maxc: maxc = j
    return maxc


def measure_back_col(grid):
    minc = grid.cols
    for i in range(grid.rows):
        for j in range(grid.cols):
            if grid.cells[i][j].state in (CellState.BURNING, CellState.BURNED):
                if j < minc: minc = j
    return minc


def scenario(fuel_code, moisture_1h, wind_ms, name="scenario", rows=30, cols=500, base_steps=240, dt=0.25, rules=None):
    steps = int(base_steps * (1.0 / dt))
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Fuel={fuel_code}  Moisture={moisture_1h:.2f}  Wind={wind_ms} m/s W→E")

    # Grid: ignition at col 5, wind from 270° (West → East)
    grid = Grid.uniform(rows, cols, cell_size=CELL, fuel_code=fuel_code,
                        moisture_1h=moisture_1h, wind_speed_ms=wind_ms,
                        wind_dir_deg=270.0, slope_pct=0.0)

    # Rothermel point prediction
    fm = ALL_FUEL_MODELS[fuel_code]
    mois = MoistureInputs(m_1h=moisture_1h, m_10h=moisture_1h+0.01,
                          m_100h=moisture_1h+0.02,
                          m_live_herb=min(moisture_1h * 6, 0.9),
                          m_live_woody=min(moisture_1h * 8, 0.9))
    cond = EnvironmentalConditions(wind_speed=wind_ms, slope_pct=0.0, angle_wind_slope=0.0)
    pred = RothermelEngine().compute(fm, mois, cond)
    print(f"  Rothermel: ROS={pred.ros:.2f} m/min  Ib={pred.fireline_intensity:.0f} kW/m"
          f"  flame={pred.flame_length:.2f}m  tau={pred.residence_time:.3f}min")

    # CA — stochastic
    if rules is None:
        rules = PropagationRules(stochastic=True, burn_duration_factor=4.0,
                                 min_burn_min=5.0, min_ros_m_min=0.01,
                                 directional_exponent=2.0, back_fire_fraction=0.15)
    sim = FireSimulation(grid, rules, seed=42)
    sim.ignite(rows // 2, 5)

    head_positions = []
    back_positions = []
    times = []

    for step in range(steps):
        sim.step(dt)
        times.append(sim.current_time)
        head_positions.append(measure_head_col(grid))
        back_positions.append(measure_back_col(grid))
        if grid.burning_count() == 0 and step > 10:
            break

    head_arr = np.array(head_positions)
    t_arr = np.array(times)

    # Measure forward ROS using linear regression on middle portion
    # (after ignition stabilizes, before fire hits grid boundary)
    valid = (head_arr > 10) & (head_arr < cols - 5)
    if valid.sum() >= 10:
        tv = t_arr[valid]
        hv = head_arr[valid]
        A = np.vstack([tv, np.ones_like(tv)]).T
        slope, intercept = np.linalg.lstsq(A, hv, rcond=None)[0]
        obs_cells_per_min = float(slope)
        obs_ros = obs_cells_per_min * CELL
    else:
        obs_ros = 0.0
        obs_cells_per_min = 0.0

    ros_ratio = obs_ros / pred.ros * 100 if pred.ros > 0 else 0.0
    print(f"  CA observed: {obs_cells_per_min:.2f} cells/min = {obs_ros:.2f} m/min")
    print(f"  Ratio CA / Rothermel: {ros_ratio:.0f}%")

    # Burn scar at final step
    n_burned = 0
    for i in range(grid.rows):
        for j in range(grid.cols):
            if grid.cells[i][j].state in (CellState.BURNING, CellState.BURNED):
                n_burned += 1
    area_ha = n_burned * CELL * CELL / 10000.0

    rows_b = np.any([[grid.cells[i][j].state in (CellState.BURNING, CellState.BURNED)
                       for j in range(grid.cols)] for i in range(grid.rows)], axis=1).sum()
    cols_b = np.any([[grid.cells[i][j].state in (CellState.BURNING, CellState.BURNED)
                       for j in range(grid.cols)] for i in range(grid.rows)], axis=0).sum()

    # Measure head-to-back ratio
    head_final = max(head_positions) if head_positions else 0
    back_final = min(back_positions) if back_positions else 0
    hb_ratio = (head_final - 5) / max((5 - back_final), 1) if (5 - back_final) > 0 else 99.0
    print(f"  Burned: {area_ha:.1f} ha ({n_burned} cells, {n_burned/(rows*cols)*100:.1f}%)")
    print(f"  Wind axis: {cols_b} cells  Cross: {rows_b} cells  Head/back: {hb_ratio:.1f}")

    # Evaluate realism
    flags = []
    if pred.ros > 30 and ros_ratio < 50:
        flags.append(f"CFL-limited (ROS>{30:.0f} m/min)")
    if 5 < pred.ros < 30 and abs(ros_ratio - 100) > 50:
        flags.append(f"ROS mismatch ({ros_ratio:.0f}% of Rothermel)")
    if area_ha < 0.1:
        flags.append("FIRE WENT OUT — check params")
    if hb_ratio < 2.0 and wind_ms > 2:
        flags.append(f"Low head/back ratio ({hb_ratio:.1f}) for wind={wind_ms}m/s")
    if hb_ratio > 10:
        flags.append(f"Very elongated ({hb_ratio:.1f})")
    if area_ha > 0.1 and hb_ratio > 1.5:
        flags.append("BEHAVIOUR PLAUSIBLE")
    flag_str = "; ".join(flags) if flags else "---"

    print(f"  Verdict: {flag_str}")

    return dict(name=name, fuel=fuel_code, moisture=moisture_1h, wind_ms=wind_ms,
                pred_ros=round(pred.ros, 2), obs_ros=round(obs_ros, 2),
                cells_per_min=round(obs_cells_per_min, 3),
                ros_ratio_pct=round(ros_ratio, 1),
                pred_ib=round(pred.fireline_intensity),
                flame_length=round(pred.flame_length, 2),
                tau=round(pred.residence_time, 3),
                head_back_ratio=round(hb_ratio, 1),
                wind_axis_cells=cols_b, cross_axis_cells=rows_b,
                burned_ha=round(area_ha, 1), burned_cells=n_burned,
                total_cells=rows * cols, pct_burned=round(n_burned/(rows*cols)*100, 1),
                sim_min=round(sim.current_time, 1), steps=sim.step_count,
                verdict=flag_str)


def main():
    results = []

    scenarios = [
        ("GR4", 0.06, 5.0, "GR4 6% 5m/s — Stocks high-intensity (~1.7 m/s)"),
        ("GR4", 0.06, 3.0, "GR4 6% 3m/s — Dry grass moderate wind"),
        ("GR4", 0.15, 5.0, "GR4 15% 5m/s — Moderate drought"),
        ("GR4", 0.28, 5.0, "GR4 28% 5m/s — Stocks low-intensity (~0.4 m/s)"),
        ("GR4", 0.35, 3.0, "GR4 35% 3m/s — Govender summer wet analog"),
        ("AF_SAHEL_GRASS", 0.15, 5.0, "SAHEL 15% 5m/s — Sahel dry season"),
        ("AF_SAHEL_GRASS", 0.28, 5.0, "SAHEL 28% 5m/s — Sahel late dry"),
        ("AF_GRASSLAND_FERTILE", 0.15, 5.0, "FERTILE 15% 5m/s — Fertile dry"),
        ("AF_GRASSLAND_FERTILE", 0.28, 5.0, "FERTILE 28% 5m/s — Fertile late dry"),
        ("AF_MIOMBO", 0.20, 4.0, "MIOMBO 20% 4m/s — Miombo understory"),
        ("AF_BUSHVELD", 0.20, 5.0, "BUSHVELD 20% 5m/s — Bushveld savanna"),
        ("AF_FYNBOS", 0.20, 5.0, "FYNBOS 20% 5m/s — Fynbos shrubland"),
        ("AF_MOPANE", 0.15, 5.0, "MOPANE 15% 5m/s — Mopane savanna"),
    ]

    for fuel, m, ws, name in scenarios:
        r = scenario(fuel, m, ws, name=name)
        results.append(r)

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Scenario':45s} {'R_pred':>7s} {'CA_obs':>7s} {'Ratio':>7s} {'ha':>6s} {'H/B':>5s} {'Verdict'}")
    print("-"*90)
    for r in results:
        r_str = f"{r['ros_ratio_pct']:.0f}%" if r['obs_ros'] > 0 else "N/A"
        print(f"{r['name']:45s} {r['pred_ros']:7.2f} {r['obs_ros']:7.2f} {r_str:>7s} "
              f"{r['burned_ha']:>6.1f} {r['head_back_ratio']:>5.1f}  {r['verdict']}")

    path = os.path.join(OUT, "validation_results.json")
    clean = []
    for r in results:
        clean.append({k: (int(v) if isinstance(v, (np.integer,)) else
                         float(v) if isinstance(v, (np.floating,)) else v)
                      for k, v in r.items()})
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
