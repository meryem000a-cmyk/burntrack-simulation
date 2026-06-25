import yaml
from pathlib import Path
from cellular_automaton.grid import Grid, CellState
from cellular_automaton.rules import PropagationRules
from burntrack.engine.rothermel import RothermelEngine, EnvironmentalConditions, MoistureInputs
from burntrack.engine.fuel_models import ALL_FUEL_MODELS

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "simulation.yaml"

def load_ca_config(path=None):
    """Load calibrated CA parameters from YAML config."""
    p = Path(path) if path else _CONFIG_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ca", {})

def create_rules(config=None, **overrides):
    """Create PropagationRules with calibrated defaults."""
    cfg = config or load_ca_config()
    cfg.update(overrides)  # allow caller overrides
    return PropagationRules(
        stochastic=cfg.get("stochastic", True),
        directional_exponent=cfg.get("directional_exponent", 4.0),
        back_fire_fraction=cfg.get("back_fire_fraction", 0.15),
        burn_duration_factor=cfg.get("burn_duration_factor", 4.0),
        min_burn_min=cfg.get("min_burn_min", 5.0),
        min_ros_m_min=cfg.get("min_ros_m_min", 0.01),
    )

def calibrate_dt(grid, rules=None, max_dt=0.5, cfl_target=0.8):
    """Estimate safe timestep from current burning cells (adaptive)."""
    max_ros = 0.0
    engine = RothermelEngine() if rules is None else rules.engine
    for i in range(grid.rows):
        for j in range(grid.cols):
            if grid.cells[i][j].state == CellState.BURNING:
                try:
                    fm = ALL_FUEL_MODELS[grid.cells[i][j].fuel_code]
                    mois = grid.cells[i][j].moisture
                    cond = EnvironmentalConditions(
                        wind_speed=grid.cells[i][j].wind_speed_ms,
                        slope_pct=grid.cells[i][j].slope_pct,
                        angle_wind_slope=0.0,
                    )
                    out = engine.compute(fm, MoistureInputs(**mois.__dict__), cond)
                    if out.ros > max_ros:
                        max_ros = out.ros
                except Exception:
                    continue
    if max_ros < 0.01:
        return max_dt
    return min(max_dt, cfl_target * grid.cell_size / max_ros)

def run_adaptive(sim, grid, max_dt=0.5, cfl_target=0.8):
    """Single adaptive step."""
    dt = calibrate_dt(grid, max_dt=max_dt, cfl_target=cfl_target)
    sim.step(dt)
    return dt
