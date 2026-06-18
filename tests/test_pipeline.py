"""
tests/test_pipeline.py
======================
Tests du pipeline complet BurnTrack :
  Rothermel -> Correcteur IA -> Cellular Automata propagation.
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# ── 1. Rothermel Engine ──────────────────────────────────────────────

class TestRothermel:
    def setup_method(self):
        from burntrack.engine import RothermelEngine, MoistureInputs, EnvironmentalConditions
        from burntrack.engine.fuel_models import get_fuel_model
        self.engine = RothermelEngine()
        self.get_fuel_model = get_fuel_model
        self.MoistureInputs = MoistureInputs
        self.EnvConds = EnvironmentalConditions

    def test_gr3_with_wind_and_slope(self):
        fuel = self.get_fuel_model("GR3")
        moisture = self.MoistureInputs(m_1h=0.06, m_10h=0.07, m_100h=0.08,
                                       m_live_herb=0.3, m_live_woody=0.6)
        conditions = self.EnvConds(wind_speed=5.0, slope_pct=20.0)
        out = self.engine.compute(fuel, moisture, conditions)

        assert out.ros > 0, "ROS should be positive with wind+slope"
        assert out.ros < 100.0, "ROS should be reasonable"
        assert out.flame_length > 0
        assert out.fireline_intensity > 0
        assert out.phi_w > 0, "Wind factor should be positive"
        assert out.phi_s > 0, "Slope factor should be positive"

    def test_higher_wind_higher_ros(self):
        fuel = self.get_fuel_model("GR3")
        moisture = self.MoistureInputs(m_1h=0.06, m_10h=0.07, m_100h=0.08,
                                       m_live_herb=0.3, m_live_woody=0.6)
        out_low = self.engine.compute(fuel, moisture, self.EnvConds(wind_speed=2.0, slope_pct=0.0))
        out_high = self.engine.compute(fuel, moisture, self.EnvConds(wind_speed=10.0, slope_pct=0.0))
        assert out_high.ros > out_low.ros, "Higher wind should increase ROS"

    def test_higher_slope_higher_ros(self):
        fuel = self.get_fuel_model("GR3")
        moisture = self.MoistureInputs(m_1h=0.06, m_10h=0.07, m_100h=0.08,
                                       m_live_herb=0.3, m_live_woody=0.6)
        out_flat = self.engine.compute(fuel, moisture, self.EnvConds(wind_speed=0.0, slope_pct=0.0))
        out_steep = self.engine.compute(fuel, moisture, self.EnvConds(wind_speed=0.0, slope_pct=30.0))
        assert out_steep.ros > out_flat.ros, "Higher slope should increase ROS"


# ── 2. AI Corrector (RF) ────────────────────────────────────────────

class TestCorrectorRF:
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_corrector.joblib")
    SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "rf_scaler.joblib")

    RF_FEATURES = [
        "ros_rothermel", "ros_terrain", "temp_c", "rh_percent", "wind_speed_ms",
        "wind_10m", "vpd_kpa", "slope_deg", "slope_pct", "angle_wind_slope",
        "w_total_kg_m2", "w_dead_kg_m2", "w_live_kg_m2", "delta_m", "sigma_m2_m3",
        "mx_percent", "h_dead_kj_kg", "phi_w", "phi_s", "phi_eff",
        "beta", "beta_opt", "beta_ratio", "gamma", "eta_M", "eta_S",
        "I_R_kW_m2", "xi", "tau_min", "ndvi", "ndwi", "lst_c",
        "dfmc_percent", "m_1h", "m_10h", "m_100h", "m_live_herb", "m_live_woody",
    ]

    def setup_method(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("rf_corrector.joblib not found")
        import joblib
        self.model = joblib.load(self.MODEL_PATH)
        self.scaler = joblib.load(self.SCALER_PATH)

    def test_model_loads(self):
        assert self.model is not None
        assert self.scaler is not None

    def test_feature_count(self):
        assert self.scaler.n_features_in_ == 38

    def test_prediction_shape(self):
        x = np.zeros((1, 38), dtype=np.float64)
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)
        assert pred.shape == (1,)

    def test_prediction_is_finite(self):
        x = np.random.rand(5, 38).astype(np.float64)
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled)
        assert np.all(np.isfinite(preds)), "Predictions should be finite"

    def test_hot_dry_gives_positive_delta(self):
        x = np.zeros((1, 38), dtype=np.float64)
        x[0, 0] = 2.0   # ros_rothermel
        x[0, 2] = 35.0  # temp_c (hot)
        x[0, 3] = 10.0  # rh_percent (dry)
        x_scaled = self.scaler.transform(x)
        delta = float(self.model.predict(x_scaled)[0])
        assert np.isfinite(delta)


# ── 3. Cellular Automaton ────────────────────────────────────────────

class TestCellularAutomaton:
    def setup_method(self):
        from cellular_automaton.grid import Grid, CellState
        from cellular_automaton.simulation import FireSimulation
        self.Grid = Grid
        self.CellState = CellState
        self.FireSimulation = FireSimulation

    def test_grid_creation(self):
        grid = self.Grid.uniform(20, 20, cell_size=30.0, fuel_code="GR3",
                                  moisture_1h=0.06, wind_speed_ms=3.0)
        assert grid.rows == 20
        assert grid.cols == 20
        assert grid.burning_count() == 0
        assert grid.burned_fraction() == 0.0

    def test_ignite_and_burn(self):
        grid = self.Grid.uniform(20, 20, cell_size=30.0, fuel_code="GR3",
                                  moisture_1h=0.06, wind_speed_ms=5.0)
        sim = self.FireSimulation(grid, seed=42)
        sim.ignite(10, 10)

        assert grid.cells[10][10].state == self.CellState.BURNING

        n_new = sim.step(dt=1.0)
        assert n_new >= 0
        assert sim.current_time == 1.0

    def test_fire_spreads(self):
        grid = self.Grid.uniform(30, 30, cell_size=30.0, fuel_code="GR3",
                                  moisture_1h=0.05, wind_speed_ms=8.0)
        sim = self.FireSimulation(grid, seed=42)
        sim.ignite(15, 15)

        for _ in range(10):
            sim.step(dt=1.0)

        burned_frac = grid.burned_fraction()
        assert burned_frac > 0, "Fire should have spread"

    def test_burned_area_increases(self):
        grid = self.Grid.uniform(30, 30, cell_size=30.0, fuel_code="GR3",
                                  moisture_1h=0.05, wind_speed_ms=5.0)
        sim = self.FireSimulation(grid, seed=42)
        sim.ignite(15, 15)

        fractions = []
        for _ in range(15):
            sim.step(dt=1.0)
            fractions.append(grid.burned_fraction())

        assert fractions[-1] > fractions[0], "Burned area should increase over time"

    def test_delta_ros_field(self):
        from cellular_automaton.grid import Cell
        c = Cell()
        assert c.delta_ros == 0.0
        c.delta_ros = 0.5
        assert c.delta_ros == 0.5


# ── 4. Pipeline End-to-End ───────────────────────────────────────────

class TestPipelineEndToEnd:
    def setup_method(self):
        from scripts.run_pipeline import PipelineRunner
        self.runner = PipelineRunner(model_dir=os.path.join(PROJECT_ROOT, "models"))

    def test_pipeline_returns_dict(self):
        result = self.runner.run(lat=36.8, lon=3.0, fuel_model="GR3")
        assert isinstance(result, dict)
        assert "rothermel" in result
        assert "corrector" in result
        assert "danger_level" in result
        assert "ros_final_m_min" in result

    def test_pipeline_corrector_output(self):
        result = self.runner.run(lat=36.8, lon=3.0, fuel_model="GR3")
        corr = result["corrector"]
        assert "ros_corrected" in corr
        assert "delta_ros" in corr
        assert "uncertainty_std" in corr
        assert np.isfinite(corr["ros_corrected"])
        assert np.isfinite(corr["delta_ros"])

    def test_pipeline_feature_count(self):
        assert len(self.runner.RF_FEATURE_NAMES) == 38
        assert self.runner.scaler is not None
        assert self.runner.scaler.n_features_in_ == 38

    def test_pipeline_with_robot_data(self):
        robot_data = {
            "wind_speed": 6.0,
            "slope_pct": 15.0,
            "temp_air": 32.0,
            "rh": 20.0,
            "ndvi": 0.4,
        }
        result = self.runner.run(lat=36.8, lon=3.0, fuel_model="GR3", robot_data=robot_data)
        assert result["ros_final_m_min"] >= 0


# ── 5. Pipeline → CA Integration ─────────────────────────────────────

class TestPipelineCAIntegration:
    def setup_method(self):
        from scripts.run_pipeline import PipelineRunner
        from cellular_automaton.grid import Grid
        from cellular_automaton.simulation import FireSimulation
        self.PipelineRunner = PipelineRunner
        self.Grid = Grid
        self.FireSimulation = FireSimulation

    def test_pipeline_feeds_ca(self):
        runner = self.PipelineRunner(model_dir=os.path.join(PROJECT_ROOT, "models"))
        result = runner.run(lat=36.8, lon=3.0, fuel_model="GR3")
        ros = result["corrector"]["ros_corrected"]

        grid = self.Grid.uniform(20, 20, cell_size=30.0, fuel_code="GR3",
                                  moisture_1h=0.06, wind_speed_ms=3.0)
        sim = self.FireSimulation(grid, seed=42)
        sim.ignite(10, 10)

        for _ in range(5):
            sim.step(dt=1.0)

        assert sim.current_time == 5.0

    def test_delta_ros_applied_in_ca(self):
        from cellular_automaton.grid import Cell
        c = Cell(delta_ros=0.5)
        assert c.delta_ros == 0.5
