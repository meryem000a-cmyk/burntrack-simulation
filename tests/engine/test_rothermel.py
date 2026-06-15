"""Test Rothermel engine computes reasonable values."""
import pytest
import numpy as np
from burntrack.engine import RothermelEngine, FuelModel, MoistureInputs, EnvironmentalConditions
from burntrack.engine import BurnTrackRothermel
from burntrack.engine.fuel_models import get_fuel_model


def test_engine_instantiation():
    engine = RothermelEngine()
    assert engine is not None
    assert engine.uc is not None
    assert engine.const is not None


def test_gr1_no_wind_no_slope():
    engine = RothermelEngine()
    fuel = get_fuel_model("GR1")

    assert fuel is not None
    assert fuel.w_total > 0

    moisture = MoistureInputs(m_1h=0.05, m_10h=0.06, m_100h=0.07, m_live_herb=0.3, m_live_woody=0.6)
    conditions = EnvironmentalConditions(wind_speed=0.0, slope_pct=0.0)
    output = engine.compute(fuel, moisture, conditions)

    assert output.ros > 0, "ROS should be positive for dry fuel"
    assert output.ros < 10.0, "ROS without wind/slope should be moderate"
    assert output.flame_length >= 0
    assert output.fireline_intensity >= 0
    assert output.beta > 0
    assert output.beta_opt > 0


def test_wind_increases_ros():
    engine = RothermelEngine()
    fuel = get_fuel_model("GR1")
    moisture = MoistureInputs()

    out_no_wind = engine.compute(fuel, moisture, EnvironmentalConditions(wind_speed=0.0, slope_pct=0.0))
    out_wind = engine.compute(fuel, moisture, EnvironmentalConditions(wind_speed=5.0, slope_pct=0.0))

    assert out_wind.ros > out_no_wind.ros, f"Wind ROS ({out_wind.ros}) should exceed no-wind ({out_no_wind.ros})"
    assert out_wind.phi_w > 0


def test_slope_increases_ros():
    engine = RothermelEngine()
    fuel = get_fuel_model("GR1")
    moisture = MoistureInputs()

    out_flat = engine.compute(fuel, moisture, EnvironmentalConditions(wind_speed=0.0, slope_pct=0.0))
    out_slope = engine.compute(fuel, moisture, EnvironmentalConditions(wind_speed=0.0, slope_pct=30.0))

    assert out_slope.ros > out_flat.ros
    assert out_slope.phi_s > 0


def test_high_moisture_reduces_ros():
    engine = RothermelEngine()
    fuel = get_fuel_model("GR1")

    dry = MoistureInputs(m_1h=0.03, m_10h=0.04, m_100h=0.05)
    wet = MoistureInputs(m_1h=0.25, m_10h=0.26, m_100h=0.27)

    out_dry = engine.compute(fuel, dry, EnvironmentalConditions(wind_speed=3.0, slope_pct=0.0))
    out_wet = engine.compute(fuel, wet, EnvironmentalConditions(wind_speed=3.0, slope_pct=0.0))

    assert out_dry.ros > out_wet.ros, f"Dry ROS ({out_dry.ros}) should > wet ({out_wet.ros})"


def test_zero_fuel_returns_zero_ros():
    engine = RothermelEngine()
    fuel = FuelModel(name="empty")
    moisture = MoistureInputs()
    conditions = EnvironmentalConditions(wind_speed=5.0, slope_pct=10.0)

    output = engine.compute(fuel, moisture, conditions)
    assert output.ros == 0.0
    assert output.flame_length == 0.0


def test_all_fuel_models_load():
    from burntrack.engine.fuel_models import BEHAVE_STANDARD, AFRICA_NORTH, AFRICA_SAVANNA

    engine = RothermelEngine()
    moisture = MoistureInputs()
    conditions = EnvironmentalConditions(wind_speed=2.0, slope_pct=5.0)

    all_models = list(BEHAVE_STANDARD.keys()) + list(AFRICA_NORTH.keys()) + list(AFRICA_SAVANNA.keys())
    assert len(all_models) >= 50, f"Expected 50+ fuel models, got {len(all_models)}"

    for code in all_models[:5]:
        fuel = get_fuel_model(code)
        assert fuel is not None, f"Fuel model {code} failed to load"
        output = engine.compute(fuel, moisture, conditions)
        assert output is not None, f"Engine.compute failed for {code}"


def test_output_contains_all_fields():
    engine = RothermelEngine()
    fuel = get_fuel_model("GR3")
    output = engine.compute(fuel, MoistureInputs(), EnvironmentalConditions(wind_speed=3.0, slope_pct=10.0))

    assert hasattr(output, 'ros')
    assert hasattr(output, 'fireline_intensity')
    assert hasattr(output, 'flame_length')
    assert hasattr(output, 'reaction_intensity')
    assert hasattr(output, 'spread_direction')
    assert hasattr(output, 'heat_per_unit_area')
    assert hasattr(output, 'phi_w')
    assert hasattr(output, 'phi_s')
    assert hasattr(output, 'phi_eff')
    assert hasattr(output, 'beta')
    assert hasattr(output, 'beta_opt')
    assert hasattr(output, 'gamma')
    assert hasattr(output, 'eta_M')
    assert hasattr(output, 'eta_S')
    assert hasattr(output, 'xi')
    assert hasattr(output, 'tau')


def test_burntrack_rothermel_interface():
    bt = BurnTrackRothermel()
    fuel = get_fuel_model("GR1")
    moisture = MoistureInputs()
    result = bt.predict(fuel, moisture, wind_speed=3.0, slope_pct=10.0, angle_wind_slope=30.0)

    assert isinstance(result, dict)
    assert 'ros' in result
    assert 'flame_length' in result
    assert 'fireline_intensity' in result
    assert result['ros'] > 0
