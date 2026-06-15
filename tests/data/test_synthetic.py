"""Test synthetic dataset generation."""
import pytest
import numpy as np
import pandas as pd
from burntrack.data.synthetic import (
    compute_vpd, compute_dfmc,
    AFRICA_CLIMATE_ZONES, BIAS_PROFILES, FUEL_MODEL_ENCODING
)


def test_compute_vpd():
    vpd = compute_vpd(35.0, 20.0)
    assert 0 < vpd < 10, f"VPD should be 0-10 kPa, got {vpd}"


def test_vpd_zero_at_saturation():
    vpd = compute_vpd(25.0, 100.0)
    assert vpd <= 0.01


def test_estimate_fuel_moisture():
    dfmc = compute_dfmc(30.0, compute_vpd(30.0, 35.0))
    assert 3.0 <= dfmc <= 40.0


def test_all_zones_have_required_keys():
    required = ['temp_mean', 'temp_std', 'rh_mean', 'rh_std', 'wind_mean',
                'wind_std', 'slope_mean', 'slope_std', 'vpd_bias', 'fuel_models', 'bias_profile']
    for zone, params in AFRICA_CLIMATE_ZONES.items():
        for key in required:
            assert key in params, f"Zone {zone} missing key: {key}"


def test_all_bias_profiles_valid():
    for name, profile in BIAS_PROFILES.items():
        assert 'base_factor' in profile
        assert 'factor_std' in profile
        assert 0.5 < profile['base_factor'] < 2.0, f"{name}: factor {profile['base_factor']} out of range"


def test_fuel_model_encoding_consistent():
    assert len(FUEL_MODEL_ENCODING) >= 6
    for code, idx in FUEL_MODEL_ENCODING.items():
        assert isinstance(idx, float) or isinstance(idx, int)
