"""Test MLP corrector instantiation and forward pass."""
import pytest
import torch
import numpy as np
from burntrack.corrector.mlp import (
    AtlasCorrectorV3, build_ia_vector, CorrectorDatasetLoader,
    FUEL_MODEL_ENCODING, N_FUEL_MODELS, REQUIRED_FEATURES, N_CONTINUOUS_FEATURES
)


def test_model_instantiation():
    model = AtlasCorrectorV3()
    assert model is not None
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 5000, f"Model too small: {n_params} params"


def test_forward_pass():
    model = AtlasCorrectorV3()
    model.eval()

    batch_size = 4
    x_cont = torch.randn(batch_size, N_CONTINUOUS_FEATURES)
    fuel_idx = torch.randint(0, N_FUEL_MODELS, (batch_size,))

    with torch.no_grad():
        out = model(x_cont, fuel_idx)

    assert out.shape == (batch_size, 2)
    assert torch.all(out[:, 0] >= -15.0) and torch.all(out[:, 0] <= 15.0)


def test_build_ia_vector():
    x_cont, fuel_idx = build_ia_vector(
        temp_c=35.0, rh_percent=25.0, wind_speed_ms=5.0, vpd_kpa=2.5,
        slope_deg=10.0, slope_pct=17.6, fuel_model_code="AF_STEPPE",
        w_total_kg_m2=0.42, w_dead_kg_m2=0.30, w_live_kg_m2=0.12,
        delta_m=0.25, sigma_m2_m3=12000, mx_percent=20, h_dead_kj_kg=18600,
        ros_rothermel=5.0, phi_w=2.5, phi_s=0.5, phi_eff=2.8,
        beta=0.0033, beta_opt=0.0042, gamma=16.0,
        eta_M=0.55, eta_S=0.95, I_R_kW_m2=150, xi=0.5, tau_min=0.5,
        ndvi=0.25, ndwi=-0.15, lst_c=48.0, dfmc_percent=13.0,
    )
    assert x_cont.shape == (N_CONTINUOUS_FEATURES,)
    assert 0 <= fuel_idx < N_FUEL_MODELS


def test_build_ia_vector_invalid_fuel():
    with pytest.raises(ValueError, match="Fuel model.*inconnu"):
        build_ia_vector(
            temp_c=35.0, rh_percent=25.0, wind_speed_ms=5.0, vpd_kpa=2.5,
            slope_deg=10.0, slope_pct=17.6, fuel_model_code="INVALID_FUEL",
            w_total_kg_m2=0.42, w_dead_kg_m2=0.30, w_live_kg_m2=0.12,
            delta_m=0.25, sigma_m2_m3=12000, mx_percent=20, h_dead_kj_kg=18600,
            ros_rothermel=5.0, phi_w=2.5, phi_s=0.5, phi_eff=2.8,
            beta=0.0033, beta_opt=0.0042, gamma=16.0,
            eta_M=0.55, eta_S=0.95, I_R_kW_m2=150, xi=0.5, tau_min=0.5,
            ndvi=0.25, ndwi=-0.15, lst_c=48.0, dfmc_percent=13.0,
        )


def test_scaler_fit_transform():
    loader = CorrectorDatasetLoader()
    X = np.random.randn(100, N_CONTINUOUS_FEATURES).astype(np.float32)
    X_scaled = loader.fit_transform(X)
    assert X_scaled.shape == X.shape
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)


def test_mc_dropout_uncertainty():
    model = AtlasCorrectorV3()
    batch_size = 8
    x_cont = torch.randn(batch_size, N_CONTINUOUS_FEATURES)
    fuel_idx = torch.randint(0, N_FUEL_MODELS, (batch_size,))

    result = model.predict_with_uncertainty(x_continuous=x_cont, fuel_idx=fuel_idx, n_samples=30)

    assert 'delta_ros' in result
    assert 'uncertainty' in result
    assert 'ci_lower' in result
    assert 'ci_upper' in result
    assert result['delta_ros'].shape == (batch_size, 1)
    assert result['uncertainty'].shape == (batch_size, 1)
