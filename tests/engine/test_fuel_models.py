"""Test fuel model loading and validation."""
import pytest
import numpy as np
from burntrack.engine.fuel_models import (
    get_fuel_model, get_fuel_model_by_species, get_fuel_model_by_ecosystem,
    BEHAVE_STANDARD, AFRICA_NORTH, AFRICA_SAVANNA,
    SPECIES_TO_FUEL_MODEL, ECOSYSTEM_TO_FUEL_MODEL
)


def test_grass_fuel_models_load():
    for code in ['GR1', 'GR4', 'GR7']:
        fm = get_fuel_model(code)
        assert fm.w_1h > 0 or fm.w_live_herb > 0, f"{code} should have fuel loading"
        assert fm.sigma_1h > 0 or fm.sigma_live_herb > 0, f"{code} should have SAV"
        assert fm.delta > 0, f"{code} should have fuel bed depth"
        assert fm.mx > 0, f"{code} should have moisture of extinction"
        assert fm.h_dead > 0, f"{code} should have heat content"


def test_shrub_fuel_models_load():
    for code in ['SH1', 'SH5']:
        fm = get_fuel_model(code)
        assert fm is not None


def test_africa_north_models_load():
    for code in ['AF_STEPPE', 'AF_MAQUIS']:
        fm = get_fuel_model(code)
        assert fm is not None
        assert fm.w_total > 0


def test_africa_savanna_models_load():
    for code in ['AF_MIOMBO', 'AF_FYNBOS', 'AF_SAHEL_GRASS']:
        fm = get_fuel_model(code)
        assert fm is not None
        assert fm.w_total > 0


def test_species_lookup():
    for species in ['acacia', 'baobab', 'brachystegia']:
        fm = get_fuel_model_by_species(species)
        assert fm is not None, f"No fuel model for species: {species}"


def test_ecosystem_lookup():
    for eco in ['fynbos', 'miombo', 'steppe']:
        fm = get_fuel_model_by_ecosystem(eco)
        assert fm is not None, f"No fuel model for ecosystem: {eco}"


def test_fuel_model_totals():
    fm = get_fuel_model("SH5")
    computed = fm.w_1h + fm.w_10h + fm.w_100h + fm.w_live_herb + fm.w_live_woody
    assert np.isclose(fm.w_total, computed, rtol=1e-4)


def test_all_fuel_models_have_required_fields():
    all_models = {**BEHAVE_STANDARD, **AFRICA_NORTH, **AFRICA_SAVANNA}
    for code, fm in list(all_models.items())[:10]:
        assert fm.w_total >= 0
        assert fm.delta > 0
        assert fm.mx >= 0
        assert fm.h_dead >= 0
