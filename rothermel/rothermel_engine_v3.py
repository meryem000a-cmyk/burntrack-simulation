# Compatibility shim — old `rothermel.rothermel_engine_v3` → `burntrack.engine.rothermel`
from burntrack.engine.rothermel import (  # noqa: F401
    RothermelEngine,
    RothermelOutput,
    EnvironmentalConditions,
    MoistureInputs,
    FuelModel,
)
RothFuelModel = FuelModel
