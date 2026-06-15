"""Constants for the Rothermel fire spread engine."""


class UnitConverter:
    """Conversions SI ↔ Impérial."""
    M_TO_FT = 3.28084
    KG_M2_TO_LB_FT2 = 0.204816
    KG_M3_TO_LB_FT3 = 0.062428
    KJ_KG_TO_BTU_LB = 0.429923
    M_MIN_TO_FT_MIN = 3.28084
    # 1 m/s = 196.8504 ft/min (pour vent Rothermel)
    MS_TO_FT_MIN = 196.8504


class RothermelConstants:
    """Constantes empiriques de Rothermel (1972)."""
    WIND_C_COEF = 7.47
    WIND_C_EXP = -0.133
    WIND_B_COEF = 0.02526
    WIND_B_EXP = 0.54
    WIND_E_COEF = 0.715
    WIND_E_EXP = -0.000359
    SLOPE_COEF = 5.275
