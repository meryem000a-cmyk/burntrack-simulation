"""
rothermel_engine.py (v2 corrigé)
====================
Moteur de calcul Rothermel pour le pipeline BurnTrack.

Corrections intégrées (v2) :
1. Coefficient de vent : (beta / beta_opt)**(-E) au lieu de beta**(-E)
2. Intensité de réaction : pondération h_dead/h_live par les charges
3. Formule I_b : alignée sur Byram standard (I_R × tau)
4. Validation : warnings explicites au lieu de clip silencieux
5. Angle vent/pente/propagation : paramètre angle_wind_slope ajouté

Implémente le modèle de propagation de surface de Rothermel (1972)
et ses extensions (Anderson 1982, Scott & Burgan 2005).

IMPORTANT : Les formules de Rothermel sont calibrées en unités impériales.
Ce module convertit automatiquement les entrées SI → impérial,
calcule en impérial, puis reconvertit les sorties en SI.

Conversion SAV : 1 m²/m³ = 0.3048 ft²/ft³

Sources :
- Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread
  in Wildland Fuels. USDA Forest Service Research Paper INT-115.
- Anderson, H.E. (1982). Aids to determining fuel models for estimating
  fire behavior. USDA Forest Service General Technical Report INT-122.
- Scott, J.H. & Burgan, R.E. (2005). Standard Fire Behavior Fuel Models.
  USDA Forest Service General Technical Report RMRS-GTR-153.
- Albini, F.A. & Baughman, R.G. (1979). Estimating windspeeds for
  predicting wildland fire behavior. USDA Forest Service Research Paper INT-221.
- Andrews, P.L. (2018). The Rothermel surface fire spread model and associated
  developments: A comprehensive explanation. USDA Forest Service RMRS-GTR-371.

Unités d'entrée (SI) :
- Charges : kg/m²
- Profondeur : m
- SAV : m²/m³ = 1/m
- Chaleur : kJ/kg
- Vent : m/s
- Pente : degrés
- Angle : degrés (0 = vent aligné avec pente, 90 = vent perpendiculaire)

Unités de sortie (SI) :
- ROS : m/min
- Intensité : kW/m
- Longueur flamme : m
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fuel_models import FuelModel, compute_dynamic_herb_load


# =============================================================================
# FACTEURS DE CONVERSION SI ↔ IMPÉRIAL
# =============================================================================

class UnitConversions:
    """Facteurs de conversion pour les calculs Rothermel."""

    # Masse
    KG_TO_LB = 2.20462

    # Surface
    M2_TO_FT2 = 10.7639

    # Longueur
    M_TO_FT = 3.28084
    FT_TO_M = 1.0 / M_TO_FT

    # Volume
    M3_TO_FT3 = 35.3147

    # SAV : m²/m³ → ft²/ft³
    # 1 m²/m³ = (3.28084 ft)² / (3.28084 ft)³ = 1 / 3.28084 = 0.3048 ft²/ft³
    SAV_M_TO_FT = 1.0 / M_TO_FT  # 0.3048

    # Chaleur : kJ/kg → BTU/lb
    KJ_KG_TO_BTU_LB = 0.4299
    BTU_LB_TO_KJ_KG = 1.0 / KJ_KG_TO_BTU_LB  # 2.326

    # Vitesse : m/min → ft/min
    M_MIN_TO_FT_MIN = M_TO_FT  # 3.28084
    FT_MIN_TO_M_MIN = FT_TO_M  # 0.3048

    # Densité : kg/m³ → lb/ft³
    KG_M3_TO_LB_FT3 = KG_TO_LB / M3_TO_FT3  # 0.06243
    LB_FT3_TO_KG_M3 = 1.0 / KG_M3_TO_LB_FT3  # 16.018

    # Charge : kg/m² → lb/ft²
    KG_M2_TO_LB_FT2 = KG_TO_LB / M2_TO_FT2  # 0.2048
    LB_FT2_TO_KG_M2 = 1.0 / KG_M2_TO_LB_FT2  # 4.882


# =============================================================================
# CONSTANTES PHYSIQUES ROTHERMEL (unités impériales)
# =============================================================================

class RothermelConstants:
    """Constantes physiques du modèle de Rothermel (unités impériales)."""

    # Teneur en minéraux (fraction)
    S_T = 0.0555
    S_E = 0.0100

    # Densité particulaire du combustible
    RHO_P = 32.0          # lb/ft³ (≈ 512 kg/m³)

    # SAV des catégories fixes (ft²/ft³)
    SIGMA_10H = 109.0     # ft²/ft³ (≈ 357 m²/m³)
    SIGMA_100H = 30.0     # ft²/ft³ (≈ 98 m²/m³)

    # Chaleur de combustion standard
    H_STD = 8000.0        # BTU/lb (≈ 18 608 kJ/kg)

    # Constantes de réaction
    GAMMA_MAX_COEF = 1.0 / (4.774 * (RHO_P ** 1.5))

    # Constantes de flux de propagation
    XI_COEF_A = 192.0 + 0.2595
    XI_COEF_B = 0.792
    XI_COEF_C = 0.681

    # Constantes de vent
    WIND_C_COEF = 7.47
    WIND_C_EXP = -0.133
    WIND_B_COEF = 0.02526
    WIND_B_EXP = 0.54
    WIND_E_COEF = 0.715
    WIND_E_EXP = -3.59e-4

    # Constantes de pente
    SLOPE_COEF = 5.275
    SLOPE_EXP = -0.3

    # Chaleur de pré-ignition
    Q_IG_BASE = 250.0     # BTU/lb
    Q_IG_COEF = 1116.0    # BTU/lb par unité d'humidité

    # Constante de chauffe effective
    EPSILON_COEF = 138.0  # ft²/ft³

    # Temps de résidence (Anderson 1969)
    TAU_COEF = 384.0      # min·ft²/ft³


# =============================================================================
# CLASSES DE DONNÉES
# =============================================================================

@dataclass
class MoistureInputs:
    """
    Teneurs en humidité des différentes catégories de combustible.
    Toutes les valeurs sont en fraction (0.0 à 1.0), pas en pourcentage.
    """
    m_1h: float = 0.06        # Humidité combustible mort fin
    m_10h: float = 0.07       # Humidité combustible mort moyen
    m_100h: float = 0.08      # Humidité combustible mort gros
    m_live_herb: float = 0.60 # Humidité herbes vivantes
    m_live_woody: float = 0.80 # Humidité ligneux vivant

    def __post_init__(self):
        # CORRECTION v2 : Warnings explicites au lieu de clip silencieux
        for attr in ['m_1h', 'm_10h', 'm_100h']:
            val = getattr(self, attr)
            if val < 0.01:
                warnings.warn(
                    f"MoistureInputs.{attr}={val} est inférieur à 0.01 (1%). "
                    f"Clippé à 0.01. Vérifiez vos données d'entrée.",
                    UserWarning
                )
                val = 0.01
            elif val > 0.60:
                warnings.warn(
                    f"MoistureInputs.{attr}={val} est supérieur à 0.60 (60%). "
                    f"Clippé à 0.60. Vérifiez vos données d'entrée.",
                    UserWarning
                )
                val = 0.60
            setattr(self, attr, val)

        for attr in ['m_live_herb', 'm_live_woody']:
            val = getattr(self, attr)
            if val < 0.30:
                warnings.warn(
                    f"MoistureInputs.{attr}={val} est inférieur à 0.30 (30%). "
                    f"Clippé à 0.30. Vérifiez vos données d'entrée.",
                    UserWarning
                )
                val = 0.30
            elif val > 3.00:
                warnings.warn(
                    f"MoistureInputs.{attr}={val} est supérieur à 3.00 (300%). "
                    f"Clippé à 3.00. Vérifiez vos données d'entrée.",
                    UserWarning
                )
                val = 3.00
            setattr(self, attr, val)


@dataclass
class EnvironmentalConditions:
    """Conditions environnementales mesurées."""
    wind_speed: float = 0.0       # m/s (vent à mi-flamme)
    wind_direction: float = 0.0   # degrés azimut (0 = Nord)
    slope_deg: float = 0.0        # degrés
    aspect_deg: float = 0.0       # degrés azimut (direction de la pente)
    # CORRECTION v2 : Angle vent/pente/propagation
    angle_wind_slope: float = 0.0  # degrés (0 = vent aligné avec pente, 90 = perpendiculaire)

    def __post_init__(self):
        if self.wind_speed < 0.0:
            warnings.warn(
                f"wind_speed={self.wind_speed} est négatif. "
                f"Clippé à 0.0.",
                UserWarning
            )
            self.wind_speed = 0.0

        if self.slope_deg < 0.0:
            warnings.warn(
                f"slope_deg={self.slope_deg} est négatif. "
                f"Clippé à 0.0.",
                UserWarning
            )
            self.slope_deg = 0.0
        elif self.slope_deg > 60.0:
            warnings.warn(
                f"slope_deg={self.slope_deg} > 60° dépasse la validité du modèle Rothermel. "
                f"Clippé à 60.0°.",
                UserWarning
            )
            self.slope_deg = 60.0

        # Normaliser l'angle entre 0 et 180°
        self.angle_wind_slope = abs(self.angle_wind_slope) % 180.0


@dataclass
class RothermelOutput:
    """Sorties complètes du moteur Rothermel."""

    ros: float = 0.0              # m/min
    flame_length: float = 0.0     # m
    fireline_intensity: float = 0.0  # kW/m
    heat_per_unit_area: float = 0.0  # kJ/m²

    # Variables intermédiaires
    I_R: float = 0.0              # kW/m²
    xi: float = 0.0
    phi_w: float = 0.0
    phi_s: float = 0.0
    beta: float = 0.0
    beta_opt: float = 0.0
    gamma: float = 0.0
    w_n: float = 0.0
    eta_M: float = 0.0
    eta_S: float = 0.0
    rho_b: float = 0.0
    epsilon: float = 0.0
    Q_ig: float = 0.0
    A: float = 0.0

    # CORRECTION v2 : Nouvelles variables
    h_avg: float = 0.0            # Chaleur de combustion moyenne pondérée (BTU/lb)
    tau: float = 0.0              # Temps de résidence (min)
    angle_correction: float = 1.0  # Facteur de correction d'angle vent/pente

    w_1h_adj_lb: float = 0.0
    w_live_herb_adj_lb: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'ros': round(self.ros, 4),
            'flame_length': round(self.flame_length, 4),
            'fireline_intensity': round(self.fireline_intensity, 4),
            'heat_per_unit_area': round(self.heat_per_unit_area, 4),
            'I_R': round(self.I_R, 4),
            'xi': round(self.xi, 6),
            'phi_w': round(self.phi_w, 4),
            'phi_s': round(self.phi_s, 4),
            'beta': round(self.beta, 6),
            'beta_opt': round(self.beta_opt, 6),
            'gamma': round(self.gamma, 6),
            'w_n': round(self.w_n, 4),
            'eta_M': round(self.eta_M, 4),
            'eta_S': round(self.eta_S, 4),
            'rho_b': round(self.rho_b, 4),
            'epsilon': round(self.epsilon, 6),
            'Q_ig': round(self.Q_ig, 4),
            'A': round(self.A, 2),
            'h_avg': round(self.h_avg, 2),
            'tau': round(self.tau, 4),
            'angle_correction': round(self.angle_correction, 4),
        }


# =============================================================================
# MOTEUR ROTHERMEL (v2 corrigé)
# =============================================================================

class RothermelEngine:
    """
    Moteur de calcul du modèle de Rothermel (v2 corrigé).
    Entrées SI → conversion impérial → calcul → sorties SI.
    """

    def __init__(self, fuel_model: FuelModel):
        self.fuel_si = fuel_model
        self.const = RothermelConstants()
        self.uc = UnitConversions()
        self.fuel = self._convert_fuel_to_imperial(fuel_model)

    def _convert_fuel_to_imperial(self, fm: FuelModel) -> Dict:
        """Convertit les paramètres du fuel model en unités impériales."""
        return {
            'w_1h': fm.w_1h * self.uc.KG_M2_TO_LB_FT2,
            'w_10h': fm.w_10h * self.uc.KG_M2_TO_LB_FT2,
            'w_100h': fm.w_100h * self.uc.KG_M2_TO_LB_FT2,
            'w_live_herb': fm.w_live_herb * self.uc.KG_M2_TO_LB_FT2,
            'w_live_woody': fm.w_live_woody * self.uc.KG_M2_TO_LB_FT2,
            'delta': fm.delta * self.uc.M_TO_FT,
            'sigma_1h': fm.sigma_1h * self.uc.SAV_M_TO_FT,
            'sigma_live_herb': fm.sigma_live_herb * self.uc.SAV_M_TO_FT,
            'sigma_live_woody': fm.sigma_live_woody * self.uc.SAV_M_TO_FT,
            'h_dead': fm.h_dead * self.uc.KJ_KG_TO_BTU_LB,
            'h_live': fm.h_live * self.uc.KJ_KG_TO_BTU_LB,
            'mx': fm.mx / 100.0,
            'is_dynamic': fm.is_dynamic,
        }

    def _adjust_fuel_loads(self, moisture: MoistureInputs) -> Tuple[float, float, float, float, float]:
        """Ajuste les charges selon le type de modèle (dynamique ou statique)."""
        if self.fuel['is_dynamic'] and self.fuel['w_live_herb'] > 0:
            w_live_herb_adj, w_1h_adj = compute_dynamic_herb_load(
                self.fuel_si, moisture.m_live_herb * 100
            )
            return (
                w_1h_adj * self.uc.KG_M2_TO_LB_FT2,
                self.fuel['w_10h'],
                self.fuel['w_100h'],
                w_live_herb_adj * self.uc.KG_M2_TO_LB_FT2,
                self.fuel['w_live_woody']
            )
        else:
            return (
                self.fuel['w_1h'],
                self.fuel['w_10h'],
                self.fuel['w_100h'],
                self.fuel['w_live_herb'],
                self.fuel['w_live_woody']
            )

    def _compute_sigma_weighted(self, w_1h: float, w_10h: float, w_100h: float,
                                 w_live_herb: float, w_live_woody: float) -> float:
        """Surface spécifique pondérée (ft²/ft³)."""
        total_load = w_1h + w_10h + w_100h + w_live_herb + w_live_woody
        if total_load == 0:
            return 0.0
        weighted_sigma = (
            w_1h * self.fuel['sigma_1h'] +
            w_10h * self.const.SIGMA_10H +
            w_100h * self.const.SIGMA_100H +
            w_live_herb * self.fuel['sigma_live_herb'] +
            w_live_woody * self.fuel['sigma_live_woody']
        )
        return weighted_sigma / total_load

    def _compute_packing_ratio(self, total_load: float) -> float:
        """Packing ratio (sans unité)."""
        if self.fuel['delta'] <= 0 or total_load <= 0:
            return 0.0
        rho_b = total_load / self.fuel['delta']
        return rho_b / self.const.RHO_P

    def _compute_optimal_packing_ratio(self, sigma: float) -> float:
        """Packing ratio optimal."""
        if sigma <= 0:
            return 0.0
        return 3.348 * (sigma ** (-0.8189))

    def _compute_reaction_velocity(self, sigma: float) -> float:
        """Vitesse de réaction (min⁻¹)."""
        if sigma <= 0:
            return 0.0
        sigma_15 = sigma ** 1.5
        return sigma_15 / (495.0 + 0.0594 * sigma_15)

    def _compute_net_fuel_loading(self, total_load: float, sigma: float) -> float:
        """Charge nette de combustible (lb/ft²)."""
        if total_load <= 0 or sigma <= 0:
            return 0.0
        return total_load / (1.0 + 0.6037 * (sigma ** (-0.54)))

    def _compute_moisture_damping(self, moisture: MoistureInputs, 
                                   w_1h: float, w_10h: float, w_100h: float,
                                   w_live_herb: float, w_live_woody: float,
                                   total_load: float) -> float:
        """Coefficient d'amortissement dû à l'humidité."""
        mx = self.fuel['mx']

        w_dead = w_1h + w_10h + w_100h
        if w_dead > 0 and mx > 0:
            m_dead_avg = (w_1h * moisture.m_1h + 
                          w_10h * moisture.m_10h + 
                          w_100h * moisture.m_100h) / w_dead
            r_dead = m_dead_avg / mx
            r_dead = np.clip(r_dead, 0.0, 1.0)
            eta_M_dead = max(0.0, 1.0 - 2.59*r_dead + 5.11*r_dead**2 - 3.52*r_dead**3)
        else:
            eta_M_dead = 0.0

        w_live = w_live_herb + w_live_woody
        if w_live > 0:
            mx_live = 2.5 * mx
            m_live_avg = (w_live_herb * moisture.m_live_herb + 
                          w_live_woody * moisture.m_live_woody) / w_live
            r_live = m_live_avg / mx_live
            r_live = np.clip(r_live, 0.0, 1.0)
            eta_M_live = max(0.0, 1.0 - 2.59*r_live + 5.11*r_live**2 - 3.52*r_live**3)
        else:
            eta_M_live = 1.0

        if total_load > 0:
            eta_M = (w_dead * eta_M_dead + w_live * eta_M_live) / total_load
        else:
            eta_M = 0.0

        return max(0.0, eta_M)

    def _compute_mineral_damping(self) -> float:
        """Coefficient d'amortissement minéral."""
        return 0.174 * (self.const.S_E ** (-0.19))

    def _compute_weighted_heat_of_combustion(
        self, w_1h: float, w_10h: float, w_100h: float,
        w_live_herb: float, w_live_woody: float, total_load: float
    ) -> float:
        """
        CORRECTION v2 : Pondération h_dead/h_live par les charges.

        Scott & Burgan (2005) prévoient une pondération par les charges
        mortes et vivantes. Si la charge vivante est majoritaire,
        h_live doit être pris en compte.
        """
        w_dead = w_1h + w_10h + w_100h
        w_live = w_live_herb + w_live_woody

        if total_load <= 0:
            return self.fuel['h_dead']

        # Pondération par les charges
        h_avg = (w_dead * self.fuel['h_dead'] + w_live * self.fuel['h_live']) / total_load
        return h_avg

    def _compute_reaction_intensity(
        self, gamma: float, w_n: float, 
        eta_M: float, eta_S: float, h_avg: float
    ) -> float:
        """
        Intensité de réaction (BTU/ft²/min).

        CORRECTION v2 : Utilise h_avg (pondéré) au lieu de h_dead fixe.
        """
        return gamma * w_n * h_avg * eta_M * eta_S

    def _compute_propagating_flux_ratio(self, beta: float, sigma: float) -> float:
        """Rapport de flux de propagation (sans unité)."""
        if sigma <= 0:
            return 0.0
        numerator = np.exp((0.792 + 0.681 * np.sqrt(sigma)) * (beta + 0.1))
        denominator = 192.0 + 0.2595 * sigma
        return numerator / denominator

    def _compute_wind_coefficient(
        self, wind_speed: float, beta: float, beta_opt: float, sigma: float
    ) -> float:
        """
        Coefficient de vent (sans unité).

        CORRECTION v2 : Utilise (beta / beta_opt)**(-E) au lieu de beta**(-E).

        Formule originale Rothermel (1972), équation 79 (Andrews 2018) :
        phi_w = C * (wind_ft_min ** B) * (beta / beta_opt)**(-E)

        Avant (bug) : phi_w = C * (wind_ft_min ** B) / (beta ** E)
        """
        if wind_speed <= 0 or beta <= 0 or beta_opt <= 0 or sigma <= 0:
            return 0.0

        wind_ft_min = wind_speed * self.uc.M_TO_FT * 60.0

        C = self.const.WIND_C_COEF * np.exp(self.const.WIND_C_EXP * (sigma ** 0.55))
        B = self.const.WIND_B_COEF * (sigma ** self.const.WIND_B_EXP)
        E = self.const.WIND_E_COEF * np.exp(self.const.WIND_E_EXP * sigma)

        # CORRECTION v2 : (beta / beta_opt)**(-E) — formule originale Rothermel
        phi_w = C * (wind_ft_min ** B) * ((beta / beta_opt) ** (-E))
        return max(0.0, phi_w)

    def _compute_slope_coefficient(self, slope_deg: float, beta: float) -> float:
        """Coefficient de pente (sans unité)."""
        if beta <= 0:
            return 0.0
        slope_rad = np.radians(slope_deg)
        tan_slope = np.tan(slope_rad)
        return self.const.SLOPE_COEF * (beta ** self.const.SLOPE_EXP) * (tan_slope ** 2)

    def _compute_angle_correction(self, angle_wind_slope: float) -> float:
        """
        CORRECTION v2 : Facteur de correction pour l'angle entre vent et pente.

        Quand le vent n'est pas aligné avec la direction de propagation,
        l'effet combiné vent+pente est réduit.

        angle_wind_slope : degrés (0 = vent aligné avec pente, 90 = perpendiculaire)

        Référence : Rothermel (1972), section sur la propagation non alignée.
        Catchpole et al. (1982) — vector addition of wind and slope effects.
        """
        if angle_wind_slope <= 0:
            return 1.0

        # Correction cosinus simple (approximation)
        # Effet maximal quand aligné (0°), nul quand perpendiculaire (90°)
        angle_rad = np.radians(angle_wind_slope)
        correction = np.cos(angle_rad)

        # Limiter à un minimum de 0.3 pour éviter extinction artificielle
        return max(0.3, correction)

    def _compute_bulk_density(self, total_load: float) -> float:
        """Densité apparente (lb/ft³)."""
        if self.fuel['delta'] <= 0:
            return 0.0
        return total_load / self.fuel['delta']

    def _compute_effective_heating_number(self, sigma: float) -> float:
        """Nombre de chauffe effective (sans unité)."""
        if sigma <= 0:
            return 0.0
        return np.exp(-self.const.EPSILON_COEF / sigma)

    def _compute_heat_of_preignition(self, moisture: MoistureInputs, 
                                       w_1h: float, w_10h: float, w_100h: float,
                                       w_live_herb: float, w_live_woody: float,
                                       total_load: float) -> float:
        """Chaleur de pré-ignition (BTU/lb)."""
        if total_load <= 0:
            return self.const.Q_IG_BASE

        m_avg = (w_1h * moisture.m_1h + 
                 w_10h * moisture.m_10h + 
                 w_100h * moisture.m_100h +
                 w_live_herb * moisture.m_live_herb +
                 w_live_woody * moisture.m_live_woody) / total_load

        return self.const.Q_IG_BASE + self.const.Q_IG_COEF * m_avg

    def compute(self, moisture: MoistureInputs, 
                conditions: EnvironmentalConditions) -> RothermelOutput:
        """
        Calcule le taux de propagation et les variables associées.

        ROS = (I_R × xi × (1 + phi_w + phi_s)) / (rho_b × epsilon × Q_ig)

        CORRECTION v2 :
        - h_avg pondéré (h_dead/h_live par charges)
        - phi_w avec (beta/beta_opt)**(-E)
        - Angle vent/pente pris en compte
        - I_b aligné sur Byram : I_b = I_R × tau avec tau = 384/sigma

        Returns:
            RothermelOutput avec toutes les valeurs en SI
        """
        output = RothermelOutput()

        # 1. Ajustement des charges
        w_1h, w_10h, w_100h, w_live_herb, w_live_woody = self._adjust_fuel_loads(moisture)
        output.w_1h_adj_lb = w_1h
        output.w_live_herb_adj_lb = w_live_herb

        total_load = w_1h + w_10h + w_100h + w_live_herb + w_live_woody

        if total_load <= 0:
            return output

        # 2. Surface spécifique pondérée
        sigma = self._compute_sigma_weighted(w_1h, w_10h, w_100h, w_live_herb, w_live_woody)
        output.A = sigma

        # 3. Packing ratios
        beta = self._compute_packing_ratio(total_load)
        beta_opt = self._compute_optimal_packing_ratio(sigma)
        output.beta = beta
        output.beta_opt = beta_opt

        # 4. Vitesse de réaction
        gamma = self._compute_reaction_velocity(sigma)
        output.gamma = gamma

        # 5. Charge nette
        w_n = self._compute_net_fuel_loading(total_load, sigma)
        output.w_n = w_n

        # 6. Coefficients d'amortissement
        eta_M = self._compute_moisture_damping(moisture, w_1h, w_10h, w_100h,
                                                w_live_herb, w_live_woody, total_load)
        eta_S = self._compute_mineral_damping()
        output.eta_M = eta_M
        output.eta_S = eta_S

        # 7. Chaleur de combustion moyenne pondérée (CORRECTION v2)
        h_avg = self._compute_weighted_heat_of_combustion(
            w_1h, w_10h, w_100h, w_live_herb, w_live_woody, total_load
        )
        output.h_avg = h_avg

        # 8. Intensité de réaction (BTU/ft²/min) — CORRECTION v2 : h_avg au lieu de h_dead
        I_R_imp = self._compute_reaction_intensity(gamma, w_n, eta_M, eta_S, h_avg)

        # 9. Rapport de flux de propagation
        xi = self._compute_propagating_flux_ratio(beta, sigma)
        output.xi = xi

        # 10. Coefficients de vent et pente
        # CORRECTION v2 : phi_w avec (beta/beta_opt)**(-E)
        phi_w = self._compute_wind_coefficient(
            conditions.wind_speed, beta, beta_opt, sigma
        )
        phi_s = self._compute_slope_coefficient(conditions.slope_deg, beta)
        output.phi_w = phi_w
        output.phi_s = phi_s

        # 11. Correction d'angle vent/pente (CORRECTION v2)
        angle_correction = self._compute_angle_correction(conditions.angle_wind_slope)
        output.angle_correction = angle_correction

        # 12. Densité apparente et chauffe effective
        rho_b = self._compute_bulk_density(total_load)
        epsilon = self._compute_effective_heating_number(sigma)
        output.rho_b = rho_b
        output.epsilon = epsilon

        # 13. Chaleur de pré-ignition
        Q_ig = self._compute_heat_of_preignition(moisture, w_1h, w_10h, w_100h,
                                                   w_live_herb, w_live_woody, total_load)
        output.Q_ig = Q_ig

        # 14. Calcul du ROS (ft/min)
        denominator = rho_b * epsilon * Q_ig
        if denominator > 0:
            # CORRECTION v2 : angle_correction appliqué au terme combiné
            ros_ft_min = (I_R_imp * xi * (1.0 + angle_correction * (phi_w + phi_s))) / denominator
            ros_ft_min = max(0.0, ros_ft_min)
        else:
            ros_ft_min = 0.0

        # Conversion ft/min → m/min
        output.ros = ros_ft_min * self.uc.FT_MIN_TO_M_MIN

        # 15. Temps de résidence tau (min) — Anderson 1969
        # tau = 384 / sigma (minutes)
        # Référence : Andrews 2018, section 4.2
        if sigma > 0:
            output.tau = self.const.TAU_COEF / sigma
        else:
            output.tau = 0.0

        # 16. Intensité ligne de feu (kW/m) — CORRECTION v2 : Byram standard
        # I_b = I_R [BTU/ft²/min] × tau [min] = BTU/ft²
        # Conversion : 1 BTU/ft²/s = 11.357 kW/m
        if output.tau > 0:
            I_b_btu_ft2 = I_R_imp * output.tau  # BTU/ft²
            # Conversion en kW/m (1 BTU/ft²/s = 11.357 kW/m, donc diviser par 60 pour /min)
            output.fireline_intensity = (I_b_btu_ft2 / 60.0) * 11.357

        # 17. Longueur de flamme (ft → m)
        if output.fireline_intensity > 0:
            I_b_btu_ft_s = output.fireline_intensity / 11.357
            flame_ft = 0.0775 * (I_b_btu_ft_s ** 0.46)
            output.flame_length = flame_ft * self.uc.FT_TO_M

        # 18. Chaleur par unité de surface (BTU/ft² → kJ/m²)
        if beta > 0:
            hpua_imp = I_R_imp * self.fuel['delta'] / beta
            output.heat_per_unit_area = hpua_imp * 11.357

        # 19. Intensité de réaction en kW/m²
        output.I_R = I_R_imp * 0.189

        return output

    def compute_danger_level(self, ros: float, flame_length: float) -> str:
        """Détermine le niveau de danger."""
        if ros < 0.1:
            return "NUL"
        elif ros < 0.5 and flame_length < 0.5:
            return "FAIBLE"
        elif ros < 2.0 and flame_length < 1.0:
            return "MODÉRÉ"
        elif ros < 5.0 and flame_length < 2.5:
            return "ÉLEVÉ"
        elif ros < 10.0 and flame_length < 4.0:
            return "TRÈS ÉLEVÉ"
        else:
            return "EXTRÊME"

    def compute_spotting_distance(self, flame_length: float, wind_speed: float) -> float:
        """Estime la distance de spotting en mètres."""
        if flame_length < 1.0 or wind_speed < 3.0:
            return 0.0
        return 10.0 * flame_length * (wind_speed ** 0.5)


# =============================================================================
# PIPELINE COMPLET
# =============================================================================

class BurnTrackRothermel:
    """Pipeline complet : sélection du fuel model + calcul Rothermel."""

    def __init__(self, fuel_model_code: str):
        from fuel_models import get_fuel_model
        self.fuel_model = get_fuel_model(fuel_model_code)
        if self.fuel_model is None:
            raise ValueError(f"Fuel model '{fuel_model_code}' non trouvé")
        self.engine = RothermelEngine(self.fuel_model)

    def predict(self,
                temp_air: float,
                rh: float,
                wind_speed: float,
                slope_deg: float = 0.0,
                angle_wind_slope: float = 0.0,  # CORRECTION v2
                live_herb_moisture: Optional[float] = None,
                live_woody_moisture: Optional[float] = None,
                dead_1h_moisture: Optional[float] = None,
                dead_10h_moisture: Optional[float] = None,
                dead_100h_moisture: Optional[float] = None) -> Dict:
        """
        Prédiction simplifiée à partir des conditions météo.
        Les humidités non fournies sont estimées à partir de T° et HR.

        CORRECTION v2 : Nouveau paramètre angle_wind_slope (degrés).
        """
        if dead_1h_moisture is None:
            es = 0.6108 * np.exp(17.27 * temp_air / (temp_air + 237.3))
            vpd = es * (1.0 - rh / 100.0)
            dfmc = np.clip(30.0 - 2.5 * vpd - 0.1 * temp_air, 3.0, 40.0)
            dead_1h_moisture = dfmc / 100.0

        if dead_10h_moisture is None:
            dead_10h_moisture = dead_1h_moisture + 0.02
        if dead_100h_moisture is None:
            dead_100h_moisture = dead_1h_moisture + 0.04

        if live_herb_moisture is None:
            live_herb_moisture = 0.30 + (100.0 - rh) / 200.0

        if live_woody_moisture is None:
            live_woody_moisture = 0.60 + (100.0 - rh) / 500.0

        moisture = MoistureInputs(
            m_1h=dead_1h_moisture,
            m_10h=dead_10h_moisture,
            m_100h=dead_100h_moisture,
            m_live_herb=live_herb_moisture,
            m_live_woody=live_woody_moisture
        )

        conditions = EnvironmentalConditions(
            wind_speed=wind_speed,
            slope_deg=slope_deg,
            angle_wind_slope=angle_wind_slope  # CORRECTION v2
        )

        output = self.engine.compute(moisture, conditions)
        danger = self.engine.compute_danger_level(output.ros, output.flame_length)
        spotting = self.engine.compute_spotting_distance(output.flame_length, wind_speed)

        return {
            'fuel_model': self.fuel_model.code,
            'fuel_name': self.fuel_model.name,
            'ros_m_min': round(output.ros, 3),
            'flame_length_m': round(output.flame_length, 2),
            'fireline_intensity_kW_m': round(output.fireline_intensity, 2),
            'heat_per_unit_area_kJ_m2': round(output.heat_per_unit_area, 2),
            'danger_level': danger,
            'spotting_distance_m': round(spotting, 1),
            'I_R_kW_m2': round(output.I_R, 2),
            'phi_w': round(output.phi_w, 3),
            'phi_s': round(output.phi_s, 3),
            'beta': round(output.beta, 4),
            'eta_M': round(output.eta_M, 4),
            'w_total_kg_m2': round(self.fuel_model.w_total, 3),
            'mx_percent': self.fuel_model.mx,
            # CORRECTION v2 : Nouvelles sorties
            'h_avg_BTU_lb': round(output.h_avg, 2),
            'tau_min': round(output.tau, 4),
            'angle_correction': round(output.angle_correction, 4),
            'beta_opt': round(output.beta_opt, 4),
        }


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ROTHERMEL ENGINE v2 (CORRIGÉ) - TEST")
    print("=" * 70)

    test_cases = [
        ("AF_STEPPE", 38, 20, 4.0, 15, 0),
        ("AF_MIOMBO", 32, 45, 3.0, 5, 0),
        ("AF_FYNBOS", 28, 35, 6.0, 20, 0),
        ("AF_SAHEL_GRASS", 42, 15, 5.0, 0, 0),
        ("GR1", 35, 25, 3.0, 0, 0),
        ("AF_GRASSLAND_FERTILE", 35, 30, 4.0, 0, 0),
    ]

    for fuel_code, temp, rh, wind, slope, angle in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Fuel: {fuel_code} | T={temp}°C | RH={rh}% | Vent={wind}m/s | Pente={slope}° | Angle={angle}°")
        print("─" * 70)

        try:
            predictor = BurnTrackRothermel(fuel_code)
            result = predictor.predict(
                temp_air=temp,
                rh=rh,
                wind_speed=wind,
                slope_deg=slope,
                angle_wind_slope=angle
            )

            print(f"  ROS:              {result['ros_m_min']} m/min")
            print(f"  Flame length:     {result['flame_length_m']} m")
            print(f"  Fireline intensity: {result['fireline_intensity_kW_m']} kW/m")
            print(f"  Heat/area:        {result['heat_per_unit_area_kJ_m2']} kJ/m²")
            print(f"  Danger level:     {result['danger_level']}")
            print(f"  Spotting distance: {result['spotting_distance_m']} m")
            print(f"  I_R:              {result['I_R_kW_m2']} kW/m²")
            print(f"  phi_w:            {result['phi_w']}")
            print(f"  phi_s:            {result['phi_s']}")
            print(f"  beta:             {result['beta']}")
            print(f"  beta_opt:         {result['beta_opt']}")
            print(f"  eta_M:            {result['eta_M']}")
            print(f"  h_avg:            {result['h_avg_BTU_lb']} BTU/lb")
            print(f"  tau:              {result['tau_min']} min")
            print(f"  angle_correction: {result['angle_correction']}")

        except Exception as e:
            print(f"  ERREUR: {e}")

    print("\n" + "=" * 70)
    print("TEST - Comparaison v1 vs v2 (coefficient de vent)")
    print("=" * 70)

    predictor = BurnTrackRothermel("AF_STEPPE")
    for wind_test in [0, 2, 4, 6, 8, 10]:
        result = predictor.predict(
            temp_air=38, rh=20, wind_speed=wind_test, 
            slope_deg=15, angle_wind_slope=0
        )
        print(f"  Vent={wind_test:2d}m/s → ROS={result['ros_m_min']:6.3f} m/min | "
              f"Flame={result['flame_length_m']:5.2f}m | "
              f"phi_w={result['phi_w']:7.3f} | "
              f"Danger={result['danger_level']}")

    print("\n" + "=" * 70)
    print("TEST - Sensibilité à l'angle vent/pente")
    print("=" * 70)

    predictor = BurnTrackRothermel("AF_FYNBOS")
    for angle_test in [0, 15, 30, 45, 60, 75, 90]:
        result = predictor.predict(
            temp_air=28, rh=35, wind_speed=6.0, 
            slope_deg=20, angle_wind_slope=angle_test
        )
        print(f"  Angle={angle_test:3d}° → ROS={result['ros_m_min']:6.3f} m/min | "
              f"angle_correction={result['angle_correction']:5.3f} | "
              f"Danger={result['danger_level']}")

    print("\n" + "=" * 70)
    print("TEST - Warnings explicites")
    print("=" * 70)

    # Test warning humidité hors bornes
    print("\nTest : m_1h = 0.005 (trop bas)")
    m = MoistureInputs(m_1h=0.005)
    print(f"  m_1h après correction = {m.m_1h}")

    print("\nTest : slope_deg = 70° (trop élevé)")
    env = EnvironmentalConditions(slope_deg=70)
    print(f"  slope_deg après correction = {env.slope_deg}")