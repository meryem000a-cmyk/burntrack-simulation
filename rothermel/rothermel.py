"""
rothermel_engine.py
====================
Moteur de calcul Rothermel pour le pipeline BurnTrack.

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

Unités d'entrée (SI) :
- Charges : kg/m²
- Profondeur : m
- SAV : m²/m³ = 1/m
- Chaleur : kJ/kg
- Vent : m/s
- Pente : degrés

Unités de sortie (SI) :
- ROS : m/min
- Intensité : kW/m
- Longueur flamme : m
"""

import numpy as np
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
        for attr in ['m_1h', 'm_10h', 'm_100h']:
            val = getattr(self, attr)
            setattr(self, attr, np.clip(val, 0.01, 0.60))
        for attr in ['m_live_herb', 'm_live_woody']:
            val = getattr(self, attr)
            setattr(self, attr, np.clip(val, 0.30, 3.00))


@dataclass
class EnvironmentalConditions:
    """Conditions environnementales mesurées."""
    wind_speed: float = 0.0       # m/s (vent à mi-flamme)
    wind_direction: float = 0.0   # degrés
    slope_deg: float = 0.0        # degrés
    aspect_deg: float = 0.0     # degrés
    
    def __post_init__(self):
        self.wind_speed = max(0.0, self.wind_speed)
        self.slope_deg = np.clip(self.slope_deg, 0.0, 60.0)


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
        }


# =============================================================================
# MOTEUR ROTHERMEL
# =============================================================================

class RothermelEngine:
    """
    Moteur de calcul du modèle de Rothermel.
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
    
    def _compute_reaction_intensity(self, gamma: float, w_n: float, 
                                     eta_M: float, eta_S: float) -> float:
        """Intensité de réaction (BTU/ft²/min)."""
        h_avg = self.fuel['h_dead']
        return gamma * w_n * h_avg * eta_M * eta_S
    
    def _compute_propagating_flux_ratio(self, beta: float, sigma: float) -> float:
        """Rapport de flux de propagation (sans unité)."""
        if sigma <= 0:
            return 0.0
        numerator = np.exp((0.792 + 0.681 * np.sqrt(sigma)) * (beta + 0.1))
        denominator = 192.0 + 0.2595 * sigma
        return numerator / denominator
    
    def _compute_wind_coefficient(self, wind_speed: float, beta: float, sigma: float) -> float:
        """Coefficient de vent (sans unité)."""
        if wind_speed <= 0 or beta <= 0 or sigma <= 0:
            return 0.0
        
        wind_ft_min = wind_speed * self.uc.M_TO_FT * 60.0
        
        C = self.const.WIND_C_COEF * np.exp(self.const.WIND_C_EXP * (sigma ** 0.55))
        B = self.const.WIND_B_COEF * (sigma ** self.const.WIND_B_EXP)
        E = self.const.WIND_E_COEF * np.exp(self.const.WIND_E_EXP * sigma)
        
        phi_w = C * (wind_ft_min ** B) / (beta ** E)
        return max(0.0, phi_w)
    
    def _compute_slope_coefficient(self, slope_deg: float, beta: float) -> float:
        """Coefficient de pente (sans unité)."""
        if beta <= 0:
            return 0.0
        slope_rad = np.radians(slope_deg)
        tan_slope = np.tan(slope_rad)
        return self.const.SLOPE_COEF * (beta ** self.const.SLOPE_EXP) * (tan_slope ** 2)
    
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
        
        # 7. Intensité de réaction (BTU/ft²/min)
        I_R_imp = self._compute_reaction_intensity(gamma, w_n, eta_M, eta_S)
        
        # 8. Rapport de flux de propagation
        xi = self._compute_propagating_flux_ratio(beta, sigma)
        output.xi = xi
        
        # 9. Coefficients de vent et pente
        phi_w = self._compute_wind_coefficient(conditions.wind_speed, beta, sigma)
        phi_s = self._compute_slope_coefficient(conditions.slope_deg, beta)
        output.phi_w = phi_w
        output.phi_s = phi_s
        
        # 10. Densité apparente et chauffe effective
        rho_b = self._compute_bulk_density(total_load)
        epsilon = self._compute_effective_heating_number(sigma)
        output.rho_b = rho_b
        output.epsilon = epsilon
        
        # 11. Chaleur de pré-ignition
        Q_ig = self._compute_heat_of_preignition(moisture, w_1h, w_10h, w_100h,
                                                   w_live_herb, w_live_woody, total_load)
        output.Q_ig = Q_ig
        
        # 12. Calcul du ROS (ft/min)
        denominator = rho_b * epsilon * Q_ig
        if denominator > 0:
            ros_ft_min = (I_R_imp * xi * (1.0 + phi_w + phi_s)) / denominator
            ros_ft_min = max(0.0, ros_ft_min)
        else:
            ros_ft_min = 0.0
        
        # Conversion ft/min → m/min
        output.ros = ros_ft_min * self.uc.FT_MIN_TO_M_MIN
        
        # 13. Intensité ligne de feu (BTU/ft/s → kW/m)
        if beta > 0:
            I_b_imp = I_R_imp * self.fuel['delta'] / (60.0 * beta) * (1.0 + phi_w + phi_s)
            output.fireline_intensity = I_b_imp * 3.461
        
        # 14. Longueur de flamme (ft → m)
        if output.fireline_intensity > 0:
            I_b_btu_ft_s = output.fireline_intensity / 3.461
            flame_ft = 0.0775 * (I_b_btu_ft_s ** 0.46)
            output.flame_length = flame_ft * self.uc.FT_TO_M
        
        # 15. Chaleur par unité de surface (BTU/ft² → kJ/m²)
        if beta > 0:
            hpua_imp = I_R_imp * self.fuel['delta'] / beta
            output.heat_per_unit_area = hpua_imp * 11.357
        
        # 16. Intensité de réaction en kW/m²
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
                live_herb_moisture: Optional[float] = None,
                live_woody_moisture: Optional[float] = None,
                dead_1h_moisture: Optional[float] = None,
                dead_10h_moisture: Optional[float] = None,
                dead_100h_moisture: Optional[float] = None) -> Dict:
        """
        Prédiction simplifiée à partir des conditions météo.
        Les humidités non fournies sont estimées à partir de T° et HR.
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
            slope_deg=slope_deg
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
        }


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ROTHERMEL ENGINE - TEST")
    print("=" * 70)
    
    test_cases = [
        ("AF_STEPPE", 38, 20, 4.0, 15),
        ("AF_MIOMBO", 32, 45, 3.0, 5),
        ("AF_FYNBOS", 28, 35, 6.0, 20),
        ("AF_SAHEL_GRASS", 42, 15, 5.0, 0),
        ("GR1", 35, 25, 3.0, 0),
        ("AF_GRASSLAND_FERTILE", 35, 30, 4.0, 0),
    ]
    
    for fuel_code, temp, rh, wind, slope in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Fuel: {fuel_code} | T={temp}°C | RH={rh}% | Vent={wind}m/s | Pente={slope}°")
        print("─" * 70)
        
        try:
            predictor = BurnTrackRothermel(fuel_code)
            result = predictor.predict(
                temp_air=temp,
                rh=rh,
                wind_speed=wind,
                slope_deg=slope
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
            print(f"  eta_M:            {result['eta_M']}")
            
        except Exception as e:
            print(f"  ERREUR: {e}")
    
    print("\n" + "=" * 70)
    print("TEST - Sensibilité à l'humidité (AF_STEPPE)")
    print("=" * 70)
    
    predictor = BurnTrackRothermel("AF_STEPPE")
    for rh_test in [10, 20, 30, 40, 50, 60]:
        result = predictor.predict(temp_air=38, rh=rh_test, wind_speed=4.0, slope_deg=15)
        print(f"  RH={rh_test:2d}% → ROS={result['ros_m_min']:6.3f} m/min | "
              f"Flame={result['flame_length_m']:5.2f}m | "
              f"Danger={result['danger_level']}")
    
    print("\n" + "=" * 70)
    print("TEST - Sensibilité au vent (AF_MIOMBO)")
    print("=" * 70)
    
    predictor = BurnTrackRothermel("AF_MIOMBO")
    for wind_test in [0, 2, 4, 6, 8, 10]:
        result = predictor.predict(temp_air=32, rh=45, wind_speed=wind_test, slope_deg=5)
        print(f"  Vent={wind_test:2d}m/s → ROS={result['ros_m_min']:6.3f} m/min | "
              f"Flame={result['flame_length_m']:5.2f}m | "
              f"Danger={result['danger_level']}")