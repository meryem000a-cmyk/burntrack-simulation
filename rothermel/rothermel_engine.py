"""
Rothermel Engine v3 — Moteur de propagation de feu de surface
Corrections appliquées :
1. gamma complet avec facteur Albini (β/β_opt)^A · exp(A(1-β/β_opt))
2. Composition vectorielle vent/pente (Catchpole 1982)
3. I_b conversion Byram correcte (I_B = I_R × τ × ROS)
4. Garde-fou phi_w avec amortissement logarithmique + saturation douce
5. Mx,live affiné selon Albini (1976)
6. Ordre compute() : beta et beta_opt avant gamma
7. Pente : tan(θ)² au lieu de (%)², sans beta_opt au dénominateur
8. Reaction intensity : 0.189 kW/m² par BTU/ft²/min
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Configuration logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


@dataclass
class FuelModel:
    """Modèle de combustible pour Rothermel.

    Compatible avec fuel_models.py : sigma_live_herb + sigma_live_woody
    au lieu de sigma_live unique. st/se optionnels (valeurs par défaut).
    """
    name: str
    w_1h: float = 0.0          # kg/m², charge mort fin (1h)
    w_10h: float = 0.0         # kg/m², charge mort moyen (10h)
    w_100h: float = 0.0        # kg/m², charge mort gros (100h)
    w_live_herb: float = 0.0   # kg/m², charge vivant herbacé
    w_live_woody: float = 0.0  # kg/m², charge vivant ligneux
    sigma_1h: float = 0.0      # m²/m³, SAV mort fin
    sigma_10h: float = 0.0     # m²/m³, SAV mort moyen
    sigma_100h: float = 0.0    # m²/m³, SAV mort gros
    sigma_live_herb: float = 0.0   # m²/m³, SAV vivant herbacé
    sigma_live_woody: float = 0.0  # m²/m³, SAV vivant ligneux
    delta: float = 0.0         # m, profondeur du fuel bed
    mx: float = 0.0            # % (pourcent), humidité d'extinction morts
    h_dead: float = 0.0        # kJ/kg, chaleur de combustion morts
    h_live: float = 0.0        # kJ/kg, chaleur de combustion vivant
    st: float = 0.0555         # fraction, contenu minéral total (optionnel)
    se: float = 0.01           # fraction, contenu minéral effectif (optionnel)


@dataclass
class MoistureInputs:
    """Humidités des combustibles."""
    m_1h: float = 0.05
    m_10h: float = 0.05
    m_100h: float = 0.05
    m_live_herb: float = 0.5
    m_live_woody: float = 0.5

    def __post_init__(self):
        for attr in ['m_1h', 'm_10h', 'm_100h', 'm_live_herb', 'm_live_woody']:
            val = getattr(self, attr)
            if val < 0 or val > 1:
                logging.warning(f"{attr}={val} hors plage [0,1]. Clippé.")
                setattr(self, attr, np.clip(val, 0.0, 1.0))


@dataclass
class EnvironmentalConditions:
    """Conditions environnementales."""
    wind_speed: float = 0.0      # m/s, vitesse du vent à mi-flamme
    slope_pct: float = 0.0       # %, pente
    angle_wind_slope: float = 0.0  # degrés, angle relatif vent/pente

    def __post_init__(self):
        if self.wind_speed < 0:
            logging.warning(f"wind_speed={self.wind_speed} négatif. Mis à 0.")
            self.wind_speed = 0.0
        if self.slope_pct < 0:
            logging.warning(f"slope_pct={self.slope_pct} négatif. Mis à 0.")
            self.slope_pct = 0.0


@dataclass
class RothermelOutput:
    """Sortie du moteur Rothermel."""
    ros: float = 0.0                    # m/min, vitesse de propagation
    fireline_intensity: float = 0.0     # kW/m, intensité ligne de feu
    flame_length: float = 0.0           # m, longueur de flamme
    reaction_intensity: float = 0.0     # kW/m², intensité de réaction
    spread_direction: float = 0.0       # degrés, direction de propagation
    heat_per_unit_area: float = 0.0   # kJ/m², chaleur par unité de surface
    residence_time: float = 0.0         # min, temps de résidence
    fuel_consumption: float = 0.0       # kg/m², consommation de combustible
    phi_w: float = 0.0                  # coefficient de vent
    phi_s: float = 0.0                  # coefficient de pente
    phi_eff: float = 0.0                # coefficient effectif combiné
    beta: float = 0.0                   # packing ratio
    beta_opt: float = 0.0               # packing ratio optimal
    gamma: float = 0.0                  # vitesse de réaction
    xi: float = 0.0                     # coefficient de propagation
    eta_M: float = 0.0                  # amortissement humidité
    eta_S: float = 0.0                  # amortissement minéral
    I_R: float = 0.0                    # BTU/ft²/min, intensité de réaction (impérial)
    tau: float = 0.0                    # min, temps de résidence


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


class RothermelEngine:
    """Moteur Rothermel v3 corrigé."""

    def __init__(self):
        self.uc = UnitConverter()
        self.const = RothermelConstants()

    # =====================================================================
    # 1. CHARGES ET SAV COMBINÉS
    # =====================================================================
    def _compute_total_loading(self, fuel: FuelModel) -> float:
        return fuel.w_1h + fuel.w_10h + fuel.w_100h + fuel.w_live_herb + fuel.w_live_woody

    def _compute_dead_loading(self, fuel: FuelModel) -> float:
        return fuel.w_1h + fuel.w_10h + fuel.w_100h

    def _compute_live_loading(self, fuel: FuelModel) -> float:
        return fuel.w_live_herb + fuel.w_live_woody

    def _compute_weighted_sav(self, fuel: FuelModel, w_total: float) -> float:
        """SAV caractéristique pondéré [m²/m³].

        Compatible fuel_models.py : combine sigma_live_herb + sigma_live_woody.
        """
        if w_total <= 0:
            return 0.0
        w_dead = self._compute_dead_loading(fuel)
        w_live = self._compute_live_loading(fuel)

        sav_dead = 0.0
        if w_dead > 0:
            sav_dead = (fuel.w_1h * fuel.sigma_1h + 
                       fuel.w_10h * fuel.sigma_10h + 
                       fuel.w_100h * fuel.sigma_100h) / w_dead

        # Moyenne pondérée des SAV vivantes (compatible fuel_models.py)
        sav_live = 0.0
        if w_live > 0:
            sav_live = (fuel.w_live_herb * fuel.sigma_live_herb + 
                       fuel.w_live_woody * fuel.sigma_live_woody) / w_live

        sigma = (w_dead * sav_dead + w_live * sav_live) / w_total
        return sigma

    def _compute_weighted_heat_content(self, fuel: FuelModel, w_total: float) -> float:
        """Chaleur de combustion moyenne [kJ/kg]."""
        if w_total <= 0:
            return 0.0
        w_dead = self._compute_dead_loading(fuel)
        w_live = self._compute_live_loading(fuel)
        h_avg = (w_dead * fuel.h_dead + w_live * fuel.h_live) / w_total
        return h_avg

    # =====================================================================
    # 2. PACKING RATIO
    # =====================================================================
    def _compute_packing_ratio(self, fuel: FuelModel, w_total: float) -> float:
        """Packing ratio beta [adimensionnel]."""
        if fuel.delta <= 0 or w_total <= 0:
            return 0.0
        rho_b = w_total / fuel.delta  # kg/m³
        rho_p = 512.5  # kg/m³, densité particulaire standard
        return rho_b / rho_p

    def _compute_optimum_packing_ratio(self, sigma: float) -> float:
        """Packing ratio optimal [Albini 1976]. sigma en m²/m³."""
        if sigma <= 0:
            return 0.0
        # Conversion sigma en ft⁻¹ pour la formule d'Albini
        sigma_ft = sigma * 0.3048
        beta_opt = 3.348 * (sigma_ft ** (-0.8189))
        return beta_opt

    # =====================================================================
    # 3. VITESSE DE RÉACTION (CORRECTION #1 : gamma complet)
    # =====================================================================
    def _compute_reaction_velocity(self, sigma: float, beta: float, beta_opt: float) -> float:
        """Vitesse de réaction complète [min⁻¹] avec correction Albini (1976)."""
        if sigma <= 0 or beta_opt <= 0:
            return 0.0

        # Conversion sigma en ft⁻¹ pour les formules d'Albini
        sigma_ft = sigma * 0.3048

        sigma_15 = sigma_ft ** 1.5
        gamma_max = sigma_15 / (495.0 + 0.0594 * sigma_15)

        # Coefficient A (Albini 1976)
        A = 133.0 * (sigma_ft ** (-0.7913))

        # Facteur de correction du packing ratio
        ratio = beta / beta_opt
        ratio = np.clip(ratio, 0.01, 10.0)

        correction = (ratio ** A) * np.exp(A * (1.0 - ratio))
        gamma = gamma_max * correction

        return gamma

    # =====================================================================
    # 4. AMORTISSEMENT HUMIDITÉ (CORRECTION #5 : Mx,live affiné)
    # =====================================================================
    def _compute_moisture_damping(self, Mf: float, Mx: float) -> float:
        """Amortissement dû à l'humidité [adimensionnel]."""
        if Mx <= 0:
            return 0.0
        r = Mf / Mx
        r = np.clip(r, 0.0, 1.0)
        eta = 1.0 - 2.59 * r + 5.11 * r**2 - 3.52 * r**3
        return max(0.0, eta)

    def _compute_live_moisture_of_extinction(self, mx_dead: float,
                                              w_dead: float,
                                              w_live: float) -> float:
        """Humidité d'extinction pour le combustible vivant [Albini 1976]."""
        w_total = w_dead + w_live
        if w_live <= 0 or w_total <= 0:
            return mx_dead

        f_live = w_live / w_total
        Mx_live = 2.9 * (f_live ** 1.5) * (mx_dead ** (-0.5))

        # Contraintes Albini (1976)
        Mx_live = max(Mx_live, mx_dead)  # Minimum: Mx_dead
        Mx_live = min(Mx_live, 3.0)      # Maximum: 300% (plafond réaliste)

        return Mx_live

    def _compute_moisture_damping_total(self, fuel: FuelModel, 
                                        moisture: MoistureInputs,
                                        w_total: float) -> float:
        """Amortissement humidité total pondéré."""
        w_dead = self._compute_dead_loading(fuel)
        w_live = self._compute_live_loading(fuel)
        w_1h = fuel.w_1h
        w_10h = fuel.w_10h
        w_100h = fuel.w_100h
        w_live_herb = fuel.w_live_herb
        w_live_woody = fuel.w_live_woody

        mx = fuel.mx / 100.0  # Conversion % → fraction

        # Morts
        eta_M_1h = self._compute_moisture_damping(moisture.m_1h, mx)
        eta_M_10h = self._compute_moisture_damping(moisture.m_10h, mx)
        eta_M_100h = self._compute_moisture_damping(moisture.m_100h, mx)

        # Vivants (Mx,live affiné)
        eta_M_live = 1.0
        if w_live > 0:
            mx_live = self._compute_live_moisture_of_extinction(mx, w_dead, w_live)
            m_live_avg = (w_live_herb * moisture.m_live_herb + 
                         w_live_woody * moisture.m_live_woody) / w_live
            eta_M_live = self._compute_moisture_damping(m_live_avg, mx_live)

        # Moyenne pondérée
        if w_total > 0:
            eta_M = (w_1h * eta_M_1h + w_10h * eta_M_10h + w_100h * eta_M_100h +
                    w_live_herb * eta_M_live + w_live_woody * eta_M_live) / w_total
        else:
            eta_M = 1.0

        return eta_M

    # =====================================================================
    # 5. AMORTISSEMENT MINÉRAL
    # =====================================================================
    def _compute_mineral_damping(self, fuel: FuelModel) -> float:
        """Amortissement dû au contenu minéral.

        Compatible fuel_models.py : utilise getattr avec valeurs par défaut
        si st/se ne sont pas définis dans le FuelModel.
        """
        st = getattr(fuel, 'st', 0.0555)
        se = getattr(fuel, 'se', 0.01)

        if st <= 0 or se <= 0:
            return 1.0
        eta_S = 0.174 * (se ** (-0.19))
        return max(0.0, eta_S)

    # =====================================================================
    # 6. COEFFICIENTS VENT ET PENTE
    # =====================================================================
    def _compute_wind_coefficient(self, wind_speed: float, beta: float, 
                                   beta_opt: float, sigma: float) -> float:
        """Coefficient de vent phi_w [Rothermel 1972, Albini 1976].

        ATTENTION : La vitesse du vent DOIT être en PIEDS PAR MINUTE (ft/min) 
        pour que les constantes C, B, E soient correctes.
        """
        if wind_speed <= 0 or beta <= 0 or beta_opt <= 0 or sigma <= 0:
            return 0.0

        # Conversion m/s → ft/min (unité originale de Rothermel)
        wind_ft_min = wind_speed * self.uc.MS_TO_FT_MIN

        # Conversion sigma en ft⁻¹ pour les constantes
        sigma_ft = sigma * 0.3048

        # Constantes empiriques de Rothermel (1972)
        C = self.const.WIND_C_COEF * np.exp(self.const.WIND_C_EXP * (sigma_ft ** 0.55))
        B = self.const.WIND_B_COEF * (sigma_ft ** self.const.WIND_B_EXP)
        E = self.const.WIND_E_COEF * np.exp(self.const.WIND_E_EXP * sigma_ft)

        # Garde-fou beta/beta_opt
        ratio = np.clip(beta / beta_opt, 0.01, 10.0)

        if ratio <= 0.01 or ratio >= 10.0:
            logging.warning(
                f"beta/beta_opt = {ratio:.2f} hors plage réaliste [0.01, 10.0]. "
                f"Vérifiez les paramètres du fuel model."
            )

        # Calcul de phi_w brut
        phi_w = C * (wind_ft_min ** B) * (ratio ** (-E))

        # =====================================================================
        # 1. AMORTISSEMENT LOGARITHMIQUE (Pour les hauts SAV > 2500)
        # =====================================================================
        if sigma_ft > 2500:
            damp = 1.0 - 0.15 * np.log10(sigma_ft / 2500.0)
            damp = max(0.60, damp)  # Minimum 60% de l'effet
            phi_w = phi_w * damp

        # =====================================================================
        # 2. SATURATION DOUCE (Smooth Cap)
        # =====================================================================
        PHI_W_SOFT_CAP = 25.0
        phi_w = PHI_W_SOFT_CAP * (1.0 - np.exp(-phi_w / PHI_W_SOFT_CAP))

        return max(0.0, phi_w)

    def _compute_slope_coefficient(self, slope_pct: float, beta: float) -> float:
        """Coefficient de pente phi_s [Rothermel 1972].

        CORRECTION : Utilise tan(θ)² où tan(θ) ≈ slope_pct/100.
        PAS de division par beta_opt (absente de la formule originale).
        """
        if slope_pct <= 0 or beta <= 0:
            return 0.0

        # La formule utilise tan(θ)². Pour une pente en %, tan(θ) ≈ slope_pct / 100.0
        slope_fraction = slope_pct / 100.0

        # Formule standard Rothermel : 5.275 * β^(-0.3) * tan(θ)²
        phi_s = 5.275 * (beta ** (-0.3)) * (slope_fraction ** 2)

        return max(0.0, phi_s)

    # =====================================================================
    # 7. COMPOSITION VECTORIELLE VENT/PENTE (CORRECTION #2)
    # =====================================================================
    def _compute_angle_correction_vectorial(self, phi_w: float, phi_s: float,
                                             angle_wind_slope: float) -> float:
        """Composition vectorielle vent/pente [Catchpole 1982].

        phi_eff = |vecteur_vent + vecteur_pente|

        angle_wind_slope : angle relatif entre vent et pente (degrés)
                           0° = vent dans la pente (même direction)
                           180° = vent contre la pente
        """
        if phi_w <= 0 and phi_s <= 0:
            return 0.0

        theta = np.radians(angle_wind_slope)

        # Vecteur vent (direction de propagation = 0°)
        wx = phi_w
        wy = 0.0

        # Vecteur pente (angle relatif)
        sx = phi_s * np.cos(theta)
        sy = phi_s * np.sin(theta)

        # Vecteur résultant
        Rx = wx + sx
        Ry = wy + sy

        phi_eff = np.sqrt(Rx**2 + Ry**2)
        return phi_eff

    # =====================================================================
    # 8. TEMPS DE RÉSIDENCE ET COEFFICIENT XI
    # =====================================================================
    def _compute_residence_time(self, sigma: float) -> float:
        """Temps de résidence [min]. Anderson 1969."""
        if sigma <= 0:
            return 0.0
        sigma_ft = sigma * 0.3048
        tau = 384.0 / sigma_ft
        return max(0.0, tau)

    def _compute_propagation_coefficient(self, sigma: float, beta: float) -> float:
        """Coefficient de propagation xi."""
        if sigma <= 0 or beta <= 0:
            return 0.0
        sigma_ft = sigma * 0.3048
        xi = np.exp((0.792 + 0.681 * (sigma_ft ** 0.5)) * (beta + 0.1)) /              (192.0 + 0.2595 * sigma_ft)
        return min(xi, 1.0)

    # =====================================================================
    # 9. LONGUEUR DE FLAMME
    # =====================================================================
    def _compute_flame_length(self, fireline_intensity: float) -> float:
        """Longueur de flamme [m]. Byram 1959."""
        if fireline_intensity <= 0:
            return 0.0
        # I_b en kW/m → L en m
        L = 0.0775 * (fireline_intensity ** 0.46)
        return max(0.0, L)

    # =====================================================================
    # 10. MÉTHODE PRINCIPALE COMPUTE (CORRECTIONS #6, #7, #8)
    # =====================================================================
    def compute(self, fuel: FuelModel, moisture: MoistureInputs, 
                conditions: EnvironmentalConditions) -> RothermelOutput:
        """Calcul principal du moteur Rothermel v3."""
        output = RothermelOutput()

        # 1. Chargements
        w_total = self._compute_total_loading(fuel)
        w_dead = self._compute_dead_loading(fuel)
        w_live = self._compute_live_loading(fuel)

        if w_total <= 0:
            logging.warning("Charge totale nulle. ROS = 0.")
            return output

        # 2. SAV et chaleur moyens
        sigma = self._compute_weighted_sav(fuel, w_total)
        h_avg = self._compute_weighted_heat_content(fuel, w_total)

        if sigma <= 0:
            logging.warning("SAV nulle. ROS = 0.")
            return output

        # 3. Packing ratio (CORRECTION #6 : calculé AVANT gamma)
        beta = self._compute_packing_ratio(fuel, w_total)
        output.beta = beta

        if beta <= 0:
            logging.warning("Packing ratio nul. ROS = 0.")
            return output

        # 4. Packing ratio optimal (CORRECTION #6 : calculé AVANT gamma)
        beta_opt = self._compute_optimum_packing_ratio(sigma)
        output.beta_opt = beta_opt

        if beta_opt <= 0:
            logging.warning("Packing ratio optimal nul. ROS = 0.")
            return output

        # 5. Vitesse de réaction (CORRECTION #1 : gamma complet)
        gamma = self._compute_reaction_velocity(sigma, beta, beta_opt)
        output.gamma = gamma

        # 6. Amortissements
        eta_M = self._compute_moisture_damping_total(fuel, moisture, w_total)
        eta_S = self._compute_mineral_damping(fuel)  # Compatible fuel_models.py
        output.eta_M = eta_M
        output.eta_S = eta_S

        # 7. Intensité de réaction (CORRECTION #8 : conversion 0.189)
        w0_total_lb_ft2 = w_total * self.uc.KG_M2_TO_LB_FT2
        h_btu_lb = h_avg * self.uc.KJ_KG_TO_BTU_LB

        I_R_imp = gamma * w0_total_lb_ft2 * h_btu_lb * eta_M * eta_S
        output.I_R = I_R_imp

        # Conversion correcte : 1 BTU/ft²/min = 0.189 kW/m²
        output.reaction_intensity = I_R_imp * 0.189  # kW/m²

        # 8. Temps de résidence
        tau = self._compute_residence_time(sigma)
        output.tau = tau
        output.residence_time = tau

        # 9. Coefficient de propagation xi
        xi = self._compute_propagation_coefficient(sigma, beta)
        output.xi = xi

        # 10. Coefficients vent et pente
        phi_w = self._compute_wind_coefficient(conditions.wind_speed, beta, beta_opt, sigma)
        phi_s = self._compute_slope_coefficient(conditions.slope_pct, beta)
        output.phi_w = phi_w
        output.phi_s = phi_s

        # 11. Composition vectorielle (CORRECTION #2)
        phi_eff = self._compute_angle_correction_vectorial(phi_w, phi_s, conditions.angle_wind_slope)
        output.phi_eff = phi_eff

        # 12. Densité bulk et epsilon
        rho_b = w_total / fuel.delta if fuel.delta > 0 else 0.0
        rho_b_lb_ft3 = rho_b * self.uc.KG_M3_TO_LB_FT3
        sigma_ft = sigma * 0.3048
        epsilon = np.exp(-138.0 / max(sigma_ft, 1.0))

        # 13. Chaleur de pré-combustion (en Btu/lb directement)
        avg_Mf = ((fuel.w_1h * moisture.m_1h + fuel.w_10h * moisture.m_10h + 
                  fuel.w_100h * moisture.m_100h + 
                  fuel.w_live_herb * moisture.m_live_herb + 
                  fuel.w_live_woody * moisture.m_live_woody) / w_total)
        Q_ig_btu_lb = 250.0 + 1116.0 * avg_Mf  # Déjà en Btu/lb

        # 14. ROS de base et effectif
        if rho_b_lb_ft3 > 0 and epsilon > 0 and Q_ig_btu_lb > 0:
            denominator = rho_b_lb_ft3 * epsilon * Q_ig_btu_lb
            R0 = (I_R_imp * xi) / denominator
            ros_ft_min = R0 * (1.0 + phi_eff)
            output.ros = ros_ft_min / self.uc.M_MIN_TO_FT_MIN  # m/min
        else:
            output.ros = 0.0

        # 15. Intensité ligne de feu (CORRECTION #3 : Byram I_B = I_R × τ × ROS)
        if tau > 0 and output.ros > 0:
            # I_B (kW/m) = Intensité de réaction (kW/m²) × Temps de résidence (min) × ROS (m/min)
            # Unités : (kJ/s/m²) × min × (m/min) = kJ/s/m = kW/m
            output.fireline_intensity = output.reaction_intensity * tau * output.ros
        else:
            output.fireline_intensity = 0.0

        # 16. Longueur de flamme
        output.flame_length = self._compute_flame_length(output.fireline_intensity)

        # 17. Chaleur par unité de surface (HPUA)
        # HPUA (kJ/m²) = Intensité de réaction (kW/m²) × Temps de résidence (min) × 60 (s/min)
        if tau > 0:
            output.heat_per_unit_area = output.reaction_intensity * tau * 60.0
        else:
            output.heat_per_unit_area = 0.0

        # 18. Consommation de combustible
        output.fuel_consumption = w_total * eta_M * eta_S

        # 19. Direction de propagation (simplifiée)
        if phi_eff > 0:
            theta = np.radians(conditions.angle_wind_slope)
            # Direction dominante = vent si phi_w > phi_s, sinon pente
            if phi_w >= phi_s:
                output.spread_direction = 0.0  # Direction du vent
            else:
                output.spread_direction = conditions.angle_wind_slope  # Direction de la pente

        return output


# =====================================================================
# BURNTRACK INTEGRATION
# =====================================================================
class BurnTrackRothermel:
    """Interface BurnTrack pour le moteur Rothermel v3."""

    def __init__(self):
        self.engine = RothermelEngine()

    def predict(self, fuel_model: FuelModel, 
                moisture: MoistureInputs,
                wind_speed: float,
                slope_pct: float,
                angle_wind_slope: float = 0.0) -> Dict:
        """Prédiction simplifiée pour BurnTrack."""
        conditions = EnvironmentalConditions(
            wind_speed=wind_speed,
            slope_pct=slope_pct,
            angle_wind_slope=angle_wind_slope
        )

        output = self.engine.compute(fuel_model, moisture, conditions)

        return {
            'ros': output.ros,
            'fireline_intensity': output.fireline_intensity,
            'flame_length': output.flame_length,
            'direction': output.spread_direction,
            'heat_per_unit_area': output.heat_per_unit_area,
            'fuel_consumption': output.fuel_consumption,
            'phi_w': output.phi_w,
            'phi_s': output.phi_s,
            'phi_eff': output.phi_eff,
            'beta': output.beta,
            'beta_opt': output.beta_opt,
            'gamma': output.gamma,
            'eta_M': output.eta_M,
            'eta_S': output.eta_S
        }