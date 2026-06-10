"""
fuel_models.py
================
Fuel Models pour le pipeline BurnTrack - Adaptés au Maroc et à l'Afrique.

Ce module définit les paramètres de combustible pour le modèle de Rothermel,
calibrés pour les écosystèmes africains.

Unités SI :
- Charges de combustible : kg/m²
- Profondeur du lit (delta) : m
- SAV (sigma) : m²/m³ = 1/m
- Chaleur de combustion (h) : kJ/kg
- Humidité d'extinction (mx) : % (fraction × 100)

Sources :
- BehavePlus / firelab (C++) - 53 modèles originaux + 40 étendus
- Scott & Burgan (2005) - Standard Fire Behavior Fuel Models
- Données terrain Afrique : FAO, van Wilgen et al., Savadogo et al.
- Données Sahel : Sow et al. (2013)
- Données savanes sud-africaines : Trollope, Scholes

Constantes Rothermel (inchangées) :
- Teneur minérale totale : 5.55%
- Teneur minérale effective (silice-free) : 1.00%
- Densité particulaire : 512 kg/m³ (32 lb/ft³)
- SAV 10h : 357 1/m (109 1/ft)
- SAV 100h : 98 1/m (30 1/ft)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FuelModel:
    """
    Paramètres de combustible pour le modèle de Rothermel.

    Tous les paramètres sont en unités SI.
    """
    # Identifiants
    code: str                    # Code du fuel model (ex: "GR1", "AF_STEPPE")
    name: str                    # Nom descriptif

    # Charges de combustible (kg/m²)
    w_1h: float                 # Combustible mort fin (< 6mm)
    w_10h: float                # Combustible mort moyen (6-25mm)
    w_100h: float               # Combustible mort gros (25-75mm)
    w_live_herb: float          # Herbes vivantes
    w_live_woody: float         # Ligneux vivant (feuilles, branches fines)

    # Propriétés physiques
    delta: float                # Profondeur du lit de combustible (m)
    sigma_1h: float             # SAV combustible fin (1/m)
    sigma_live_herb: float      # SAV herbes vivantes (1/m)
    sigma_live_woody: float     # SAV ligneux vivant (1/m)

    # Propriétés thermiques
    h_dead: float               # Chaleur combustion mort (kJ/kg)
    h_live: float               # Chaleur combustion vivant (kJ/kg)
    mx: float                   # Humidité d'extinction mort (%)

    # Métadonnées
    is_dynamic: bool = False     # Modèle dynamique (herbes vivantes → mortes)
    region: str = "general"      # Région d'application
    description: str = ""       # Description détaillée

    # --- Propriétés dérivées ---

    @property
    def w_total(self) -> float:
        """Charge totale de combustible (kg/m²)."""
        return self.w_1h + self.w_10h + self.w_100h + self.w_live_herb + self.w_live_woody

    @property
    def w_dead(self) -> float:
        """Charge totale de combustible mort (kg/m²)."""
        return self.w_1h + self.w_10h + self.w_100h

    @property
    def w_live(self) -> float:
        """Charge totale de combustible vivant (kg/m²)."""
        return self.w_live_herb + self.w_live_woody

    @property
    def has_live_herb(self) -> bool:
        """True si le modèle contient des herbes vivantes."""
        return self.w_live_herb > 0

    @property
    def is_grass_dominant(self) -> bool:
        """True si le combustible est dominé par les herbes."""
        return (self.w_live_herb + self.w_1h) > (self.w_live_woody + self.w_10h + self.w_100h)

    def __repr__(self) -> str:
        return f"FuelModel({self.code}: {self.name}, w_total={self.w_total:.3f} kg/m², delta={self.delta}m)"


# =============================================================================
# MODÈLES STANDARD BEHAVE (convertis en SI depuis lb/ft², ft, BTU/lb)
# =============================================================================
# Conversion : 1 lb/ft² = 4.882 kg/m², 1 ft = 0.3048 m, 1 BTU/lb = 2.326 kJ/kg
# =============================================================================

BEHAVE_STANDARD: Dict[str, FuelModel] = {
    # --- HERBES (GR) ---
    "GR1": FuelModel(
        code="GR1", name="Short, sparse, dry climate grass",
        w_1h=0.005, w_10h=0, w_100h=0, w_live_herb=0.015, w_live_woody=0,
        delta=0.12, sigma_1h=7216, sigma_live_herb=6562, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes courtes, clairsemées, climat sec. ROS modérée, flammes courtes."
    ),
    "GR2": FuelModel(
        code="GR2", name="Low load, dry climate grass",
        w_1h=0.005, w_10h=0, w_100h=0, w_live_herb=0.049, w_live_woody=0,
        delta=0.30, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à faible charge, climat sec."
    ),
    "GR3": FuelModel(
        code="GR3", name="Low load, very coarse, humid climate grass",
        w_1h=0.005, w_10h=0.020, w_100h=0, w_live_herb=0.073, w_live_woody=0,
        delta=0.61, sigma_1h=4921, sigma_live_herb=4265, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Herbes grossières, climat humide."
    ),
    "GR4": FuelModel(
        code="GR4", name="Moderate load, dry climate grass",
        w_1h=0.012, w_10h=0, w_100h=0, w_live_herb=0.093, w_live_woody=0,
        delta=0.61, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à charge modérée, climat sec."
    ),
    "GR5": FuelModel(
        code="GR5", name="Low load, humid climate grass",
        w_1h=0.020, w_10h=0, w_100h=0, w_live_herb=0.122, w_live_woody=0,
        delta=0.46, sigma_1h=5906, sigma_live_herb=5249, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à faible charge, climat humide."
    ),
    "GR6": FuelModel(
        code="GR6", name="Moderate load, humid climate grass",
        w_1h=0.005, w_10h=0, w_100h=0, w_live_herb=0.166, w_live_woody=0,
        delta=0.46, sigma_1h=7216, sigma_live_herb=6562, sigma_live_woody=4921,
        h_dead=20934, h_live=20934, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à charge modérée, climat humide."
    ),
    "GR7": FuelModel(
        code="GR7", name="High load, dry climate grass",
        w_1h=0.049, w_10h=0, w_100h=0, w_live_herb=0.264, w_live_woody=0,
        delta=0.91, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à haute charge, climat sec. ROS élevée possible."
    ),
    "GR8": FuelModel(
        code="GR8", name="High load, very coarse, humid climate grass",
        w_1h=0.024, w_10h=0.049, w_100h=0, w_live_herb=0.356, w_live_woody=0,
        delta=1.22, sigma_1h=4921, sigma_live_herb=4265, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Herbes très grossières, haute charge, humide."
    ),
    "GR9": FuelModel(
        code="GR9", name="Very high load, humid climate grass",
        w_1h=0.049, w_10h=0.049, w_100h=0, w_live_herb=0.439, w_live_woody=0,
        delta=1.52, sigma_1h=5906, sigma_live_herb=5249, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à très haute charge, humide. ROS et flammes extrêmes."
    ),

    # --- HERBES-ARBRISSEAU (GS) ---
    "GS1": FuelModel(
        code="GS1", name="Low load, dry climate grass-shrub",
        w_1h=0.010, w_10h=0, w_100h=0, w_live_herb=0.024, w_live_woody=0.032,
        delta=0.27, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=5906,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, faible charge, sec."
    ),
    "GS2": FuelModel(
        code="GS2", name="Moderate load, dry climate grass-shrub",
        w_1h=0.024, w_10h=0.024, w_100h=0, w_live_herb=0.029, w_live_woody=0.049,
        delta=0.46, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=5906,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, charge modérée, sec."
    ),
    "GS3": FuelModel(
        code="GS3", name="Moderate load, humid climate grass-shrub",
        w_1h=0.015, w_10h=0.012, w_100h=0, w_live_herb=0.071, w_live_woody=0.061,
        delta=0.55, sigma_1h=5906, sigma_live_herb=5249, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, charge modérée, humide."
    ),
    "GS4": FuelModel(
        code="GS4", name="High load, humid climate grass-shrub",
        w_1h=0.093, w_10h=0.015, w_100h=0.005, w_live_herb=0.166, w_live_woody=0.347,
        delta=0.64, sigma_1h=5906, sigma_live_herb=5249, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, haute charge, humide."
    ),

    # --- ARBRISSEAU (SH) ---
    "SH1": FuelModel(
        code="SH1", name="Low load, dry climate shrub",
        w_1h=0.012, w_10h=0.012, w_100h=0, w_live_herb=0.007, w_live_woody=0.063,
        delta=0.30, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à faible charge, sec."
    ),
    "SH2": FuelModel(
        code="SH2", name="Moderate load, dry climate shrub",
        w_1h=0.066, w_10h=0.117, w_100h=0.037, w_live_herb=0, w_live_woody=0.188,
        delta=0.30, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à charge modérée, sec."
    ),
    "SH3": FuelModel(
        code="SH3", name="Moderate load, humid climate shrub",
        w_1h=0.022, w_10h=0.147, w_100h=0, w_live_herb=0, w_live_woody=0.303,
        delta=0.73, sigma_1h=5249, sigma_live_herb=5906, sigma_live_woody=4593,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à charge modérée, humide."
    ),
    "SH4": FuelModel(
        code="SH4", name="Low load, humid climate timber-shrub",
        w_1h=0.042, w_10h=0.056, w_100h=0.010, w_live_herb=0, w_live_woody=0.125,
        delta=0.91, sigma_1h=6562, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Arbustes + forêt claire, faible charge, humide."
    ),
    "SH5": FuelModel(
        code="SH5", name="High load, dry climate shrub",
        w_1h=0.176, w_10h=0.103, w_100h=0, w_live_herb=0, w_live_woody=0.142,
        delta=1.83, sigma_1h=2461, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à haute charge, sec. ROS et flammes élevées."
    ),
    "SH6": FuelModel(
        code="SH6", name="Low load, humid climate shrub",
        w_1h=0.142, w_10h=0.071, w_100h=0, w_live_herb=0, w_live_woody=0.068,
        delta=0.61, sigma_1h=2461, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Arbustes à faible charge, humide."
    ),
    "SH7": FuelModel(
        code="SH7", name="Very high load, dry climate shrub",
        w_1h=0.171, w_10h=0.259, w_100h=0.107, w_live_herb=0, w_live_woody=0.166,
        delta=1.83, sigma_1h=2461, sigma_live_herb=5906, sigma_live_woody=5249,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à très haute charge, sec. Comportement extrême."
    ),
    "SH8": FuelModel(
        code="SH8", name="High load, humid climate shrub",
        w_1h=0.100, w_10h=0.166, w_100h=0.042, w_live_herb=0, w_live_woody=0.212,
        delta=0.91, sigma_1h=2461, sigma_live_herb=5906, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à haute charge, humide."
    ),
    "SH9": FuelModel(
        code="SH9", name="Very high load, humid climate shrub",
        w_1h=0.220, w_10h=0.120, w_100h=0, w_live_herb=0.071, w_live_woody=0.342,
        delta=1.34, sigma_1h=2461, sigma_live_herb=5906, sigma_live_woody=4921,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à très haute charge, humide. Comportement extrême."
    ),
}


# =============================================================================
# MODÈLES AFRIQUE DU NORD / MAROC (calibrés pour écosystèmes méditerranéens)
# =============================================================================

AFRICA_NORTH: Dict[str, FuelModel] = {
    # --- STEPPE MAROCAINE ---
    "AF_STEPPE": FuelModel(
        code="AF_STEPPE", name="Steppe marocaine (Artemisia, Stipa)",
        w_1h=0.08, w_10h=0.02, w_100h=0, w_live_herb=0.04, w_live_woody=0.02,
        delta=0.25, sigma_1h=5500, sigma_live_herb=5500, sigma_live_woody=3500,
        h_dead=18600, h_live=19000, mx=12,
        is_dynamic=True, region="north_africa",
        description="Steppe à Artemisia herba-alba, Stipa tenacissima. "
                    "Climat aride, hiver frais, été très chaud. "
                    "Charge faible, combustion rapide."
    ),
    "AF_STEPPE_DENSE": FuelModel(
        code="AF_STEPPE_DENSE", name="Steppe dense (haut Atlas)",
        w_1h=0.15, w_10h=0.05, w_100h=0.01, w_live_herb=0.08, w_live_woody=0.05,
        delta=0.40, sigma_1h=5000, sigma_live_herb=5000, sigma_live_woody=3000,
        h_dead=18600, h_live=19000, mx=15,
        is_dynamic=True, region="north_africa",
        description="Steppe dense de montagne. Plus de biomasse que la steppe basse."
    ),

    # --- ARGANIER ---
    "AF_ARGAN": FuelModel(
        code="AF_ARGAN", name="Arganeraie (Argania spinosa)",
        w_1h=0.10, w_10h=0.15, w_100h=0.05, w_live_herb=0.02, w_live_woody=0.30,
        delta=0.35, sigma_1h=4500, sigma_live_herb=4500, sigma_live_woody=2800,
        h_dead=18600, h_live=19500, mx=18,
        is_dynamic=True, region="north_africa",
        description="Forêt d'arganier (Argania spinosa). "
                    "Arbres tordus, feuillage persistant, épineux. "
                    "Lit de feuilles mortes + herbes rares."
    ),

    # --- CHÊNE-LIÈGE ---
    "AF_CHENE_LIEGE": FuelModel(
        code="AF_CHENE_LIEGE", name="Chênaie-liège (Quercus suber)",
        w_1h=0.25, w_10h=0.49, w_100h=0.10, w_live_herb=0.05, w_live_woody=0.59,
        delta=0.30, sigma_1h=5500, sigma_live_herb=5500, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=25,
        is_dynamic=True, region="north_africa",
        description="Forêt de chêne-liège (Quercus suber). "
                    "Mamora, Rif. Feuillage persistant, écorce épaisse. "
                    "Lit dense de feuilles et branches."
    ),

    # --- CÈDRE DE L'ATLAS ---
    "AF_CEDRE": FuelModel(
        code="AF_CEDRE", name="Cèdraie de l'Atlas (Cedrus atlantica)",
        w_1h=0.15, w_10h=0.39, w_100h=0.15, w_live_herb=0.02, w_live_woody=0.49,
        delta=0.25, sigma_1h=6000, sigma_live_herb=6000, sigma_live_woody=4000,
        h_dead=20000, h_live=20500, mx=30,
        is_dynamic=True, region="north_africa",
        description="Forêt de cèdre de l'Atlas (Cedrus atlantica). "
                    "Moyen Atlas, Haut Atlas. Aiguilles, branches, écorce. "
                    "Feux de surface modérés, risque cime."
    ),

    # --- MAQUIS MÉDITERRANÉEN ---
    "AF_MAQUIS": FuelModel(
        code="AF_MAQUIS", name="Maquis méditerranéen (Rif, Moyen Atlas)",
        w_1h=0.29, w_10h=0.24, w_100h=0.05, w_live_herb=0.10, w_live_woody=0.39,
        delta=0.50, sigma_1h=4500, sigma_live_herb=4500, sigma_live_woody=3000,
        h_dead=19000, h_live=19500, mx=20,
        is_dynamic=True, region="north_africa",
        description="Maquis méditerranéen (ciste, lentisque, arbousier). "
                    "Rif, Prérif. Dense, persistant, résineux. "
                    "Feux intenses, reprise rapide."
    ),

    # --- PLAINE CÉRÉALIÈRE ---
    "AF_CEREALES": FuelModel(
        code="AF_CEREALES", name="Plaine céréalière (résidus de récolte)",
        w_1h=0.73, w_10h=0, w_100h=0, w_live_herb=0, w_live_woody=0,
        delta=0.10, sigma_1h=8000, sigma_live_herb=8000, sigma_live_woody=4921,
        h_dead=17000, h_live=17000, mx=12,
        is_dynamic=False, region="north_africa",
        description="Résidus de céréales (blé, orge) après récolte. "
                    "Haouz, Tadla, Gharb. Lit très bas, très sec. "
                    "Feux rapides de surface."
    ),

    # --- PALMIER DATTIER ---
    "AF_PALMIER": FuelModel(
        code="AF_PALMIER", name="Palmeraie (Phoenix dactylifera)",
        w_1h=0.20, w_10h=0.10, w_100h=0.05, w_live_herb=0.02, w_live_woody=0.15,
        delta=0.30, sigma_1h=4000, sigma_live_herb=4000, sigma_live_woody=2500,
        h_dead=18000, h_live=18500, mx=15,
        is_dynamic=True, region="north_africa",
        description="Palmeraie oasienne (Phoenix dactylifera). "
                    "Feuilles mortes, dattes, herbes rares. "
                    "Feux de surface modérés."
    ),

    # --- TAMARIX (Oueds, ripisylves) ---
    "AF_TAMARIX": FuelModel(
        code="AF_TAMARIX", name="Tamarix (ripisylve des oueds)",
        w_1h=0.10, w_10h=0.15, w_100h=0.05, w_live_herb=0.03, w_live_woody=0.25,
        delta=0.40, sigma_1h=4500, sigma_live_herb=4500, sigma_live_woody=2800,
        h_dead=18600, h_live=19500, mx=20,
        is_dynamic=True, region="north_africa",
        description="Ripisylve à Tamarix (oueds sahariens, pré-sahariens). "
                    "Bois dense, salin, persistant."
    ),

    # --- JUJUBIER (Ziziphus lotus) ---
    "AF_JUJUBIER": FuelModel(
        code="AF_JUJUBIER", name="Jujubier (Ziziphus lotus)",
        w_1h=0.08, w_10h=0.05, w_100h=0.02, w_live_herb=0.02, w_live_woody=0.15,
        delta=0.35, sigma_1h=4000, sigma_live_herb=4000, sigma_live_woody=2500,
        h_dead=18600, h_live=19000, mx=18,
        is_dynamic=True, region="north_africa",
        description="Arbustes à Ziziphus lotus (steppe saharienne). "
                    "Épineux, persistant, très résistant à la sécheresse."
    ),
}


# =============================================================================
# MODÈLES AFRIQUE SUBSAHARIENNE / SAVANES
# =============================================================================
# Sources : FAO, van Wilgen et al., Savadogo et al., Trollope, Scholes
# =============================================================================

AFRICA_SAVANNA: Dict[str, FuelModel] = {
    # --- SAHEL / SAVANE SÈCHE ---
    "AF_SAHEL_GRASS": FuelModel(
        code="AF_SAHEL_GRASS", name="Savane sahélienne herbeuse",
        w_1h=0.04, w_10h=0, w_100h=0, w_live_herb=0.06, w_live_woody=0,
        delta=0.20, sigma_1h=8000, sigma_live_herb=8000, sigma_live_woody=4921,
        h_dead=18600, h_live=18600, mx=12,
        is_dynamic=True, region="sahel",
        description="Savane sahélienne purement herbeuse. "
                    "Andropogon, Loudetia. Charge très faible, très sec. "
                    "Feux rapides de surface, saison brûlage courte."
    ),
    "AF_SAHEL_WOODED": FuelModel(
        code="AF_SAHEL_WOODED", name="Savane sahélienne arborée (Acacia)",
        w_1h=0.06, w_10h=0.02, w_100h=0, w_live_herb=0.08, w_live_woody=0.05,
        delta=0.30, sigma_1h=7000, sigma_live_herb=7000, sigma_live_woody=3500,
        h_dead=18600, h_live=19000, mx=15,
        is_dynamic=True, region="sahel",
        description="Savane sahélienne arborée (Acacia senegal, A. tortilis). "
                    "Herbes + arbustes épars. Feux de surface modérés."
    ),

    # --- SUDAN / SAVANE SOUDANIENNE ---
    "AF_SUDAN_GRASS": FuelModel(
        code="AF_SUDAN_GRASS", name="Savane soudanienne herbeuse",
        w_1h=0.10, w_10h=0.02, w_100h=0, w_live_herb=0.15, w_live_woody=0,
        delta=0.50, sigma_1h=7500, sigma_live_herb=7500, sigma_live_woody=4921,
        h_dead=18600, h_live=18600, mx=15,
        is_dynamic=True, region="sudan",
        description="Savane soudanienne herbeuse. "
                    "Hyparrhenia, Andropogon. Charge modérée, saison sèche marquée."
    ),
    "AF_SUDAN_WOODED": FuelModel(
        code="AF_SUDAN_WOODED", name="Savane soudanienne boisée",
        w_1h=0.12, w_10h=0.05, w_100h=0.02, w_live_herb=0.12, w_live_woody=0.15,
        delta=0.60, sigma_1h=6500, sigma_live_herb=6500, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=18,
        is_dynamic=True, region="sudan",
        description="Savane soudanienne boisée (Combretum, Terminalia). "
                    "Herbes + arbres feuillus. Feux de surface + brandons."
    ),

    # --- MIOMBO (Afrique australe/centrale) ---
    "AF_MIOMBO": FuelModel(
        code="AF_MIOMBO", name="Miombo (Brachystegia, Julbernardia)",
        w_1h=0.15, w_10h=0.10, w_100h=0.05, w_live_herb=0.08, w_live_woody=0.25,
        delta=0.50, sigma_1h=5500, sigma_live_herb=5500, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=25,
        is_dynamic=True, region="miombo",
        description="Forêt claire de miombo (Brachystegia, Julbernardia). "
                    "Zambie, Zimbabwe, Tanzanie. Feuilles composées, écorce résistante. "
                    "Feux de surface annuels, reprise végétale rapide."
    ),
    "AF_MIOMBO_DENSE": FuelModel(
        code="AF_MIOMBO_DENSE", name="Miombo dense",
        w_1h=0.20, w_10h=0.15, w_100h=0.08, w_live_herb=0.05, w_live_woody=0.40,
        delta=0.60, sigma_1h=5000, sigma_live_herb=5000, sigma_live_woody=2800,
        h_dead=20000, h_live=20500, mx=28,
        is_dynamic=True, region="miombo",
        description="Miombo dense, moins perturbé. Plus de biomasse ligneuse."
    ),

    # --- MOPANE (Afrique australe) ---
    "AF_MOPANE": FuelModel(
        code="AF_MOPANE", name="Mopane (Colophospermum mopane)",
        w_1h=0.12, w_10h=0.08, w_100h=0.03, w_live_herb=0.06, w_live_woody=0.20,
        delta=0.45, sigma_1h=5000, sigma_live_herb=5000, sigma_live_woody=2800,
        h_dead=19500, h_live=20000, mx=22,
        is_dynamic=True, region="mopane",
        description="Forêt claire de mopane (Colophospermum mopane). "
                    "Botswana, Namibie, Zambie, Mozambique. "
                    "Bois dur, feuillage en papillon, très résistant."
    ),

    # --- ACACIA SAVANNA (Afrique de l'Est) ---
    "AF_ACACIA_SAVANNA": FuelModel(
        code="AF_ACACIA_SAVANNA", name="Savane à Acacia (Afrique de l'Est)",
        w_1h=0.10, w_10h=0.05, w_100h=0.02, w_live_herb=0.12, w_live_woody=0.15,
        delta=0.50, sigma_1h=6000, sigma_live_herb=6000, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=20,
        is_dynamic=True, region="east_africa",
        description="Savane à Acacia (A. tortilis, A. nilotica, A. senegal). "
                    "Kenya, Tanzanie, Éthiopie. Herbes + acacias épars."
    ),

    # --- SERENGETI / GRASSLANDS FERTILES ---
    "AF_GRASSLAND_FERTILE": FuelModel(
        code="AF_GRASSLAND_FERTILE", name="Grassland fertile (Serengeti)",
        w_1h=0.15, w_10h=0.02, w_100h=0, w_live_herb=0.20, w_live_woody=0,
        delta=0.80, sigma_1h=7500, sigma_live_herb=7500, sigma_live_woody=4921,
        h_dead=18600, h_live=18600, mx=18,
        is_dynamic=True, region="east_africa",
        description="Grassland fertile (Serengeti, Masai Mara). "
                    "Themeda triandra, Digitaria. Hautes herbes, charge élevée. "
                    "Feux intenses de surface, combustion quasi-totale."
    ),

    # --- FYNBOS (Afrique du Sud) ---
    "AF_FYNBOS": FuelModel(
        code="AF_FYNBOS", name="Fynbos (Afrique du Sud)",
        w_1h=0.20, w_10h=0.10, w_100h=0.05, w_live_herb=0.05, w_live_woody=0.35,
        delta=0.70, sigma_1h=8000, sigma_live_herb=8000, sigma_live_woody=4500,
        h_dead=20000, h_live=21000, mx=25,
        is_dynamic=True, region="south_africa",
        description="Fynbos du Cap (Protea, Erica, Restio). "
                    "Shrubland dense, feuillage coriace, résineux. "
                    "Feux cycliques (12-15 ans), très intenses, régénération obligatoire."
    ),
    "AF_FYNBOS_YOUNG": FuelModel(
        code="AF_FYNBOS_YOUNG", name="Fynbos jeune (< 5 ans post-feu)",
        w_1h=0.08, w_10h=0.03, w_100h=0, w_live_herb=0.02, w_live_woody=0.10,
        delta=0.30, sigma_1h=8000, sigma_live_herb=8000, sigma_live_woody=5000,
        h_dead=20000, h_live=21000, mx=25,
        is_dynamic=True, region="south_africa",
        description="Fynbos jeune, peu de fuel. Feux possibles mais moins intenses."
    ),

    # --- BUSHVELD (Afrique du Sud) ---
    "AF_BUSHVELD": FuelModel(
        code="AF_BUSHVELD", name="Bushveld (Afrique du Sud)",
        w_1h=0.18, w_10h=0.12, w_100h=0.05, w_live_herb=0.08, w_live_woody=0.30,
        delta=0.80, sigma_1h=5500, sigma_live_herb=5500, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=22,
        is_dynamic=True, region="south_africa",
        description="Bushveld sud-africain. Combretum, Terminalia, Acacia. "
                    "Herbes + arbustes + arbres. Feux de surface + brandons."
    ),

    # --- BAOBAB SAVANNA (Madagascar, Afrique de l'Ouest) ---
    "AF_BAOBAB": FuelModel(
        code="AF_BAOBAB", name="Savane à baobabs (Adansonia digitata)",
        w_1h=0.08, w_10h=0.02, w_100h=0, w_live_herb=0.10, w_live_woody=0.05,
        delta=0.40, sigma_1h=7000, sigma_live_herb=7000, sigma_live_woody=3000,
        h_dead=18600, h_live=19000, mx=18,
        is_dynamic=True, region="west_africa",
        description="Savane à baobabs (Adansonia digitata). "
                    "Mali, Burkina, Sénégal. Herbes entre arbres géants. "
                    "Feux de surface, peu de combustible ligneux."
    ),

    # --- FORÊT TROPICALE SECHE (Guinée, Côte d'Ivoire) ---
    "AF_FOREST_DRY": FuelModel(
        code="AF_FOREST_DRY", name="Forêt tropicale sèche (Guinée)",
        w_1h=0.20, w_10h=0.15, w_100h=0.08, w_live_herb=0.05, w_live_woody=0.40,
        delta=0.50, sigma_1h=5000, sigma_live_herb=5000, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=28,
        is_dynamic=True, region="west_africa",
        description="Forêt tropicale sèche (Daniellia, Isoberlinia). "
                    "Guinée, Côte d'Ivoire, Ghana. Feuilles caduques, lit dense. "
                    "Feux de surface annuels, rarement cime."
    ),

    # --- AFROMONTANE (Éthiopie, Kenya, Tanzanie, RDC) ---
    "AF_AFROMONTANE": FuelModel(
        code="AF_AFROMONTANE", name="Forêt afromontane",
        w_1h=0.25, w_10h=0.20, w_100h=0.10, w_live_herb=0.03, w_live_woody=0.50,
        delta=0.40, sigma_1h=5500, sigma_live_herb=5500, sigma_live_woody=3500,
        h_dead=20000, h_live=20500, mx=35,
        is_dynamic=True, region="afromontane",
        description="Forêt afromontane (Juniperus, Podocarpus, Hagenia). "
                    "Éthiopie (Bale), Kenya (Aberdare), Tanzanie (Kilimandjaro). "
                    "Humide, lit dense, feux rares mais intenses."
    ),

    # --- MANGROVE (Afrique de l'Ouest) ---
    "AF_MANGROVE": FuelModel(
        code="AF_MANGROVE", name="Mangrove (Rhizophora, Avicennia)",
        w_1h=0.15, w_10h=0.10, w_100h=0.05, w_live_herb=0.02, w_live_woody=0.25,
        delta=0.50, sigma_1h=4500, sigma_live_herb=4500, sigma_live_woody=2500,
        h_dead=18000, h_live=18500, mx=35,
        is_dynamic=True, region="coastal",
        description="Mangrove d'Afrique de l'Ouest. "
                    "Rhizophora, Avicennia. Bois dense, salin, très humide. "
                    "Feux rares (sécheresse extrême), combustion incomplète."
    ),

    # --- PÂTURAGES / RANGELANDS ---
    "AF_RANGE_DEGRADED": FuelModel(
        code="AF_RANGE_DEGRADED", name="Pâturage dégradé (surpâturage)",
        w_1h=0.03, w_10h=0, w_100h=0, w_live_herb=0.02, w_live_woody=0.01,
        delta=0.10, sigma_1h=8000, sigma_live_herb=8000, sigma_live_woody=4921,
        h_dead=18600, h_live=18600, mx=10,
        is_dynamic=True, region="general",
        description="Pâturage surexploité. Très peu de combustible. "
                    "Feux possibles mais propagation difficile."
    ),
    "AF_RANGE_INTACT": FuelModel(
        code="AF_RANGE_INTACT", name="Pâturage intact (Afrique de l'Est)",
        w_1h=0.12, w_10h=0.02, w_100h=0, w_live_herb=0.15, w_live_woody=0.02,
        delta=0.60, sigma_1h=7500, sigma_live_herb=7500, sigma_live_woody=4921,
        h_dead=18600, h_live=18600, mx=18,
        is_dynamic=True, region="east_africa",
        description="Pâturage intact (Masai, Samburu). Herbes hautes, charge élevée."
    ),
}


# =============================================================================
# REGROUPEMENT DE TOUS LES MODÈLES
# =============================================================================

ALL_FUEL_MODELS: Dict[str, FuelModel] = {}
ALL_FUEL_MODELS.update(BEHAVE_STANDARD)
ALL_FUEL_MODELS.update(AFRICA_NORTH)
ALL_FUEL_MODELS.update(AFRICA_SAVANNA)


# =============================================================================
# MAPPING ESPÈCES VISION → FUEL MODELS
# =============================================================================
# Ton modèle de vision reconnaît ces 16 espèces :
# acacia, andropogon, colophospermum, euphorbia, macaranga, tamarix,
# adansonia, baobab, combretum, ficus, protea, themeda, aloe,
# brachystegia, erica, khaya, senegalia
# =============================================================================

SPECIES_TO_FUEL_MODEL: Dict[str, str] = {
    # Herbes / Graminées
    "andropogon": "AF_SUDAN_GRASS",      # Grande herbe des savanes
    "themeda": "AF_GRASSLAND_FERTILE",   # Themeda triandra (Serengeti)

    # Arbustes épineux / Steppe
    "acacia": "AF_ACACIA_SAVANNA",       # Acacia (générique Afrique)
    "senegalia": "AF_SAHEL_WOODED",      # Senegalia (ex-Acacia)
    "tamarix": "AF_TAMARIX",             # Tamarix (ripisylve)

    # Arbustes toxiques / Maquis
    "euphorbia": "AF_MAQUIS",            # Euphorbia (maquis méditerranéen)
    "erica": "AF_FYNBOS",                # Erica (fynbos)
    "protea": "AF_FYNBOS",               # Protea (fynbos)

    # Bois durs / Forêts claires
    "colophospermum": "AF_MOPANE",       # Mopane
    "brachystegia": "AF_MIOMBO",         # Miombo
    "khaya": "AF_FOREST_DRY",            # Acajou (forêt tropicale sèche)

    # Arbres à feuilles larges
    "macaranga": "AF_FOREST_DRY",        # Forêt secondaire
    "ficus": "AF_FOREST_DRY",            # Forêt galerie
    "combretum": "AF_SUDAN_WOODED",      # Combretum (soudanien)

    # Succulentes / Spéciaux
    "aloe": "AF_STEPPE",                 # Aloe (steppe sèche)
    "adansonia": "AF_BAOBAB",            # Baobab
    "baobab": "AF_BAOBAB",               # Baobab (alias)
}


# =============================================================================
# MAPPING ÉCOSYSTÈMES → FUEL MODELS (fallback si vision échoue)
# =============================================================================

ECOSYSTEM_TO_FUEL_MODEL: Dict[str, str] = {
    # Maroc / Afrique du Nord
    "steppe": "AF_STEPPE",
    "steppe_dense": "AF_STEPPE_DENSE",
    "argan": "AF_ARGAN",
    "chene_liege": "AF_CHENE_LIEGE",
    "cedre": "AF_CEDRE",
    "maquis": "AF_MAQUIS",
    "plaine_cerealiere": "AF_CEREALES",
    "palmeraie": "AF_PALMIER",
    "jujubier": "AF_JUJUBIER",

    # Sahel / Soudan
    "sahel_herbeuse": "AF_SAHEL_GRASS",
    "sahel_arboree": "AF_SAHEL_WOODED",
    "soudan_herbeuse": "AF_SUDAN_GRASS",
    "soudan_boisee": "AF_SUDAN_WOODED",

    # Afrique australe / centrale
    "miombo": "AF_MIOMBO",
    "miombo_dense": "AF_MIOMBO_DENSE",
    "mopane": "AF_MOPANE",
    "fynbos": "AF_FYNBOS",
    "fynbos_jeune": "AF_FYNBOS_YOUNG",
    "bushveld": "AF_BUSHVELD",

    # Afrique de l'Est / Ouest
    "acacia_savanna": "AF_ACACIA_SAVANNA",
    "grassland_fertile": "AF_GRASSLAND_FERTILE",
    "baobab_savanna": "AF_BAOBAB",
    "forest_dry": "AF_FOREST_DRY",
    "afromontane": "AF_AFROMONTANE",
    "mangrove": "AF_MANGROVE",

    # Génériques
    "range_degraded": "AF_RANGE_DEGRADED",
    "range_intact": "AF_RANGE_INTACT",

    # Fallbacks Behave
    "grass_short": "GR1",
    "grass_tall": "GR7",
    "grass_humid": "GR5",
    "shrub_dry": "SH5",
    "shrub_humid": "SH8",
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_fuel_model(code: str) -> Optional[FuelModel]:
    """
    Récupère un fuel model par son code.

    Args:
        code: Code du fuel model (ex: "AF_STEPPE", "GR1")

    Returns:
        FuelModel ou None si non trouvé
    """
    return ALL_FUEL_MODELS.get(code)


def get_fuel_model_by_species(species_name: str) -> Optional[FuelModel]:
    """
    Récupère un fuel model à partir du nom d'espèce reconnu par la vision.

    Args:
        species_name: Nom de l'espèce (ex: "acacia", "brachystegia")

    Returns:
        FuelModel ou None si l'espèce n'est pas reconnue
    """
    code = SPECIES_TO_FUEL_MODEL.get(species_name.lower())
    if code:
        return get_fuel_model(code)
    return None


def get_fuel_model_by_ecosystem(ecosystem: str) -> Optional[FuelModel]:
    """
    Récupère un fuel model à partir du nom d'écosystème.

    Args:
        ecosystem: Nom de l'écosystème (ex: "steppe", "miombo")

    Returns:
        FuelModel ou None si non trouvé
    """
    code = ECOSYSTEM_TO_FUEL_MODEL.get(ecosystem.lower())
    if code:
        return get_fuel_model(code)
    return None


def list_fuel_models(region: Optional[str] = None) -> List[str]:
    """
    Liste tous les codes de fuel models disponibles.

    Args:
        region: Filtrer par région ("north_africa", "sahel", "miombo", etc.)

    Returns:
        Liste des codes
    """
    if region is None:
        return list(ALL_FUEL_MODELS.keys())
    return [code for code, fm in ALL_FUEL_MODELS.items() if fm.region == region]


def list_regions() -> List[str]:
    """Liste toutes les régions disponibles."""
    regions = set(fm.region for fm in ALL_FUEL_MODELS.values())
    return sorted(list(regions))


def list_species() -> List[str]:
    """Liste toutes les espèces reconnues par le mapping."""
    return sorted(list(SPECIES_TO_FUEL_MODEL.keys()))


def list_ecosystems() -> List[str]:
    """Liste tous les écosystèmes du mapping."""
    return sorted(list(ECOSYSTEM_TO_FUEL_MODEL.keys()))


def get_fuel_model_info(code: str) -> Dict:
    """
    Retourne les informations détaillées d'un fuel model sous forme de dict.
    """
    fm = get_fuel_model(code)
    if not fm:
        return {}
    return {
        "code": fm.code,
        "name": fm.name,
        "region": fm.region,
        "description": fm.description,
        "w_total_kg_m2": fm.w_total,
        "w_dead_kg_m2": fm.w_dead,
        "w_live_kg_m2": fm.w_live,
        "delta_m": fm.delta,
        "sigma_1h": fm.sigma_1h,
        "mx_percent": fm.mx,
        "h_dead_kJ_kg": fm.h_dead,
        "h_live_kJ_kg": fm.h_live,
        "is_dynamic": fm.is_dynamic,
        "is_grass_dominant": fm.is_grass_dominant,
    }


def compute_dynamic_herb_load(fuel_model: FuelModel, live_herb_moisture: float) -> Tuple[float, float]:
    """
    Calcule la répartition herbe vivante / morte pour un modèle dynamique.

    Formule de Burgan (1979) / Scott & Burgan (2005):
    - Si M_herb >= 120% : 100% vivant
    - Si M_herb <= 30% : 100% mort (transféré vers w_1h)
    - Entre 30% et 120% : interpolation linéaire

    Args:
        fuel_model: Le fuel model (doit être dynamique)
        live_herb_moisture: Humidité des herbes vivantes (%)

    Returns:
        (w_live_herb_adjusted, w_1h_adjusted)
    """
    if not fuel_model.is_dynamic or fuel_model.w_live_herb == 0:
        return fuel_model.w_live_herb, fuel_model.w_1h

    m = live_herb_moisture
    w_herb_total = fuel_model.w_live_herb

    if m >= 120:
        # Tout reste vivant
        return w_herb_total, fuel_model.w_1h
    elif m <= 30:
        # Tout transféré vers mort
        return 0.0, fuel_model.w_1h + w_herb_total
    else:
        # Interpolation linéaire
        fraction_dead = (120 - m) / (120 - 30)  # 0 à 1
        w_dead_transferred = w_herb_total * fraction_dead
        w_live_remaining = w_herb_total - w_dead_transferred
        return w_live_remaining, fuel_model.w_1h + w_dead_transferred


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FUEL MODELS BURNTRACK - AFRIQUE")
    print("=" * 70)

    print(f"\nTotal fuel models: {len(ALL_FUEL_MODELS)}")
    print(f"  - Behave standard: {len(BEHAVE_STANDARD)}")
    print(f"  - Afrique du Nord: {len(AFRICA_NORTH)}")
    print(f"  - Afrique subsaharienne: {len(AFRICA_SAVANNA)}")

    print(f"\nRégions disponibles: {list_regions()}")
    print(f"\nEspèces reconnues ({len(list_species())}):")
    for sp in list_species():
        fm_code = SPECIES_TO_FUEL_MODEL[sp]
        fm = get_fuel_model(fm_code)
        print(f"  {sp:20s} → {fm_code:25s} ({fm.name if fm else 'N/A'})")

    print(f"\nÉcosystèmes disponibles ({len(list_ecosystems())}):")
    for eco in list_ecosystems()[:10]:
        print(f"  {eco}")
    print(f"  ... et {len(list_ecosystems()) - 10} autres")

    # Test d'un fuel model
    print("\n" + "=" * 70)
    print("TEST - Fuel Model: AF_MIOMBO")
    print("=" * 70)
    fm = get_fuel_model("AF_MIOMBO")
    if fm:
        info = get_fuel_model_info("AF_MIOMBO")
        for k, v in info.items():
            print(f"  {k:25s}: {v}")

    # Test mapping espèce
    print("\n" + "=" * 70)
    print("TEST - Mapping espèce: brachystegia")
    print("=" * 70)
    fm = get_fuel_model_by_species("brachystegia")
    if fm:
        print(f"  Espèce 'brachystegia' → {fm.code}: {fm.name}")
        print(f"  Charge totale: {fm.w_total:.3f} kg/m²")
        print(f"  Profondeur: {fm.delta} m")

    # Test dynamique
    print("\n" + "=" * 70)
    print("TEST - Modèle dynamique (AF_GRASSLAND_FERTILE)")
    print("=" * 70)
    fm = get_fuel_model("AF_GRASSLAND_FERTILE")
    for m in [150, 120, 75, 30, 20]:
        w_live, w_dead = compute_dynamic_herb_load(fm, m)
        print(f"  M_herb={m:3d}% → w_live={w_live:.3f}, w_1h={w_dead:.3f} kg/m²")
        