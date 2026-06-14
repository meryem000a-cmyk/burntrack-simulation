"""
fuel_models.py
================
Fuel Models pour BurnTrack - Adaptés au Maroc et à l'Afrique.
Unités SI : kg/m², m, m²/m³, kJ/kg, %

CORRECTIONS v2 appliquées :
1. Loadings BEHAVE_STANDARD ×10 (erreur conversion lb/ft² → kg/m²)
2. Ajout sigma_10h/sigma_100h manquants
3. Commentaire mx corrigé (% au lieu de fraction)
4. Validation __post_init__ pour sécurité
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FuelModel:
    """Modèle de combustible pour BurnTrack."""
    code: str
    name: str
    w_1h: float
    w_10h: float
    w_100h: float
    w_live_herb: float
    w_live_woody: float
    delta: float
    sigma_1h: float
    sigma_10h: float = 0.0      # m²/m³, SAV mort moyen (défaut 0 si absent)
    sigma_100h: float = 0.0     # m²/m³, SAV mort gros (défaut 0 si absent)
    sigma_live_herb: float = 0.0  # m²/m³, SAV vivant herbacé
    sigma_live_woody: float = 0.0  # m²/m³, SAV vivant ligneux
    h_dead: float = 18608
    h_live: float = 18608
    mx: float = 15              # % (pourcent), humidité d'extinction morts
    is_dynamic: bool = False
    region: str = "general"
    description: str = ""

    def __post_init__(self):
        """Validation de base pour éviter des modèles corrompus."""
        if self.w_1h < 0 or self.w_10h < 0 or self.w_100h < 0 or            self.w_live_herb < 0 or self.w_live_woody < 0:
            raise ValueError(f"Charges négatives interdites pour {self.code}")
        if self.delta <= 0:
            raise ValueError(f"Profondeur delta <= 0 interdite pour {self.code}")
        if self.sigma_1h <= 0:
            raise ValueError(f"SAV sigma_1h <= 0 interdite pour {self.code}")
        if self.mx <= 0 or self.mx > 100:
            raise ValueError(f"mx={self.mx} hors plage [0, 100] pour {self.code}")

    @property
    def w_total(self) -> float:
        return self.w_1h + self.w_10h + self.w_100h + self.w_live_herb + self.w_live_woody

    @property
    def w_dead(self) -> float:
        return self.w_1h + self.w_10h + self.w_100h

    @property
    def w_live(self) -> float:
        return self.w_live_herb + self.w_live_woody

    def __repr__(self) -> str:
        return f"FuelModel({self.code}: {self.name}, w_total={self.w_total:.3f} kg/m²)"


# =============================================================================
# MODÈLES STANDARD BEHAVE (convertis en SI depuis lb/ft², ft, BTU/lb)
# CORRECTION v2 : Loadings ×10 (erreur conversion originale)
# =============================================================================

BEHAVE_STANDARD: Dict[str, FuelModel] = {
    "GR1": FuelModel(
        code="GR1", name="Short, sparse, dry climate grass",
        w_1h=1.66, w_10h=0.0, w_100h=0.0, w_live_herb=0.67, w_live_woody=0.0,
        delta=0.305, sigma_1h=11483, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=10499, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes courtes, clairsemées, climat sec."
    ),
    "GR2": FuelModel(
        code="GR2", name="Low load, dry climate grass",
        w_1h=1.66, w_10h=0.0, w_100h=0.0, w_live_herb=2.20, w_live_woody=0.0,
        delta=0.305, sigma_1h=10499, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à faible charge, climat sec."
    ),
    "GR3": FuelModel(
        code="GR3", name="Low load, very coarse, humid climate grass",
        w_1h=1.66, w_10h=6.69, w_100h=0.0, w_live_herb=3.27, w_live_woody=0.0,
        delta=0.610, sigma_1h=7874, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=6823, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Herbes grossières, climat humide."
    ),
    "GR4": FuelModel(
        code="GR4", name="Moderate load, dry climate grass",
        w_1h=4.15, w_10h=0.0, w_100h=0.0, w_live_herb=4.18, w_live_woody=0.0,
        delta=0.610, sigma_1h=10499, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à charge modérée, climat sec."
    ),
    "GR5": FuelModel(
        code="GR5", name="Low load, humid climate grass",
        w_1h=6.67, w_10h=0.0, w_100h=0.0, w_live_herb=5.49, w_live_woody=0.0,
        delta=0.457, sigma_1h=9449, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=8399, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à faible charge, climat humide."
    ),
    "GR6": FuelModel(
        code="GR6", name="Moderate load, humid climate grass",
        w_1h=1.66, w_10h=0.0, w_100h=0.0, w_live_herb=7.47, w_live_woody=0.0,
        delta=0.457, sigma_1h=11483, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=10499, sigma_live_woody=0.0,
        h_dead=20934, h_live=20934, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à charge modérée, climat humide."
    ),
    "GR7": FuelModel(
        code="GR7", name="High load, dry climate grass",
        w_1h=16.28, w_10h=0.0, w_100h=0.0, w_live_herb=11.89, w_live_woody=0.0,
        delta=0.914, sigma_1h=10499, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes à haute charge, climat sec."
    ),
    "GR8": FuelModel(
        code="GR8", name="High load, very coarse, humid climate grass",
        w_1h=7.96, w_10h=16.28, w_100h=0.0, w_live_herb=16.03, w_live_woody=0.0,
        delta=1.219, sigma_1h=7874, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=6823, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Herbes très grossières, haute charge, humide."
    ),
    "GR9": FuelModel(
        code="GR9", name="Very high load, humid climate grass",
        w_1h=16.28, w_10h=16.28, w_100h=0.0, w_live_herb=19.76, w_live_woody=0.0,
        delta=1.524, sigma_1h=9449, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=8399, sigma_live_woody=0.0,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes à très haute charge, humide."
    ),
    "GS1": FuelModel(
        code="GS1", name="Low load, dry climate grass-shrub",
        w_1h=3.32, w_10h=0.0, w_100h=0.0, w_live_herb=1.08, w_live_woody=1.44,
        delta=0.274, sigma_1h=10499, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=9449,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, faible charge, sec."
    ),
    "GS2": FuelModel(
        code="GS2", name="Moderate load, dry climate grass-shrub",
        w_1h=7.96, w_10h=7.96, w_100h=0.0, w_live_herb=1.31, w_live_woody=2.20,
        delta=0.457, sigma_1h=10499, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=9449,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, charge modérée, sec."
    ),
    "GS3": FuelModel(
        code="GS3", name="Moderate load, humid climate grass-shrub",
        w_1h=4.98, w_10h=4.15, w_100h=0.0, w_live_herb=3.18, w_live_woody=2.73,
        delta=0.549, sigma_1h=9449, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=8399, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, charge modérée, humide."
    ),
    "GS4": FuelModel(
        code="GS4", name="High load, humid climate grass-shrub",
        w_1h=30.91, w_10h=4.98, w_100h=1.66, w_live_herb=7.47, w_live_woody=15.61,
        delta=0.640, sigma_1h=9449, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=8399, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Herbes + arbustes, haute charge, humide."
    ),
    "SH1": FuelModel(
        code="SH1", name="Low load, dry climate shrub",
        w_1h=4.15, w_10h=4.15, w_100h=0.0, w_live_herb=0.31, w_live_woody=2.83,
        delta=0.305, sigma_1h=10499, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à faible charge, sec."
    ),
    "SH2": FuelModel(
        code="SH2", name="Moderate load, dry climate shrub",
        w_1h=22.87, w_10h=40.57, w_100h=12.86, w_live_herb=0.0, w_live_woody=65.16,
        delta=0.305, sigma_1h=10499, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=0.0, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à charge modérée, sec."
    ),
    "SH3": FuelModel(
        code="SH3", name="Moderate load, humid climate shrub",
        w_1h=7.62, w_10h=50.97, w_100h=0.0, w_live_herb=0.0, w_live_woody=105.15,
        delta=0.732, sigma_1h=8399, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=0.0, sigma_live_woody=7349,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à charge modérée, humide."
    ),
    "SH4": FuelModel(
        code="SH4", name="Low load, humid climate timber-shrub",
        w_1h=14.55, w_10h=19.40, w_100h=3.47, w_live_herb=0.0, w_live_woody=43.33,
        delta=0.914, sigma_1h=10499, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=0.0, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Arbustes + forêt claire, faible charge, humide."
    ),
    "SH5": FuelModel(
        code="SH5", name="High load, dry climate shrub",
        w_1h=61.00, w_10h=35.58, w_100h=0.0, w_live_herb=0.0, w_live_woody=49.20,
        delta=1.829, sigma_1h=3937, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=0.0, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à haute charge, sec."
    ),
    "SH6": FuelModel(
        code="SH6", name="Low load, humid climate shrub",
        w_1h=49.22, w_10h=24.61, w_100h=0.0, w_live_herb=0.0, w_live_woody=23.54,
        delta=0.610, sigma_1h=3937, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=0.0, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=30,
        is_dynamic=True, region="general",
        description="Arbustes à faible charge, humide."
    ),
    "SH7": FuelModel(
        code="SH7", name="Very high load, dry climate shrub",
        w_1h=59.29, w_10h=89.77, w_100h=37.04, w_live_herb=0.0, w_live_woody=57.54,
        delta=1.829, sigma_1h=3937, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=0.0, sigma_live_woody=8399,
        h_dead=18608, h_live=18608, mx=15,
        is_dynamic=True, region="general",
        description="Arbustes à très haute charge, sec."
    ),
    "SH8": FuelModel(
        code="SH8", name="High load, humid climate shrub",
        w_1h=34.66, w_10h=57.54, w_100h=14.55, w_live_herb=0.0, w_live_woody=73.48,
        delta=0.914, sigma_1h=3937, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=0.0, sigma_live_woody=7874,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à haute charge, humide."
    ),
    "SH9": FuelModel(
        code="SH9", name="Very high load, humid climate shrub",
        w_1h=76.26, w_10h=41.55, w_100h=0.0, w_live_herb=2.45, w_live_woody=118.49,
        delta=1.341, sigma_1h=3937, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=9449, sigma_live_woody=7874,
        h_dead=18608, h_live=18608, mx=40,
        is_dynamic=True, region="general",
        description="Arbustes à très haute charge, humide."
    ),
}


# =============================================================================
# MODÈLES AFRIQUE DU NORD / MAROC
# CORRECTION v2 : Ajout sigma_10h/sigma_100h manquants
# =============================================================================

AFRICA_NORTH: Dict[str, FuelModel] = {
    "AF_STEPPE": FuelModel(
        code="AF_STEPPE", name="Steppe marocaine (Artemisia, Stipa)",
        w_1h=0.25, w_10h=0.05, w_100h=0.0, w_live_herb=0.10, w_live_woody=0.02,
        delta=0.25, sigma_1h=12000, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=3500,
        h_dead=18600, h_live=19000, mx=20,
        is_dynamic=True, region="north_africa",
        description="Steppe à Artemisia herba-alba, Stipa tenacissima. Climat aride."
    ),
    "AF_STEPPE_DENSE": FuelModel(
        code="AF_STEPPE_DENSE", name="Steppe dense (haut Atlas)",
        w_1h=0.40, w_10h=0.10, w_100h=0.02, w_live_herb=0.15, w_live_woody=0.05,
        delta=0.40, sigma_1h=10000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=10000, sigma_live_woody=3000,
        h_dead=18600, h_live=19000, mx=15,
        is_dynamic=True, region="north_africa",
        description="Steppe dense de montagne."
    ),
    "AF_ARGAN": FuelModel(
        code="AF_ARGAN", name="Arganeraie (Argania spinosa)",
        w_1h=0.20, w_10h=0.30, w_100h=0.05, w_live_herb=0.02, w_live_woody=0.40,
        delta=0.35, sigma_1h=8000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=8000, sigma_live_woody=2800,
        h_dead=18600, h_live=19500, mx=18,
        is_dynamic=True, region="north_africa",
        description="Forêt d'arganier. Arbres tordus, feuillage persistant."
    ),
    "AF_CHENE_LIEGE": FuelModel(
        code="AF_CHENE_LIEGE", name="Chênaie-liège (Quercus suber)",
        w_1h=0.50, w_10h=1.00, w_100h=0.20, w_live_herb=0.05, w_live_woody=0.80,
        delta=0.30, sigma_1h=8000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=8000, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=25,
        is_dynamic=True, region="north_africa",
        description="Forêt de chêne-liège. Mamora, Rif."
    ),
    "AF_CEDRE": FuelModel(
        code="AF_CEDRE", name="Cèdraie de l'Atlas (Cedrus atlantica)",
        w_1h=0.30, w_10h=0.80, w_100h=0.30, w_live_herb=0.02, w_live_woody=0.60,
        delta=0.25, sigma_1h=10000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=10000, sigma_live_woody=4000,
        h_dead=20000, h_live=20500, mx=30,
        is_dynamic=True, region="north_africa",
        description="Forêt de cèdre de l'Atlas. Moyen Atlas, Haut Atlas."
    ),
    "AF_MAQUIS": FuelModel(
        code="AF_MAQUIS", name="Maquis méditerranéen (Rif, Moyen Atlas)",
        w_1h=0.60, w_10h=0.50, w_100h=0.10, w_live_herb=0.10, w_live_woody=0.50,
        delta=0.50, sigma_1h=7000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=7000, sigma_live_woody=3000,
        h_dead=19000, h_live=19500, mx=20,
        is_dynamic=True, region="north_africa",
        description="Maquis méditerranéen. Dense, persistant, résineux."
    ),
    "AF_CEREALES": FuelModel(
        code="AF_CEREALES", name="Plaine céréalière (résidus de récolte)",
        w_1h=1.50, w_10h=0.0, w_100h=0.0, w_live_herb=0.0, w_live_woody=0.0,
        delta=0.10, sigma_1h=15000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=15000, sigma_live_woody=0.0,
        h_dead=17000, h_live=17000, mx=18,
        is_dynamic=False, region="north_africa",
        description="Résidus de céréales après récolte. Haouz, Tadla."
    ),
    "AF_PALMIER": FuelModel(
        code="AF_PALMIER", name="Palmeraie (Phoenix dactylifera)",
        w_1h=0.30, w_10h=0.15, w_100h=0.05, w_live_herb=0.02, w_live_woody=0.20,
        delta=0.30, sigma_1h=6000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=6000, sigma_live_woody=2500,
        h_dead=18000, h_live=18500, mx=15,
        is_dynamic=True, region="north_africa",
        description="Palmeraie oasienne. Feuilles mortes, dattes."
    ),
    "AF_TAMARIX": FuelModel(
        code="AF_TAMARIX", name="Tamarix (ripisylve des oueds)",
        w_1h=0.20, w_10h=0.30, w_100h=0.10, w_live_herb=0.03, w_live_woody=0.35,
        delta=0.40, sigma_1h=7000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=7000, sigma_live_woody=2800,
        h_dead=18600, h_live=19500, mx=20,
        is_dynamic=True, region="north_africa",
        description="Ripisylve à Tamarix. Bois dense, salin."
    ),
    "AF_JUJUBIER": FuelModel(
        code="AF_JUJUBIER", name="Jujubier (Ziziphus lotus)",
        w_1h=0.15, w_10h=0.10, w_100h=0.03, w_live_herb=0.02, w_live_woody=0.20,
        delta=0.35, sigma_1h=6000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=6000, sigma_live_woody=2500,
        h_dead=18600, h_live=19000, mx=18,
        is_dynamic=True, region="north_africa",
        description="Arbustes à Ziziphus lotus. Épineux, persistant."
    ),
}


# =============================================================================
# MODÈLES AFRIQUE SUBSAHARIENNE
# CORRECTION v2 : Ajout sigma_10h/sigma_100h manquants
# =============================================================================

AFRICA_SAVANNA: Dict[str, FuelModel] = {
    "AF_SAHEL_GRASS": FuelModel(
        code="AF_SAHEL_GRASS", name="Savane sahélienne herbeuse",
        w_1h=0.15, w_10h=0.0, w_100h=0.0, w_live_herb=0.20, w_live_woody=0.0,
        delta=0.30, sigma_1h=15000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=15000, sigma_live_woody=0.0,
        h_dead=18600, h_live=18600, mx=20,
        is_dynamic=True, region="sahel",
        description="Savane sahélienne purement herbeuse. Andropogon, Loudetia."
    ),
    "AF_SAHEL_WOODED": FuelModel(
        code="AF_SAHEL_WOODED", name="Savane sahélienne arborée (Acacia)",
        w_1h=0.20, w_10h=0.05, w_100h=0.0, w_live_herb=0.25, w_live_woody=0.08,
        delta=0.40, sigma_1h=12000, sigma_10h=11483, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=3500,
        h_dead=18600, h_live=19000, mx=22,
        is_dynamic=True, region="sahel",
        description="Savane sahélienne arborée. Acacia senegal, A. tortilis."
    ),
    "AF_SUDAN_GRASS": FuelModel(
        code="AF_SUDAN_GRASS", name="Savane soudanienne herbeuse",
        w_1h=0.30, w_10h=0.05, w_100h=0.0, w_live_herb=0.40, w_live_woody=0.0,
        delta=0.60, sigma_1h=12000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=0.0,
        h_dead=18600, h_live=18600, mx=20,
        is_dynamic=True, region="sudan",
        description="Savane soudanienne herbeuse. Hyparrhenia, Andropogon."
    ),
    "AF_SUDAN_WOODED": FuelModel(
        code="AF_SUDAN_WOODED", name="Savane soudanienne boisée",
        w_1h=0.35, w_10h=0.12, w_100h=0.05, w_live_herb=0.30, w_live_woody=0.25,
        delta=0.70, sigma_1h=10000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=10000, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=22,
        is_dynamic=True, region="sudan",
        description="Savane soudanienne boisée. Combretum, Terminalia."
    ),
    "AF_MIOMBO": FuelModel(
        code="AF_MIOMBO", name="Miombo (Brachystegia, Julbernardia)",
        w_1h=0.40, w_10h=0.25, w_100h=0.10, w_live_herb=0.15, w_live_woody=0.40,
        delta=0.50, sigma_1h=9000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=9000, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=25,
        is_dynamic=True, region="miombo",
        description="Forêt claire de miombo. Zambie, Zimbabwe, Tanzanie."
    ),
    "AF_MIOMBO_DENSE": FuelModel(
        code="AF_MIOMBO_DENSE", name="Miombo dense",
        w_1h=0.50, w_10h=0.35, w_100h=0.15, w_live_herb=0.10, w_live_woody=0.60,
        delta=0.60, sigma_1h=8000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=8000, sigma_live_woody=2800,
        h_dead=20000, h_live=20500, mx=28,
        is_dynamic=True, region="miombo",
        description="Miombo dense, moins perturbé."
    ),
    "AF_MOPANE": FuelModel(
        code="AF_MOPANE", name="Mopane (Colophospermum mopane)",
        w_1h=0.30, w_10h=0.20, w_100h=0.08, w_live_herb=0.10, w_live_woody=0.30,
        delta=0.45, sigma_1h=8000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=8000, sigma_live_woody=2800,
        h_dead=19500, h_live=20000, mx=22,
        is_dynamic=True, region="mopane",
        description="Forêt claire de mopane. Botswana, Namibie, Zambie."
    ),
    "AF_ACACIA_SAVANNA": FuelModel(
        code="AF_ACACIA_SAVANNA", name="Savane à Acacia (Afrique de l'Est)",
        w_1h=0.25, w_10h=0.10, w_100h=0.03, w_live_herb=0.25, w_live_woody=0.20,
        delta=0.50, sigma_1h=10000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=10000, sigma_live_woody=3500,
        h_dead=18600, h_live=19500, mx=20,
        is_dynamic=True, region="east_africa",
        description="Savane à Acacia. Kenya, Tanzanie, Éthiopie."
    ),
    "AF_GRASSLAND_FERTILE": FuelModel(
        code="AF_GRASSLAND_FERTILE", name="Grassland fertile (Serengeti)",
        w_1h=0.40, w_10h=0.05, w_100h=0.0, w_live_herb=0.50, w_live_woody=0.0,
        delta=0.80, sigma_1h=12000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=0.0,
        h_dead=18600, h_live=18600, mx=18,
        is_dynamic=True, region="east_africa",
        description="Grassland fertile. Themeda triandra, Digitaria."
    ),
    "AF_FYNBOS": FuelModel(
        code="AF_FYNBOS", name="Fynbos (Afrique du Sud)",
        w_1h=0.50, w_10h=0.25, w_100h=0.10, w_live_herb=0.05, w_live_woody=0.50,
        delta=0.70, sigma_1h=12000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=12000, sigma_live_woody=4500,
        h_dead=20000, h_live=21000, mx=25,
        is_dynamic=True, region="south_africa",
        description="Fynbos du Cap. Protea, Erica, Restio."
    ),
    "AF_FYNBOS_YOUNG": FuelModel(
        code="AF_FYNBOS_YOUNG", name="Fynbos jeune (< 5 ans post-feu)",
        w_1h=0.20, w_10h=0.08, w_100h=0.0, w_live_herb=0.03, w_live_woody=0.15,
        delta=0.30, sigma_1h=12000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=5000,
        h_dead=20000, h_live=21000, mx=25,
        is_dynamic=True, region="south_africa",
        description="Fynbos jeune, peu de fuel."
    ),
    "AF_BUSHVELD": FuelModel(
        code="AF_BUSHVELD", name="Bushveld (Afrique du Sud)",
        w_1h=0.45, w_10h=0.30, w_100h=0.10, w_live_herb=0.15, w_live_woody=0.45,
        delta=0.80, sigma_1h=9000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=9000, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=22,
        is_dynamic=True, region="south_africa",
        description="Bushveld sud-africain. Combretum, Terminalia, Acacia."
    ),
    "AF_BAOBAB": FuelModel(
        code="AF_BAOBAB", name="Savane à baobabs (Adansonia digitata)",
        w_1h=0.20, w_10h=0.05, w_100h=0.0, w_live_herb=0.25, w_live_woody=0.08,
        delta=0.40, sigma_1h=12000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=3000,
        h_dead=18600, h_live=19000, mx=18,
        is_dynamic=True, region="west_africa",
        description="Savane à baobabs. Mali, Burkina, Sénégal."
    ),
    "AF_FOREST_DRY": FuelModel(
        code="AF_FOREST_DRY", name="Forêt tropicale sèche (Guinée)",
        w_1h=0.50, w_10h=0.35, w_100h=0.15, w_live_herb=0.08, w_live_woody=0.60,
        delta=0.50, sigma_1h=8000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=8000, sigma_live_woody=3000,
        h_dead=19500, h_live=20000, mx=28,
        is_dynamic=True, region="west_africa",
        description="Forêt tropicale sèche. Daniellia, Isoberlinia."
    ),
    "AF_AFROMONTANE": FuelModel(
        code="AF_AFROMONTANE", name="Forêt afromontane",
        w_1h=0.60, w_10h=0.45, w_100h=0.20, w_live_herb=0.05, w_live_woody=0.70,
        delta=0.40, sigma_1h=9000, sigma_10h=11483, sigma_100h=4572,
        sigma_live_herb=9000, sigma_live_woody=3500,
        h_dead=20000, h_live=20500, mx=35,
        is_dynamic=True, region="afromontane",
        description="Forêt afromontane. Juniperus, Podocarpus, Hagenia."
    ),
    "AF_MANGROVE": FuelModel(
        code="AF_MANGROVE", name="Mangrove (Rhizophora, Avicennia)",
        w_1h=0.35, w_10h=0.20, w_100h=0.10, w_live_herb=0.03, w_live_woody=0.35,
        delta=0.50, sigma_1h=7000, sigma_10h=11000, sigma_100h=4000,
        sigma_live_herb=7000, sigma_live_woody=2500,
        h_dead=18000, h_live=18500, mx=35,
        is_dynamic=True, region="coastal",
        description="Mangrove d'Afrique de l'Ouest. Rhizophora, Avicennia."
    ),
    "AF_RANGE_DEGRADED": FuelModel(
        code="AF_RANGE_DEGRADED", name="Pâturage dégradé (surpâturage)",
        w_1h=0.05, w_10h=0.0, w_100h=0.0, w_live_herb=0.03, w_live_woody=0.01,
        delta=0.10, sigma_1h=15000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=15000, sigma_live_woody=0.0,
        h_dead=18600, h_live=18600, mx=15,
        is_dynamic=True, region="general",
        description="Pâturage surexploité. Très peu de combustible."
    ),
    "AF_RANGE_INTACT": FuelModel(
        code="AF_RANGE_INTACT", name="Pâturage intact (Afrique de l'Est)",
        w_1h=0.30, w_10h=0.05, w_100h=0.0, w_live_herb=0.35, w_live_woody=0.03,
        delta=0.70, sigma_1h=12000, sigma_10h=0.0, sigma_100h=0.0,
        sigma_live_herb=12000, sigma_live_woody=0.0,
        h_dead=18600, h_live=18600, mx=20,
        is_dynamic=True, region="east_africa",
        description="Pâturage intact. Herbes hautes, charge élevée."
    ),
}


# =============================================================================
# REGROUPEMENT
# =============================================================================

ALL_FUEL_MODELS: Dict[str, FuelModel] = {}
ALL_FUEL_MODELS.update(BEHAVE_STANDARD)
ALL_FUEL_MODELS.update(AFRICA_NORTH)
ALL_FUEL_MODELS.update(AFRICA_SAVANNA)


# =============================================================================
# MAPPING ESPÈCES VISION → FUEL MODELS
# =============================================================================

SPECIES_TO_FUEL_MODEL: Dict[str, str] = {
    "andropogon": "AF_SUDAN_GRASS",
    "themeda": "AF_GRASSLAND_FERTILE",
    "acacia": "AF_ACACIA_SAVANNA",
    "senegalia": "AF_SAHEL_WOODED",
    "tamarix": "AF_TAMARIX",
    "euphorbia": "AF_MAQUIS",
    "erica": "AF_FYNBOS",
    "protea": "AF_FYNBOS",
    "colophospermum": "AF_MOPANE",
    "brachystegia": "AF_MIOMBO",
    "khaya": "AF_FOREST_DRY",
    "macaranga": "AF_FOREST_DRY",
    "ficus": "AF_FOREST_DRY",
    "combretum": "AF_SUDAN_WOODED",
    "aloe": "AF_STEPPE",
    "adansonia": "AF_BAOBAB",
    "baobab": "AF_BAOBAB",
}


# =============================================================================
# MAPPING ÉCOSYSTÈMES → FUEL MODELS
# =============================================================================

ECOSYSTEM_TO_FUEL_MODEL: Dict[str, str] = {
    "steppe": "AF_STEPPE",
    "steppe_dense": "AF_STEPPE_DENSE",
    "argan": "AF_ARGAN",
    "chene_liege": "AF_CHENE_LIEGE",
    "cedre": "AF_CEDRE",
    "maquis": "AF_MAQUIS",
    "plaine_cerealiere": "AF_CEREALES",
    "palmeraie": "AF_PALMIER",
    "jujubier": "AF_JUJUBIER",
    "sahel_herbeuse": "AF_SAHEL_GRASS",
    "sahel_arboree": "AF_SAHEL_WOODED",
    "soudan_herbeuse": "AF_SUDAN_GRASS",
    "soudan_boisee": "AF_SUDAN_WOODED",
    "miombo": "AF_MIOMBO",
    "miombo_dense": "AF_MIOMBO_DENSE",
    "mopane": "AF_MOPANE",
    "fynbos": "AF_FYNBOS",
    "fynbos_jeune": "AF_FYNBOS_YOUNG",
    "bushveld": "AF_BUSHVELD",
    "acacia_savanna": "AF_ACACIA_SAVANNA",
    "grassland_fertile": "AF_GRASSLAND_FERTILE",
    "baobab_savanna": "AF_BAOBAB",
    "forest_dry": "AF_FOREST_DRY",
    "afromontane": "AF_AFROMONTANE",
    "mangrove": "AF_MANGROVE",
    "range_degraded": "AF_RANGE_DEGRADED",
    "range_intact": "AF_RANGE_INTACT",
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
    return ALL_FUEL_MODELS.get(code)


def get_fuel_model_by_species(species_name: str) -> Optional[FuelModel]:
    code = SPECIES_TO_FUEL_MODEL.get(species_name.lower())
    if code:
        return get_fuel_model(code)
    return None


def get_fuel_model_by_ecosystem(ecosystem: str) -> Optional[FuelModel]:
    code = ECOSYSTEM_TO_FUEL_MODEL.get(ecosystem.lower())
    if code:
        return get_fuel_model(code)
    return None


def list_fuel_models(region: Optional[str] = None) -> List[str]:
    if region is None:
        return list(ALL_FUEL_MODELS.keys())
    return [code for code, fm in ALL_FUEL_MODELS.items() if fm.region == region]


def list_regions() -> List[str]:
    regions = set(fm.region for fm in ALL_FUEL_MODELS.values())
    return sorted(list(regions))


def list_species() -> List[str]:
    return sorted(list(SPECIES_TO_FUEL_MODEL.keys()))


def list_ecosystems() -> List[str]:
    return sorted(list(ECOSYSTEM_TO_FUEL_MODEL.keys()))


def get_fuel_model_info(code: str) -> Dict:
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
    }


def compute_dynamic_herb_load(fuel_model: FuelModel, live_herb_moisture: float) -> Tuple[float, float]:
    if not fuel_model.is_dynamic or fuel_model.w_live_herb == 0:
        return fuel_model.w_live_herb, fuel_model.w_1h

    m = live_herb_moisture
    w_herb_total = fuel_model.w_live_herb

    if m >= 120:
        return w_herb_total, fuel_model.w_1h
    elif m <= 30:
        return 0.0, fuel_model.w_1h + w_herb_total
    else:
        fraction_dead = (120 - m) / (120 - 30)
        w_dead_transferred = w_herb_total * fraction_dead
        w_live_remaining = w_herb_total - w_dead_transferred
        return w_live_remaining, fuel_model.w_1h + w_dead_transferred


if __name__ == "__main__":
    print("=" * 70)
    print("FUEL MODELS BURNTRACK - AFRIQUE v2 (CORRIGÉ)")
    print("=" * 70)
    print(f"\nTotal fuel models: {len(ALL_FUEL_MODELS)}")
    print(f"  - Behave standard: {len(BEHAVE_STANDARD)}")
    print(f"  - Afrique du Nord: {len(AFRICA_NORTH)}")
    print(f"  - Afrique subsaharienne: {len(AFRICA_SAVANNA)}")
    print(f"\nRégions: {list_regions()}")
    print(f"\nEspèces ({len(list_species())}): {', '.join(list_species())}")

    # Test de validation
    print("\n" + "=" * 70)
    print("VALIDATION DES MODÈLES")
    print("=" * 70)
    for code in ["GR1", "AF_STEPPE", "AF_MIOMBO"]:
        fm = get_fuel_model(code)
        if fm:
            print(f"\n{code}: {fm.name}")
            print(f"  w_total={fm.w_total:.2f} kg/m², delta={fm.delta:.2f} m")
            print(f"  sigma_1h={fm.sigma_1h}, sigma_10h={fm.sigma_10h}, sigma_100h={fm.sigma_100h}")
            print(f"  mx={fm.mx}% (fraction={fm.mx/100:.2f})")