"""
download_era5_africa.py
=======================
Script optionnel pour télécharger les données ERA5-Land sur des zones de feu
en Afrique. Utilise l'API CDS (Copernicus Data Store).

Prérequis :
    pip install cdsapi
    # Créer un compte sur https://cds.climate.copernicus.eu/
    # Configurer ~/.cdsapirc avec votre UID et API key

Zones couvertes :
    - Sahel (Burkina Faso, Mali, Niger)
    - Afrique du Sud (Kruger, Fynbos)
    - Madagascar
    - Kenya/Tanzanie (Miombo)

Variables téléchargées :
    - 2m temperature
    - 2m dewpoint temperature
    - 10m u/v wind components
    - Total precipitation
    - Surface solar radiation
"""

import os
import cdsapi
import xarray as xr
from typing import List, Dict, Tuple
from datetime import datetime


# =============================================================================
# ZONES D'INTÉRÊT (bbox : [N, W, S, E])
# =============================================================================

AFRICA_FIRE_ZONES = {
    'sahel': {
        'bbox': [20.0, -15.0, 10.0, 15.0],  # Burkina, Mali, Niger
        'description': 'Sahel — Herbes sèches, feux de brousse',
    },
    'south_africa_kruger': {
        'bbox': [-22.0, 30.0, -26.0, 33.0],  # Parc Kruger
        'description': 'Afrique du Sud (Kruger) — Savane, Miombo',
    },
    'south_africa_fynbos': {
        'bbox': [-33.0, 18.0, -35.0, 21.0],  # Cape Peninsula
        'description': 'Afrique du Sud (Fynbos) — Fynbos méditerranéen',
    },
    'madagascar': {
        'bbox': [-11.0, 43.0, -26.0, 51.0],  # Madagascar
        'description': 'Madagascar — Forêt sèche, savane',
    },
    'east_africa': {
        'bbox': [5.0, 33.0, -5.0, 42.0],  # Kenya, Tanzanie
        'description': 'Afrique de l'Est — Miombo, savane',
    },
}


# =============================================================================
# VARIABLES ERA5-LAND
# =============================================================================

ERA5_VARIABLES = {
    '2m_temperature': '2m_temperature',
    '2m_dewpoint_temperature': '2m_dewpoint_temperature',
    '10m_u_component_of_wind': '10m_u_component_of_wind',
    '10m_v_component_of_wind': '10m_v_component_of_wind',
    'total_precipitation': 'total_precipitation',
    'surface_solar_radiation_downwards': 'surface_solar_radiation_downwards',
}


# =============================================================================
# TÉLÉCHARGEMENT
# =============================================================================

class ERA5Downloader:
    """Téléchargeur de données ERA5-Land pour l'Afrique."""

    def __init__(self, output_dir: str = 'era5_africa_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            self.client = cdsapi.Client()
            print("✅ Connexion CDS API établie")
        except Exception as e:
            print(f"❌ Erreur connexion CDS API : {e}")
            print("   Vérifiez votre fichier ~/.cdsapirc")
            self.client = None

    def download_zone(
        self,
        zone_name: str,
        year: int,
        months: List[str] = None,
        variables: List[str] = None,
    ) -> str:
        """
        Télécharge les données ERA5-Land pour une zone et une année.

        Args:
            zone_name: Nom de la zone (clé de AFRICA_FIRE_ZONES)
            year: Année (ex: 2020)
            months: Liste des mois ['01', '02', ...] ou None pour toute l'année
            variables: Liste des variables ou None pour toutes

        Returns:
            Chemin du fichier téléchargé
        """
        if self.client is None:
            raise RuntimeError("Client CDS non initialisé")

        if zone_name not in AFRICA_FIRE_ZONES:
            raise ValueError(f"Zone inconnue : {zone_name}. Zones disponibles : {list(AFRICA_FIRE_ZONES.keys())}")

        zone = AFRICA_FIRE_ZONES[zone_name]
        bbox = zone['bbox']

        if months is None:
            months = [f"{m:02d}" for m in range(1, 13)]

        if variables is None:
            variables = list(ERA5_VARIABLES.keys())

        output_file = os.path.join(
            self.output_dir,
            f"era5_land_{zone_name}_{year}.nc"
        )

        print(f"\n📥 Téléchargement ERA5-Land :")
        print(f"   Zone : {zone_name} — {zone['description']}")
        print(f"   Année : {year}")
        print(f"   Mois : {months}")
        print(f"   Variables : {variables}")
        print(f"   Bbox : {bbox}")
        print(f"   Fichier : {output_file}")

        request = {
            'format': 'netcdf',
            'variable': variables,
            'year': str(year),
            'month': months,
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(0, 24, 6)],  # 6h interval
            'area': bbox,
        }

        try:
            self.client.retrieve(
                'reanalysis-era5-land',
                request,
                output_file
            )
            print(f"✅ Téléchargement terminé : {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Erreur téléchargement : {e}")
            raise

    def download_multiple(
        self,
        zones: List[str],
        years: List[int],
    ) -> List[str]:
        """Télécharge plusieurs zones et années."""
        downloaded = []
        for zone in zones:
            for year in years:
                try:
                    path = self.download_zone(zone, year)
                    downloaded.append(path)
                except Exception as e:
                    print(f"⚠️ Erreur {zone}/{year} : {e}")
        return downloaded

    def load_and_inspect(self, filepath: str) -> xr.Dataset:
        """Charge et inspecte un fichier NetCDF."""
        print(f"\n🔍 Inspection : {filepath}")
        ds = xr.open_dataset(filepath)
        print(f"   Dimensions : {dict(ds.dims)}")
        print(f"   Variables : {list(ds.data_vars)}")
        print(f"   Période : {ds.time.min().values} → {ds.time.max().values}")
        return ds


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargeur ERA5-Land Afrique")
    parser.add_argument("--zone", type=str, default="sahel", 
                        help=f"Zone à télécharger. Disponibles : {list(AFRICA_FIRE_ZONES.keys())}")
    parser.add_argument("--year", type=int, default=2020, help="Année")
    parser.add_argument("--output-dir", type=str, default="era5_africa_data", help="Dossier de sortie")
    parser.add_argument("--list-zones", action="store_true", help="Lister les zones disponibles")

    args = parser.parse_args()

    if args.list_zones:
        print("Zones disponibles :")
        for name, info in AFRICA_FIRE_ZONES.items():
            print(f"  {name:20s} — {info['description']}")
            print(f"                       bbox : {info['bbox']}")
        exit(0)

    downloader = ERA5Downloader(args.output_dir)

    try:
        filepath = downloader.download_zone(args.zone, args.year)
        downloader.load_and_inspect(filepath)
    except Exception as e:
        print(f"❌ Erreur : {e}")
