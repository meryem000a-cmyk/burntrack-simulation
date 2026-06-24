import pandas as pd
from burntrack.data.real_dataset import build_real_dataset
from burntrack.data.firms import preprocess_firms_data, build_fire_fronts

def main():
    print("1. Chargement des données FIRMS locales...")
    # On charge le CSV que vous avez uploadé
    df = pd.read_csv("data/firms_raw_data.csv")
    print(f"   -> {len(df)} points FIRMS bruts chargés.")
    
    print("2. Prétraitement et Clustering DBSCAN...")
    # Le prétraitement normalise les dates, les heures, et calcule le FRP
    df = preprocess_firms_data(df)
    # On reconstitue les fronts de feu (par région et temporalité)
    fronts = build_fire_fronts(df, eps_km=2.0, min_samples=3)
    print(f"   -> {len(fronts)} fronts de feu reconstitués.")
    
    print("3. Intégration Météo & Modèle de Rothermel (Pipeline complet)...")
    # Ce processus va enrichir chaque front avec les données météo Open-Meteo
    # et calculer les valeurs théoriques de Rothermel pour obtenir vos 53 colonnes.
    dataset = build_real_dataset(fronts)
    
    # Sauvegarde du fichier final attendu par le MLP
    output_path = "data/processed/african_ground_truth.csv"
    dataset.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCÈS ! Dataset final généré : {output_path}")
    print(f"   -> Lignes (Données) : {len(dataset)}")
    print(f"   -> Colonnes (Features) : {len(dataset.columns)}")

if __name__ == "__main__":
    main()
