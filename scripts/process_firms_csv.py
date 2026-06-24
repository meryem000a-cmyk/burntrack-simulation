import os
import pandas as pd
from burntrack.data.firms import reconstruct_propagation
from burntrack.data.weather import fetch_weather_for_points
from burntrack.data.real_dataset import compute_rothermel_baseline

def main():
    print("1. Chargement des données FIRMS locales...")
    # On charge le CSV que vous avez uploadé
    df = pd.read_csv("data/firms_raw_data.csv")
    print(f"   -> {len(df)} points FIRMS bruts chargés.")
    
    print("2. Reconstitution Spatio-Temporelle (Clustering DBSCAN)...")
    # Reconstruit la propagation (vecteurs de vitesse et de direction)
    propagation_df = reconstruct_propagation(df)
    print(f"   -> {len(propagation_df)} vecteurs de propagation extraits.")
    
    if propagation_df.empty:
        print("Erreur : Aucun vecteur n'a pu être extrait. Vérifiez les données.")
        return
        
    print("3. Récupération Météo Historique (Open-Meteo)...")
    weather_df = fetch_weather_for_points(propagation_df)
    print(f"   -> Météo associée pour {len(weather_df)} points.")
    
    print("4. Intégration du Modèle de Rothermel...")
    # Calcule les valeurs théoriques de Rothermel et le Delta (Erreur)
    dataset_df = compute_rothermel_baseline(weather_df)
    
    # Création du dossier processed s'il n'existe pas
    output_path = "data/processed/african_ground_truth.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarde du fichier final attendu par le MLP
    dataset_df.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCÈS ! Dataset final généré : {output_path}")
    print(f"   -> Lignes (Données) : {len(dataset_df)}")
    print(f"   -> Colonnes (Features) : {len(dataset_df.columns)}")

if __name__ == "__main__":
    main()
