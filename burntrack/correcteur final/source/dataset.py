"""
Préparation des données BurnTrack
- Target encoding des fuels
- Merge réel / synthétique
- Normalisation StandardScaler
- Split train/val/test stratifié
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import json
import pickle
from typing import Tuple, List, Dict


class BurnTrackDataset:
    """
    Prépare les données pour l'entraînement.

    Pipeline:
        1. Analyse des biais par fuel (données réelles)
        2. Target encoding: fuel -> biais moyen
        3. Merge réel + synthétique
        4. Split stratifié
        5. Normalisation StandardScaler
    """

    def __init__(self, real_path: str, synth_path: str):
        self.real_path = real_path
        self.synth_path = synth_path
        self.fuel_encoding: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.ros_measured_col = None
        self.ros_r_col = None

        self.df_real = None
        self.df_synth = None
        self.df_merged = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.is_real_train = None
        self.is_real_val = None
        self.is_real_test = None

    def _detect_columns(self, df: pd.DataFrame):
        """Détecte automatiquement les colonnes ROS et conditions."""
        for col in ['ros_measured', 'ROS']:
            if col in df.columns:
                self.ros_measured_col = col
                break
        for col in ['ros_rothermel', 'ROS_r']:
            if col in df.columns:
                self.ros_r_col = col
                break

        print(f"  ROS mesurée: {self.ros_measured_col}")
        print(f"  ROS Rothermel: {self.ros_r_col}")

    def analyze_bias(self) -> pd.DataFrame:
        """Analyse les biais de Rothermel par fuel model."""
        print("\n=== ANALYSE DES BIAIS PAR FUEL ===")
        self.df_real = pd.read_csv(self.real_path)
        self._detect_columns(self.df_real)

        self.df_real['delta_ros'] = self.df_real[self.ros_measured_col] - self.df_real[self.ros_r_col]

        bias_analysis = []
        for fuel in sorted(self.df_real['fuel_model'].unique()):
            subset = self.df_real[self.df_real['fuel_model'] == fuel]['delta_ros']
            t_stat, p_value = stats.ttest_1samp(subset, 0)

            bias_analysis.append({
                'fuel_model': fuel,
                'n': len(subset),
                'mean_bias': subset.mean(),
                'std_bias': subset.std(),
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        bias_df = pd.DataFrame(bias_analysis).sort_values('mean_bias')

        print(f"\n{len(bias_df)} fuels analysés:")
        print(bias_df.to_string(index=False, float_format='%.3f'))

        herbes = bias_df[bias_df['mean_bias'] < -3]['fuel_model'].tolist()
        boises = bias_df[bias_df['mean_bias'] > 1]['fuel_model'].tolist()
        print(f"\nHerbes (sur-estimés): {herbes}")
        print(f"Boisés (sous-estimés): {boises}")

        return bias_df

    def compute_target_encoding(self, bias_df: pd.DataFrame) -> Dict[str, float]:
        """Calcule le target encoding: fuel -> biais moyen."""
        self.fuel_encoding = bias_df.set_index('fuel_model')['mean_bias'].to_dict()

        print(f"\n=== TARGET ENCODING ===")
        for fuel, bias in sorted(self.fuel_encoding.items(), key=lambda x: x[1]):
            print(f"  {fuel:25s} -> {bias:+.3f}")

        with open('fuel_encoding.json', 'w') as f:
            json.dump(self.fuel_encoding, f, indent=2)

        return self.fuel_encoding

    def prepare(self, feature_candidates: List[str] = None) -> Tuple:
        """Prépare les données complètes: merge, split, normalisation."""
        print("\n=== PRÉPARATION DES DONNÉES ===")

        self.df_synth = pd.read_csv(self.synth_path)

        self.df_real['fuel_encoded'] = self.df_real['fuel_model'].map(self.fuel_encoding)
        self.df_real['source'] = 'real'
        self.df_synth['fuel_encoded'] = self.df_synth['fuel_model'].map(self.fuel_encoding)
        self.df_synth['source'] = 'synthetic'

        unknown = self.df_synth[self.df_synth['fuel_encoded'].isna()]['fuel_model'].unique()
        if len(unknown) > 0:
            print(f"⚠️ Fuels inconnus dans synthétique: {unknown}")
            global_mean = self.df_real['delta_ros'].mean()
            self.df_synth['fuel_encoded'] = self.df_synth['fuel_encoded'].fillna(global_mean)

        common_cols = [c for c in self.df_real.columns if c in self.df_synth.columns]
        self.df_merged = pd.concat([self.df_real[common_cols], self.df_synth[common_cols]], ignore_index=True)
        print(f"\nDataset fusionné: {len(self.df_merged)} éch. (réels: {len(self.df_real)}, synth: {len(self.df_synth)})")

        if feature_candidates is None:
            feature_candidates = ['fuel_encoded', 'wind_speed', 'humidity', 'slope', 'ros_rothermel']

        self.feature_cols = [c for c in feature_candidates if c in self.df_merged.columns]
        if 'slope' not in self.feature_cols:
            for alt in ['slope_deg', 'slope_pct']:
                if alt in self.df_merged.columns:
                    self.feature_cols.append(alt)
                    break
        if 'humidity' not in self.feature_cols and 'rh_percent' in self.df_merged.columns:
            self.feature_cols.append('rh_percent')

        print(f"\nFeatures: {self.feature_cols}")

        self.df_merged['target'] = self.df_merged[self.ros_measured_col] - self.df_merged[self.ros_r_col]

        df_real_m = self.df_merged[self.df_merged['source'] == 'real']
        df_synth_m = self.df_merged[self.df_merged['source'] == 'synthetic']

        real_train, real_temp = train_test_split(df_real_m, test_size=0.3, random_state=42)
        real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)

        synth_train, synth_temp = train_test_split(df_synth_m, test_size=0.2, random_state=42)
        synth_val, synth_test = train_test_split(synth_temp, test_size=0.5, random_state=42)

        df_train = pd.concat([real_train, synth_train])
        df_val = pd.concat([real_val, synth_val])
        df_test = pd.concat([real_test, synth_test])

        print(f"\nSplit:")
        print(f"  Train: {len(df_train)} (réels: {len(real_train)}, synth: {len(synth_train)})")
        print(f"  Val:   {len(df_val)} (réels: {len(real_val)}, synth: {len(synth_val)})")
        print(f"  Test:  {len(df_test)} (réels: {len(real_test)}, synth: {len(synth_test)})")

        self.X_train = self.scaler.fit_transform(df_train[self.feature_cols])
        self.X_val = self.scaler.transform(df_val[self.feature_cols])
        self.X_test = self.scaler.transform(df_test[self.feature_cols])

        self.y_train = df_train['target'].values
        self.y_val = df_val['target'].values
        self.y_test = df_test['target'].values

        self.is_real_train = (df_train['source'] == 'real').values.astype(np.float32)
        self.is_real_val = (df_val['source'] == 'real').values.astype(np.float32)
        self.is_real_test = (df_test['source'] == 'real').values.astype(np.float32)

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\n✅ Données prêtes. Shape features: {self.X_train.shape}")

        return (self.X_train, self.X_val, self.X_test, 
                self.y_train, self.y_val, self.y_test,
                self.is_real_train, self.is_real_val, self.is_real_test)
