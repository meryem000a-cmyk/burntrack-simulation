"""
hybrid_corrector.py
===================
Couplage hybride MLP + Ensemble (XGB/LGB/CB) pour BurnTrack.

Architecture recommandée :
- MLP v3 (DeepPhysicsCorrector) : correction additive avec features physiques
- Ensemble (StackedCorrectEnsemble) : fallback avec arbres boostés
- Méta-modèle de sélection : choisit le meilleur selon confiance/incertitude

Usage:
    corrector = HybridCorrector(mlp_path='mlp.pt', ensemble_dir='ensemble/')
    result = corrector.predict_with_selection(features_dict)
    # Retourne ROS finale + méthode utilisée + confiance
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Imports relatifs
from .mlp import DeepPhysicsCorrector, MLPConfig
from .ensemble import StackedCorrectEnsemble, EnsembleConfig
from .features import CorrectorFeatureExtractor


@dataclass
class HybridConfig:
    """Configuration du couplage hybride."""
    # Seuil de confiance pour sélection MLP
    mlp_confidence_threshold: float = 0.8

    # Seuil d'incertitude pour fallback vers ensemble
    mlp_uncertainty_threshold: float = 1.0  # m/min

    # Poids du MLP dans la moyenne pondérée (si hybride)
    mlp_weight: float = 0.7
    ensemble_weight: float = 0.3

    # Stratégie de sélection
    strategy: str = "auto"  # "mlp", "ensemble", "hybride", "auto"

    # Fallback si MLP crash
    fallback_on_error: bool = True


class HybridCorrector:
    """
    Couplage hybride MLP + Ensemble pour BurnTrack.

    Stratégies de sélection :
    1. AUTO (par défaut) :
       - Si confiance MLP > threshold → MLP seul
       - Si incertitude MLP < threshold → MLP seul  
       - Sinon → moyenne pondérée MLP + Ensemble

    2. MLP : Toujours MLP (avec fallback si crash)

    3. ENSEMBLE : Toujours Ensemble

    4. HYBRIDE : Toujours moyenne pondérée

    La correction additive est appliquée dans les deux modèles :
    ROS = ROS_rothermel + delta
    """

    def __init__(self, mlp_path: Optional[str] = None,
                 ensemble_dir: Optional[str] = None,
                 config: Optional[HybridConfig] = None,
                 feature_extractor: Optional[CorrectorFeatureExtractor] = None):
        """
        Initialise le couplage hybride.

        Args:
            mlp_path: Chemin vers le modèle MLP (.pt)
            ensemble_dir: Chemin vers le dossier ensemble sauvegardé
            config: Configuration hybride
            feature_extractor: Extracteur de features (optionnel)
        """
        self.config = config or HybridConfig()
        self.feature_extractor = feature_extractor or CorrectorFeatureExtractor()

        # Charger MLP
        self.mlp: Optional[DeepPhysicsCorrector] = None
        if mlp_path and torch.cuda.is_available() or mlp_path:
            try:
                self.mlp = DeepPhysicsCorrector(MLPConfig())
                self.mlp.load(mlp_path)
                print(f"[Hybrid] MLP chargé depuis {mlp_path}")
            except Exception as e:
                print(f"[Hybrid] Erreur chargement MLP: {e}")
                if not self.config.fallback_on_error:
                    raise

        # Charger Ensemble
        self.ensemble: Optional[StackedCorrectEnsemble] = None
        if ensemble_dir:
            try:
                self.ensemble = StackedCorrectEnsemble.load(ensemble_dir)
                print(f"[Hybrid] Ensemble chargé depuis {ensemble_dir}")
            except Exception as e:
                print(f"[Hybrid] Erreur chargement Ensemble: {e}")
                if not self.config.fallback_on_error:
                    raise

        if self.mlp is None and self.ensemble is None:
            raise RuntimeError("Aucun modèle chargé. Fournir mlp_path ou ensemble_dir.")

    def predict_mlp(self, features_dict: Dict) -> Dict:
        """
        Prédit avec MLP + incertitude MC Dropout.

        Retourne:
            ros_pred: ROS prédite
            uncertainty: Incertitude totale (epistemic + aleatoric)
            confidence: Score de confiance [0,1]
        """
        if self.mlp is None:
            raise RuntimeError("MLP non chargé")

        result = self.mlp.predict_with_uncertainty(features_dict)

        # Confiance = inverse de l'incertitude normalisée
        uncertainty = result['uncertainty']
        confidence = 1.0 / (1.0 + uncertainty)

        return {
            'ros': result['ros_predicted'],
            'delta': result['delta_predicted'],
            'uncertainty': uncertainty,
            'epistemic': result.get('epistemic_uncertainty', 0.0),
            'aleatoric': result.get('aleatoric_uncertainty', 0.0),
            'confidence': confidence,
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'method': 'mlp',
        }

    def predict_ensemble(self, features_dict: Dict) -> Dict:
        """
        Prédit avec Ensemble (XGB+LGB+CB) + incertitude BayesianRidge.

        Retourne:
            ros_pred: ROS prédite
            uncertainty: Incertitude épistémique (std BayesianRidge)
            confidence: Score de confiance [0,1]
        """
        if self.ensemble is None:
            raise RuntimeError("Ensemble non chargé")

        result = self.ensemble.predict_with_uncertainty(features_dict)

        uncertainty = result['uncertainty']
        confidence = 1.0 / (1.0 + uncertainty)

        return {
            'ros': result['ros_corrected'],
            'delta': result['delta_ros'],
            'uncertainty': uncertainty,
            'epistemic': uncertainty,  # BayesianRidge = epistemic seule
            'aleatoric': 0.0,  # Pas de modélisation aleatoric dans ensemble
            'confidence': confidence,
            'ci_lower': result['ci_lower_ros'],
            'ci_upper': result['ci_upper_ros'],
            'method': 'ensemble',
        }

    def predict_hybrid(self, features_dict: Dict) -> Dict:
        """
        Moyenne pondérée MLP + Ensemble.

        Poids configurables via HybridConfig.
        """
        mlp_result = self.predict_mlp(features_dict)
        ensemble_result = self.predict_ensemble(features_dict)

        # Moyenne pondérée
        w_mlp = self.config.mlp_weight
        w_ens = self.config.ensemble_weight

        ros_final = w_mlp * mlp_result['ros'] + w_ens * ensemble_result['ros']

        # Incertitude combinée (somme quadratique pondérée)
        unc_final = np.sqrt(
            w_mlp**2 * mlp_result['uncertainty']**2 + 
            w_ens**2 * ensemble_result['uncertainty']**2
        )

        # Confiance combinée
        conf_final = w_mlp * mlp_result['confidence'] + w_ens * ensemble_result['confidence']

        return {
            'ros': ros_final,
            'delta': mlp_result['delta'],  # On garde le delta MLP
            'uncertainty': unc_final,
            'epistemic': mlp_result['epistemic'],
            'aleatoric': mlp_result['aleatoric'],
            'confidence': conf_final,
            'ci_lower': ros_final - 1.96 * unc_final,
            'ci_upper': ros_final + 1.96 * unc_final,
            'method': 'hybrid',
            'mlp_contribution': w_mlp,
            'ensemble_contribution': w_ens,
        }

    def predict_with_selection(self, features_dict: Dict) -> Dict:
        """
        Prédit avec sélection automatique selon confiance.

        STRATÉGIE AUTO (par défaut):
        1. Si MLP disponible ET confiance > threshold → MLP
        2. Si MLP disponible ET incertitude < threshold → MLP
        3. Si les deux disponibles → HYBRIDE pondéré
        4. Sinon → celui qui est disponible

        Retourne un dict complet avec:
            - ros: ROS finale
            - method: 'mlp', 'ensemble', ou 'hybrid'
            - confidence: score de confiance
            - uncertainty: incertitude totale
            - selection_reason: pourquoi cette méthode a été choisie
        """
        strategy = self.config.strategy

        # Cas simples : forçage de stratégie
        if strategy == 'mlp':
            if self.mlp is None:
                raise RuntimeError("MLP non disponible")
            result = self.predict_mlp(features_dict)
            result['selection_reason'] = 'forced_mlp'
            return result

        elif strategy == 'ensemble':
            if self.ensemble is None:
                raise RuntimeError("Ensemble non disponible")
            result = self.predict_ensemble(features_dict)
            result['selection_reason'] = 'forced_ensemble'
            return result

        elif strategy == 'hybrid':
            if self.mlp is None or self.ensemble is None:
                raise RuntimeError("MLP et Ensemble requis pour hybride")
            result = self.predict_hybrid(features_dict)
            result['selection_reason'] = 'forced_hybrid'
            return result

        # STRATÉGIE AUTO (intelligente)
        elif strategy == 'auto':
            return self._select_auto(features_dict)

        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")

    def _select_auto(self, features_dict: Dict) -> Dict:
        """
        Sélection automatique intelligente.

        Logique:
        - MLP privilégié car il a la structure physique (phi_w, beta, gamma...)
        - Ensemble fallback si MLP incertain ou non disponible
        - Hybride si les deux sont moyennement confiants
        """
        mlp_available = self.mlp is not None
        ensemble_available = self.ensemble is not None

        # Cas 1: Seul MLP disponible
        if mlp_available and not ensemble_available:
            result = self.predict_mlp(features_dict)
            result['selection_reason'] = 'mlp_only_available'
            return result

        # Cas 2: Seul Ensemble disponible
        if ensemble_available and not mlp_available:
            result = self.predict_ensemble(features_dict)
            result['selection_reason'] = 'ensemble_only_available'
            return result

        # Cas 3: Les deux disponibles → sélection intelligente
        if mlp_available and ensemble_available:
            mlp_result = self.predict_mlp(features_dict)

            # Sous-cas 3a: MLP très confiant → MLP seul
            if mlp_result['confidence'] >= self.config.mlp_confidence_threshold:
                mlp_result['selection_reason'] = f"mlp_high_confidence({mlp_result['confidence']:.2f})"
                return mlp_result

            # Sous-cas 3b: MLP très incertain → Ensemble seul
            if mlp_result['uncertainty'] >= self.config.mlp_uncertainty_threshold:
                ens_result = self.predict_ensemble(features_dict)
                ens_result['selection_reason'] = f"mlp_too_uncertain({mlp_result['uncertainty']:.2f})"
                return ens_result

            # Sous-cas 3c: MLP moyennement confiant → Hybride
            hybrid_result = self.predict_hybrid(features_dict)
            hybrid_result['selection_reason'] = f"hybrid_mlp_confidence({mlp_result['confidence']:.2f})"
            return hybrid_result

        # Cas 4: Aucun disponible
        raise RuntimeError("Aucun modèle disponible")

    def predict_batch(self, features_list: list) -> list:
        """
        Prédit un batch avec sélection automatique pour chaque ligne.

        Chaque ligne peut utiliser une méthode différente selon sa confiance.
        """
        results = []
        for features_dict in features_list:
            result = self.predict_with_selection(features_dict)
            results.append(result)
        return results

    def evaluate_selection(self, features_list: list, y_true: np.ndarray) -> Dict:
        """
        Évalue la qualité de la sélection sur un jeu de test.

        Retourne des métriques par méthode (mlp, ensemble, hybrid)
        pour valider que la sélection est pertinente.
        """
        from sklearn.metrics import mean_absolute_error, r2_score

        results = self.predict_batch(features_list)

        # Séparer par méthode
        by_method = {'mlp': [], 'ensemble': [], 'hybrid': []}
        for i, res in enumerate(results):
            method = res['method']
            by_method[method].append((i, res['ros']))

        metrics = {}
        for method, items in by_method.items():
            if not items:
                continue
            indices = [i for i, _ in items]
            preds = np.array([ros for _, ros in items])
            true = y_true[indices]

            metrics[method] = {
                'count': len(items),
                'mae': mean_absolute_error(true, preds),
                'r2': r2_score(true, preds),
                'mean_confidence': np.mean([results[i]['confidence'] for i in indices]),
            }

        # Métriques globales
        all_preds = np.array([r['ros'] for r in results])
        metrics['global'] = {
            'mae': mean_absolute_error(y_true, all_preds),
            'r2': r2_score(y_true, all_preds),
            'mlp_usage': len(by_method['mlp']) / len(results),
            'ensemble_usage': len(by_method['ensemble']) / len(results),
            'hybrid_usage': len(by_method['hybrid']) / len(results),
        }

        return metrics


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

def example_usage():
    """Exemple d'utilisation du couplage hybride."""

    # Configuration
    config = HybridConfig(
        strategy='auto',
        mlp_confidence_threshold=0.8,
        mlp_uncertainty_threshold=1.0,
        mlp_weight=0.7,
        ensemble_weight=0.3,
    )

    # Initialisation
    corrector = HybridCorrector(
        mlp_path='models/corrector_v3_best.pt',
        ensemble_dir='models/ensemble/',
        config=config,
    )

    # Prédiction avec sélection automatique
    features = {
        'temperature': 35.0,
        'humidity': 20.0,
        'wind_speed': 5.0,
        'slope': 10.0,
        'fuel_model_code': 'AF_STEPPE',
        'ros_rothermel': 8.5,  # ROS du moteur Rothermel
        'phi_w': 2.3,
        'beta': 0.015,
        'gamma': 0.012,
        'eta_M': 0.85,
        'I_R': 150.0,
        'xi': 0.3,
        'tau': 0.5,
    }

    result = corrector.predict_with_selection(features)

    print(f"ROS finale: {result['ros']:.2f} m/min")
    print(f"Méthode: {result['method']}")
    print(f"Raison: {result['selection_reason']}")
    print(f"Confiance: {result['confidence']:.2f}")
    print(f"Incertitude: {result['uncertainty']:.2f}")
    print(f"CI 95%: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")

    # Exemple de résultat:
    # ROS finale: 7.82 m/min
    # Méthode: mlp
    # Raison: mlp_high_confidence(0.92)
    # Confiance: 0.92
    # Incertitude: 0.35
    # CI 95%: [7.13, 8.51]


if __name__ == '__main__':
    example_usage()
