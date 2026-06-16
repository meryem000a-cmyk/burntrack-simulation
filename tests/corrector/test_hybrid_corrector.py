
# test_hybrid_corrector.py
# Script de test et validation du couplage hybride

import numpy as np
from hybrid_corrector import HybridCorrector, HybridConfig

def test_hybrid_corrector():
    """Test complet du couplage hybride."""

    # Configuration
    config = HybridConfig(
        strategy='auto',
        mlp_confidence_threshold=0.8,
        mlp_uncertainty_threshold=1.0,
        mlp_weight=0.7,
        ensemble_weight=0.3,
    )

    # Initialisation (sans modèles pour le test)
    # En production: charger les vrais modèles
    corrector = HybridCorrector(
        mlp_path=None,  # 'models/corrector_v3_best.pt'
        ensemble_dir=None,  # 'models/ensemble/'
        config=config,
    )

    # Features de test
    features = {
        'temperature': 35.0,
        'humidity': 20.0,
        'wind_speed': 5.0,
        'slope': 10.0,
        'fuel_model_code': 'AF_STEPPE',
        'ros_rothermel': 8.5,
        'phi_w': 2.3,
        'beta': 0.015,
        'gamma': 0.012,
        'eta_M': 0.85,
        'I_R': 150.0,
        'xi': 0.3,
        'tau': 0.5,
    }

    # Test 1: Sélection automatique
    print("=== Test 1: Sélection automatique ===")
    try:
        result = corrector.predict_with_selection(features)
        print(f"ROS: {result['ros']:.2f}")
        print(f"Méthode: {result['method']}")
        print(f"Raison: {result['selection_reason']}")
        print(f"Confiance: {result['confidence']:.2f}")
        print(f"CI 95%: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    except Exception as e:
        print(f"Erreur (normal si modèles non chargés): {e}")

    # Test 2: Batch prediction
    print("\n=== Test 2: Batch prediction ===")
    features_list = [features for _ in range(10)]
    try:
        results = corrector.predict_batch(features_list)
        methods = [r['method'] for r in results]
        print(f"Méthodes utilisées: {set(methods)}")
        print(f"Confiance moyenne: {np.mean([r['confidence'] for r in results]):.2f}")
    except Exception as e:
        print(f"Erreur: {e}")

    # Test 3: Évaluation de la sélection
    print("\n=== Test 3: Évaluation sélection ===")
    # Simuler des données
    y_true = np.random.normal(8.0, 2.0, 100)
    features_list = [features for _ in range(100)]
    try:
        metrics = corrector.evaluate_selection(features_list, y_true)
        print(f"Métriques globales: {metrics['global']}")
    except Exception as e:
        print(f"Erreur: {e}")

    print("\n=== Tests terminés ===")

if __name__ == '__main__':
    test_hybrid_corrector()
