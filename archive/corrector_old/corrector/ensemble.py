"""
ensemble.py
===========
Stacked ensemble of XGBoost, LightGBM, and CatBoost base learners
with a Bayesian Ridge meta-learner for delta_ros correction.
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from sklearn.linear_model import BayesianRidge

from burntrack.corrector.base import BaseCorrector
from burntrack.corrector.features import CorrectorFeatureExtractor

logger = logging.getLogger(__name__)


class BaseTreeLearner(ABC):
    """Abstract base class for tree-based base learners."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
        
    @abstractmethod
    def save(self, path: str):
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseTreeLearner':
        pass


class XGBoostBaseLearner(BaseTreeLearner):
    """XGBoost base learner with quantile regression for uncertainty."""
    
    def __init__(self, n_estimators=1000, max_depth=10, learning_rate=0.03, **kwargs):
        if xgb is None:
            raise ImportError("XGBoost is not installed. Please install it to use XGBoostBaseLearner.")
            
        self.mean_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            n_jobs=-1,
            **kwargs
        )
        self.lower_model = xgb.XGBRegressor(
            n_estimators=n_estimators // 2,
            max_depth=max_depth - 2,
            learning_rate=learning_rate,
            objective='reg:quantileerror',
            quantile_alpha=0.05,
            early_stopping_rounds=50,
            n_jobs=-1,
            **kwargs
        )
        self.upper_model = xgb.XGBRegressor(
            n_estimators=n_estimators // 2,
            max_depth=max_depth - 2,
            learning_rate=learning_rate,
            objective='reg:quantileerror',
            quantile_alpha=0.95,
            early_stopping_rounds=50,
            n_jobs=-1,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        logger.info("Training XGBoost mean model...")
        self.mean_model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        logger.info("Training XGBoost lower quantile model...")
        self.lower_model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        logger.info("Training XGBoost upper quantile model...")
        self.upper_model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mean_model.predict(X)
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        mean_pred = self.mean_model.predict(X)
        lower_pred = self.lower_model.predict(X)
        upper_pred = self.upper_model.predict(X)
        
        return {
            'delta_ros': mean_pred,
            'ci_lower': lower_pred,
            'ci_upper': upper_pred,
            'uncertainty': (upper_pred - lower_pred) / (2 * 1.96)  # Approx standard dev
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.mean_model.save_model(os.path.join(path, "xgb_mean.json"))
        self.lower_model.save_model(os.path.join(path, "xgb_lower.json"))
        self.upper_model.save_model(os.path.join(path, "xgb_upper.json"))

    @classmethod
    def load(cls, path: str) -> 'XGBoostBaseLearner':
        instance = cls()
        instance.mean_model.load_model(os.path.join(path, "xgb_mean.json"))
        instance.lower_model.load_model(os.path.join(path, "xgb_lower.json"))
        instance.upper_model.load_model(os.path.join(path, "xgb_upper.json"))
        return instance


class LightGBMBaseLearner(BaseTreeLearner):
    """LightGBM base learner."""
    
    def __init__(self, n_estimators=1000, max_depth=12, learning_rate=0.03, **kwargs):
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it to use LightGBMBaseLearner.")
            
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = [lgb.early_stopping(stopping_rounds=50)] if eval_set else None
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.booster_.save_model(os.path.join(path, "lgb_model.txt"))

    @classmethod
    def load(cls, path: str) -> 'LightGBMBaseLearner':
        instance = cls()
        instance.model = lgb.Booster(model_file=os.path.join(path, "lgb_model.txt"))
        # Patch predict to work like Sklearn
        _original_predict = instance.model.predict
        instance.predict = lambda X: _original_predict(X)
        return instance


class CatBoostBaseLearner(BaseTreeLearner):
    """CatBoost base learner."""
    
    def __init__(self, iterations=800, depth=8, learning_rate=0.05, **kwargs):
        if cb is None:
            raise ImportError("CatBoost is not installed. Please install it to use CatBoostBaseLearner.")
            
        self.model = cb.CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=False,
            thread_count=-1,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, "cb_model.cbm"))

    @classmethod
    def load(cls, path: str) -> 'CatBoostBaseLearner':
        instance = cls()
        instance.model.load_model(os.path.join(path, "cb_model.cbm"))
        return instance


class StackedCorrectEnsemble(BaseCorrector):
    """Stacking meta-learner ensemble for fire behavior correction."""
    
    def __init__(self, use_xgb=True, use_lgb=True, use_cb=True):
        self.feature_extractor = CorrectorFeatureExtractor()
        
        self.base_learners = {}
        if use_xgb and xgb is not None:
            self.base_learners['xgb'] = XGBoostBaseLearner()
        if use_lgb and lgb is not None:
            self.base_learners['lgb'] = LightGBMBaseLearner()
        if use_cb and cb is not None:
            self.base_learners['cb'] = CatBoostBaseLearner()
            
        if not self.base_learners:
            raise ValueError("No base learners available. Please install xgboost, lightgbm, or catboost.")
            
        self.meta_learner = BayesianRidge(compute_score=True)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit the ensemble in two stages: base learners, then meta-learner."""
        
        # Stage 1: Fit base learners
        logger.info(f"Training {len(self.base_learners)} base learners...")
        for name, learner in self.base_learners.items():
            logger.info(f"Training {name}...")
            learner.fit(X_train, y_train, X_val, y_val)
            
        # Stage 2: Create meta-features (predictions on validation set)
        logger.info("Training meta-learner...")
        
        # We need out-of-fold predictions or just use validation set to train meta-learner
        # If no validation set is provided, we reuse training set (suboptimal but works as fallback)
        X_meta_train = X_val if X_val is not None else X_train
        y_meta_train = y_val if y_val is not None else y_train
        
        meta_features = self.get_base_predictions(X_meta_train)
        
        # Fit Bayesian Ridge meta-learner
        self.meta_learner.fit(meta_features, y_meta_train)
        self.is_fitted = True
        logger.info(f"Meta-learner weights: {self.meta_learner.coef_}")

    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base learners. Returns [n_samples, n_learners]."""
        preds = []
        for name, learner in self.base_learners.items():
            preds.append(learner.predict(X))
        return np.column_stack(preds)

    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict delta_ros and return dictionary for single observation."""
        res = self.predict_with_uncertainty(features)
        return {
            'delta_ros': res['delta_ros'],
            'uncertainty': res['uncertainty']
        }

    def predict_with_uncertainty(self, features: Dict) -> Dict:
        """Predict delta_ros with uncertainty bounds."""
        if not self.is_fitted:
            raise ValueError("Ensemble is not fitted yet.")
            
        # Extract features
        x_vec = self.feature_extractor.extract_row(features)
        feature_names = self.feature_extractor.get_feature_names()
        x_arr = np.array([[x_vec.get(name, 0.0) for name in feature_names]])
        
        # Get base predictions
        meta_features = self.get_base_predictions(x_arr)
        
        # Meta-learner prediction with uncertainty
        delta_ros, std = self.meta_learner.predict(meta_features, return_std=True)
        delta_ros = float(delta_ros[0])
        std = float(std[0])
        
        # Incorporate epistemic uncertainty from XGBoost if available
        xgb_unc = 0.0
        if 'xgb' in self.base_learners:
            xgb_preds = self.base_learners['xgb'].predict_with_uncertainty(x_arr)
            xgb_unc = float(xgb_preds['uncertainty'][0])
            
        # Combined uncertainty (aleatoric from meta-learner, epistemic from XGB)
        total_unc = np.sqrt(std**2 + xgb_unc**2)
        
        return {
            'delta_ros': delta_ros,
            'uncertainty': total_unc,
            'ci_lower': delta_ros - 1.96 * total_unc,
            'ci_upper': delta_ros + 1.96 * total_unc,
            'base_predictions': {name: float(meta_features[0, i]) for i, name in enumerate(self.base_learners.keys())}
        }

    def save(self, output_dir: str):
        """Save all models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base learners
        for name, learner in self.base_learners.items():
            learner_dir = os.path.join(output_dir, f"base_{name}")
            learner.save(learner_dir)
            
        # Save meta-learner
        meta_state = {
            'coef_': self.meta_learner.coef_.tolist(),
            'intercept_': float(self.meta_learner.intercept_),
            'alpha_': float(self.meta_learner.alpha_),
            'lambda_': float(self.meta_learner.lambda_),
            'sigma_': self.meta_learner.sigma_.tolist() if hasattr(self.meta_learner, 'sigma_') else None,
            'base_learners_keys': list(self.base_learners.keys())
        }
        with open(os.path.join(output_dir, "meta_learner.json"), "w") as f:
            json.dump(meta_state, f)

    @classmethod
    def load(cls, output_dir: str) -> 'StackedCorrectEnsemble':
        """Load ensemble from disk."""
        # Need to know which base learners exist
        with open(os.path.join(output_dir, "meta_learner.json"), "r") as f:
            meta_state = json.load(f)
            
        learner_keys = meta_state.get('base_learners_keys', [])
        
        instance = cls(
            use_xgb='xgb' in learner_keys,
            use_lgb='lgb' in learner_keys,
            use_cb='cb' in learner_keys
        )
        
        # Load base learners
        for name in learner_keys:
            learner_dir = os.path.join(output_dir, f"base_{name}")
            if name == 'xgb':
                instance.base_learners[name] = XGBoostBaseLearner.load(learner_dir)
            elif name == 'lgb':
                instance.base_learners[name] = LightGBMBaseLearner.load(learner_dir)
            elif name == 'cb':
                instance.base_learners[name] = CatBoostBaseLearner.load(learner_dir)
                
        # Restore meta-learner
        instance.meta_learner.coef_ = np.array(meta_state['coef_'])
        instance.meta_learner.intercept_ = meta_state['intercept_']
        instance.meta_learner.alpha_ = meta_state['alpha_']
        instance.meta_learner.lambda_ = meta_state['lambda_']
        if meta_state.get('sigma_'):
            instance.meta_learner.sigma_ = np.array(meta_state['sigma_'])
            
        instance.is_fitted = True
        return instance
