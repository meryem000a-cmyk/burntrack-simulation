from abc import ABC, abstractmethod
from typing import Dict


class BaseCorrector(ABC):
    @abstractmethod
    def predict(self, features: Dict) -> Dict[str, float]:
        """Return {'delta_ros': float, 'ros_corrected': float, ...}"""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, features: Dict) -> Dict:
        """Return dict with 'delta_ros', 'ros_corrected', 'uncertainty'"""
        pass
