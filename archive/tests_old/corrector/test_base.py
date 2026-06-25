"""Test BaseCorrector ABC interface."""
import pytest
from burntrack.corrector.base import BaseCorrector


def test_base_corrector_is_abstract():
    with pytest.raises(TypeError):
        BaseCorrector()


def test_concrete_implementation():
    class DummyCorrector(BaseCorrector):
        def predict(self, features):
            return {'delta_ros': 0.5, 'ros_corrected': 5.5}

        def predict_with_uncertainty(self, features):
            return {'delta_ros': 0.5, 'ros_corrected': 5.5, 'uncertainty': 0.1}

    c = DummyCorrector()
    result = c.predict({})
    assert 'delta_ros' in result
    assert 'ros_corrected' in result
