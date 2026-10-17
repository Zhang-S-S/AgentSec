"""Missing-aware multi-source MIL account classifier.

This package is deliberately separate from :mod:`account_classifier` so the
published Mean/Max/Attention baseline and its checkpoints remain unchanged.
"""

from .config import ModelConfig, TrainConfig
from .model import MultiSourceRiskModel

__all__ = ["ModelConfig", "TrainConfig", "MultiSourceRiskModel"]
