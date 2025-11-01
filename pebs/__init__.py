"""
PEBS: Personalized Evidence-Based Screening System
==================================================

A machine learning system for alcoholism risk assessment using
environmental (survey) and biological (EEG) data.

Main modules:
- data: Data loading and preprocessing
- features: Feature extraction from EEG signals
- models: Machine learning models (ERI, BVI, Risk Classifier)
- utils: Visualization and metrics utilities
"""

__version__ = "1.0.0"
__author__ = "PEBS Development Team"

from . import data
from . import features
from . import models
from . import utils

__all__ = ["data", "features", "models", "utils"]
