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

# Import statements removed to avoid circular import issues on Windows
# Users should import directly from submodules:
#   from pebs.data.loader import NSDUHLoader, SMNILoader
#   from pebs.features.eeg_extractor import EEGFeatureExtractor
#   from pebs.models.eri_model import ERIModel
#   from pebs.models.bvi_model import BVIModel
#   from pebs.models.risk_classifier import RiskClassifier
#   from pebs.utils.visualization import Visualizer
#   from pebs.utils.metrics import Metrics

__all__ = []
