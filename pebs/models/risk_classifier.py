"""
Risk Classification Model.
Combines ERI and BVI scores to classify into 4 risk categories.
"""

import numpy as np
import pickle


class RiskClassifier:
    """
    Risk Classification Model.
    Combines Environmental Risk Index (ERI) and Biological Vulnerability Index (BVI)
    to classify individuals into 4 risk categories:
    0: Low Risk (low ERI + low BVI)
    1: Medium-Environmental (high ERI + low BVI)
    2: Medium-Biological (low ERI + high BVI)
    3: High Risk (high ERI + high BVI)
    """

    def __init__(self, eri_threshold=0.5, bvi_threshold=0.5):
        """
        Initialize Risk Classifier.

        Args:
            eri_threshold: Threshold for high environmental risk
            bvi_threshold: Threshold for high biological risk
        """
        self.eri_threshold = eri_threshold
        self.bvi_threshold = bvi_threshold

        self.category_names = {
            0: "Low Risk",
            1: "Medium-Environmental",
            2: "Medium-Biological",
            3: "High Risk"
        }

        self.category_descriptions = {
            0: "Low environmental and biological risk factors.",
            1: "High environmental risk but low biological vulnerability. Focus on environmental interventions.",
            2: "Low environmental risk but high biological vulnerability. Consider biological/genetic factors.",
            3: "Both environmental and biological risk factors are elevated. Comprehensive intervention recommended."
        }

    def classify(self, eri_scores, bvi_scores):
        """
        Classify samples into risk categories.

        Args:
            eri_scores: Environmental Risk Index scores (0-1)
            bvi_scores: Biological Vulnerability Index scores (0-1)

        Returns:
            Array of risk categories (0-3)
        """
        # Ensure inputs are arrays
        eri_scores = np.asarray(eri_scores)
        bvi_scores = np.asarray(bvi_scores)

        # Check dimensions
        if eri_scores.shape != bvi_scores.shape:
            raise ValueError("ERI and BVI scores must have the same shape")

        # Classify based on thresholds
        risk_categories = np.zeros(len(eri_scores), dtype=int)

        for i in range(len(eri_scores)):
            high_eri = eri_scores[i] >= self.eri_threshold
            high_bvi = bvi_scores[i] >= self.bvi_threshold

            if not high_eri and not high_bvi:
                risk_categories[i] = 0  # Low Risk
            elif high_eri and not high_bvi:
                risk_categories[i] = 1  # Medium-Environmental
            elif not high_eri and high_bvi:
                risk_categories[i] = 2  # Medium-Biological
            else:  # high_eri and high_bvi
                risk_categories[i] = 3  # High Risk

        return risk_categories

    def classify_single(self, eri_score, bvi_score):
        """
        Classify a single sample.

        Args:
            eri_score: Environmental Risk Index score (0-1)
            bvi_score: Biological Vulnerability Index score (0-1)

        Returns:
            Dictionary with category, name, and description
        """
        high_eri = eri_score >= self.eri_threshold
        high_bvi = bvi_score >= self.bvi_threshold

        if not high_eri and not high_bvi:
            category = 0
        elif high_eri and not high_bvi:
            category = 1
        elif not high_eri and high_bvi:
            category = 2
        else:
            category = 3

        return {
            'category': category,
            'name': self.category_names[category],
            'description': self.category_descriptions[category],
            'eri_score': float(eri_score),
            'bvi_score': float(bvi_score),
            'eri_high': high_eri,
            'bvi_high': high_bvi
        }

    def get_distribution(self, risk_categories, verbose=True):
        """
        Get distribution of risk categories.

        Args:
            risk_categories: Array of risk categories
            verbose: Print distribution

        Returns:
            Dictionary with counts and percentages
        """
        risk_categories = np.asarray(risk_categories)
        total = len(risk_categories)

        distribution = {}
        for i in range(4):
            count = np.sum(risk_categories == i)
            pct = count / total * 100 if total > 0 else 0
            distribution[i] = {
                'name': self.category_names[i],
                'count': int(count),
                'percentage': float(pct)
            }

        if verbose:
            print("\n📊 Risk Category Distribution:")
            for i, info in distribution.items():
                print(f"   {i}. {info['name']:25s}: {info['count']:4d} ({info['percentage']:5.1f}%)")

        return distribution

    def save(self, filepath):
        """
        Save classifier to file.

        Args:
            filepath: Path to save classifier
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Risk Classifier saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load classifier from file.

        Args:
            filepath: Path to classifier file

        Returns:
            Loaded RiskClassifier instance
        """
        with open(filepath, 'rb') as f:
            classifier = pickle.load(f)
        print(f"✅ Risk Classifier loaded from {filepath}")
        return classifier

    def get_category_name(self, category):
        """Get human-readable category name."""
        return self.category_names.get(category, "Unknown")

    def get_category_description(self, category):
        """Get category description."""
        return self.category_descriptions.get(category, "Unknown category")

    def set_thresholds(self, eri_threshold=None, bvi_threshold=None):
        """
        Update classification thresholds.

        Args:
            eri_threshold: New ERI threshold (optional)
            bvi_threshold: New BVI threshold (optional)
        """
        if eri_threshold is not None:
            self.eri_threshold = eri_threshold
        if bvi_threshold is not None:
            self.bvi_threshold = bvi_threshold

        print(f"✅ Thresholds updated: ERI={self.eri_threshold}, BVI={self.bvi_threshold}")
