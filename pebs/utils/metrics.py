"""
Metrics and evaluation utilities for PEBS system.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


class Metrics:
    """
    Metrics calculation and evaluation utilities.
    """

    @staticmethod
    def calculate_binary_metrics(y_true, y_pred, y_proba=None):
        """
        Calculate comprehensive binary classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # Add AUC if probabilities are provided
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = None

        return metrics

    @staticmethod
    def print_evaluation_report(name, y_true, y_pred, y_proba=None):
        """
        Print comprehensive evaluation report.

        Args:
            name: Model name (e.g., 'ERI Model', 'BVI Model')
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        """
        print("\n" + "="*80)
        print(f"{name} - EVALUATION REPORT")
        print("="*80)

        # Calculate metrics
        metrics = Metrics.calculate_binary_metrics(y_true, y_pred, y_proba)

        # Print metrics
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")

        if metrics.get('auc') is not None:
            print(f"   AUC-ROC:   {metrics['auc']:.4f}")

        # Confusion matrix
        print(f"\n📈 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
        print(f"    [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")

        # Classification report
        print(f"\n📝 Detailed Classification Report:")
        print(classification_report(y_true, y_pred))

        print("="*80)

    @staticmethod
    def calculate_risk_metrics(risk_categories):
        """
        Calculate metrics for risk classification.

        Args:
            risk_categories: Array of risk categories (0-3)

        Returns:
            Dictionary with distribution statistics
        """
        category_names = {
            0: "Low Risk",
            1: "Medium-Environmental",
            2: "Medium-Biological",
            3: "High Risk"
        }

        total = len(risk_categories)
        distribution = {}

        for i in range(4):
            count = np.sum(risk_categories == i)
            pct = count / total * 100 if total > 0 else 0

            distribution[i] = {
                'name': category_names[i],
                'count': int(count),
                'percentage': float(pct)
            }

        # Calculate risk concentration metrics
        high_risk_count = np.sum(np.isin(risk_categories, [2, 3]))
        high_risk_pct = high_risk_count / total * 100 if total > 0 else 0

        return {
            'distribution': distribution,
            'total_samples': total,
            'high_risk_count': int(high_risk_count),
            'high_risk_percentage': float(high_risk_pct)
        }

    @staticmethod
    def print_risk_report(risk_categories):
        """
        Print risk classification report.

        Args:
            risk_categories: Array of risk categories (0-3)
        """
        metrics = Metrics.calculate_risk_metrics(risk_categories)

        print("\n" + "="*80)
        print("RISK CLASSIFICATION REPORT")
        print("="*80)

        print(f"\nTotal Samples: {metrics['total_samples']:,}")
        print(f"\n📊 Risk Category Distribution:")

        for i, info in metrics['distribution'].items():
            print(f"   {i}. {info['name']:25s}: {info['count']:4d} ({info['percentage']:5.1f}%)")

        print(f"\n⚠️  High Risk Summary:")
        print(f"   Categories 2+3: {metrics['high_risk_count']:4d} ({metrics['high_risk_percentage']:5.1f}%)")

        print("="*80)

    @staticmethod
    def compare_models(model1_metrics, model2_metrics,
                       model1_name='Model 1', model2_name='Model 2'):
        """
        Compare two models side by side.

        Args:
            model1_metrics: Metrics dictionary from model 1
            model2_metrics: Metrics dictionary from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        print(f"\n{'Metric':<15} | {model1_name:>12} | {model2_name:>12} | Difference")
        print("-" * 60)

        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        for metric in metrics_to_compare:
            val1 = model1_metrics.get(metric)
            val2 = model2_metrics.get(metric)

            if val1 is not None and val2 is not None:
                diff = val2 - val1
                diff_str = f"{diff:+.4f}"
                print(f"{metric:<15} | {val1:>12.4f} | {val2:>12.4f} | {diff_str:>12}")

        print("="*80)

    @staticmethod
    def calculate_score_statistics(scores, name='Scores'):
        """
        Calculate statistics for a set of scores.

        Args:
            scores: Array of scores
            name: Name of scores (e.g., 'ERI Scores', 'BVI Scores')

        Returns:
            Dictionary of statistics
        """
        scores = np.asarray(scores)

        stats = {
            'name': name,
            'count': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        }

        return stats

    @staticmethod
    def print_score_statistics(scores, name='Scores'):
        """
        Print score statistics.

        Args:
            scores: Array of scores
            name: Name of scores
        """
        stats = Metrics.calculate_score_statistics(scores, name)

        print(f"\n📊 {stats['name']} Statistics:")
        print(f"   Count:  {stats['count']:,}")
        print(f"   Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"   Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   Median: {stats['median']:.4f}")
        print(f"   Q1-Q3:  [{stats['q25']:.4f}, {stats['q75']:.4f}]")

    @staticmethod
    def calculate_correlation(scores1, scores2, name1='Score 1', name2='Score 2'):
        """
        Calculate correlation between two sets of scores.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            name1: Name of first scores
            name2: Name of second scores

        Returns:
            Dictionary with correlation statistics
        """
        scores1 = np.asarray(scores1)
        scores2 = np.asarray(scores2)

        correlation = np.corrcoef(scores1, scores2)[0, 1]
        covariance = np.cov(scores1, scores2)[0, 1]

        return {
            'name1': name1,
            'name2': name2,
            'correlation': correlation,
            'covariance': covariance,
            'interpretation': Metrics._interpret_correlation(correlation)
        }

    @staticmethod
    def _interpret_correlation(r):
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        if abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.4:
            return "Moderate"
        elif abs_r >= 0.2:
            return "Weak"
        else:
            return "Very Weak/None"

    @staticmethod
    def print_correlation_report(scores1, scores2, name1='ERI', name2='BVI'):
        """
        Print correlation analysis report.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            name1: Name of first scores
            name2: Name of second scores
        """
        corr = Metrics.calculate_correlation(scores1, scores2, name1, name2)

        print(f"\n📊 Correlation Analysis: {corr['name1']} vs {corr['name2']}")
        print(f"   Pearson Correlation: {corr['correlation']:.4f}")
        print(f"   Covariance:          {corr['covariance']:.4f}")
        print(f"   Interpretation:      {corr['interpretation']}")
