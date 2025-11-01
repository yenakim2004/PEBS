"""
Visualization utilities for PEBS system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class Visualizer:
    """
    Visualization utilities for PEBS system results.
    """

    def __init__(self, save_figures=True, figures_path='figures/', dpi=300, format='png'):
        """
        Initialize visualizer.

        Args:
            save_figures: Whether to save figures to disk
            figures_path: Directory to save figures
            dpi: Figure resolution
            format: Figure format ('png', 'pdf', 'svg')
        """
        self.save_figures = save_figures
        self.figures_path = figures_path
        self.dpi = dpi
        self.format = format

        # Create figures directory if it doesn't exist
        if self.save_figures:
            os.makedirs(self.figures_path, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)

    def _save_fig(self, filename):
        """Save figure if save_figures is True."""
        if self.save_figures:
            filepath = os.path.join(self.figures_path, f"{filename}.{self.format}")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"   💾 Saved: {filepath}")

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', labels=None, cmap='Blues'):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            title: Plot title
            labels: Class labels
            cmap: Color map
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels)
            plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)

        plt.tight_layout()
        self._save_fig(title.lower().replace(' ', '_'))
        plt.show()

    def plot_score_distribution(self, scores, title='Score Distribution',
                                 threshold=None, xlabel='Score'):
        """
        Plot score distribution histogram.

        Args:
            scores: Array of scores
            title: Plot title
            threshold: Threshold line to plot
            xlabel: X-axis label
        """
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')

        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold})')
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        self._save_fig(title.lower().replace(' ', '_'))
        plt.show()

    def plot_risk_space(self, eri_scores, bvi_scores, risk_categories,
                        eri_threshold=0.5, bvi_threshold=0.5):
        """
        Plot PEBS risk space (ERI vs BVI).

        Args:
            eri_scores: Environmental Risk Index scores
            bvi_scores: Biological Vulnerability Index scores
            risk_categories: Risk category labels (0-3)
            eri_threshold: ERI threshold line
            bvi_threshold: BVI threshold line
        """
        plt.figure(figsize=(10, 8))

        # Scatter plot
        scatter = plt.scatter(eri_scores, bvi_scores, c=risk_categories,
                             cmap='RdYlGn_r', alpha=0.6,
                             edgecolors='black', linewidth=0.5, s=50)

        # Threshold lines
        plt.axhline(bvi_threshold, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(eri_threshold, color='gray', linestyle='--', alpha=0.5)

        # Labels and title
        plt.xlabel('ERI Score (Environmental Risk)', fontsize=12)
        plt.ylabel('BVI Score (Biological Vulnerability)', fontsize=12)
        plt.title('PEBS Risk Space', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)

        # Legend
        category_names = ['Low Risk', 'Medium-Environmental',
                          'Medium-Biological', 'High Risk']
        colors = ['green', 'yellow', 'orange', 'red']
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=10, alpha=0.7)
                   for c in colors]
        plt.legend(handles, category_names, loc='upper left', fontsize=10)

        plt.tight_layout()
        self._save_fig('pebs_risk_space')
        plt.show()

    def plot_risk_distribution(self, risk_categories, category_names=None):
        """
        Plot risk category distribution (bar + pie).

        Args:
            risk_categories: Array of risk categories
            category_names: Names of categories
        """
        if category_names is None:
            category_names = ['Low Risk', 'Medium-Environmental',
                              'Medium-Biological', 'High Risk']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        counts = [np.sum(risk_categories == i) for i in range(4)]
        colors = ['green', 'yellow', 'orange', 'red']

        axes[0].bar(range(4), counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(category_names, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Risk Category Distribution')
        axes[0].grid(axis='y', alpha=0.3)

        # Pie chart
        axes[1].pie(counts, labels=category_names, autopct='%1.1f%%',
                    colors=colors, startangle=90)
        axes[1].set_title('Risk Category Proportions')

        plt.tight_layout()
        self._save_fig('risk_distribution')
        plt.show()

    def plot_correlation(self, eri_scores, bvi_scores):
        """
        Plot ERI vs BVI correlation.

        Args:
            eri_scores: Environmental Risk Index scores
            bvi_scores: Biological Vulnerability Index scores
        """
        plt.figure(figsize=(10, 6))

        # Scatter plot
        plt.scatter(eri_scores, bvi_scores, alpha=0.5)

        # Linear fit
        correlation = np.corrcoef(eri_scores, bvi_scores)[0, 1]
        z = np.polyfit(eri_scores, bvi_scores, 1)
        p = np.poly1d(z)
        plt.plot(eri_scores, p(eri_scores), "r--", alpha=0.8, linewidth=2,
                 label=f'Linear Fit (r={correlation:.3f})')

        plt.xlabel('ERI Score (Environmental Risk)')
        plt.ylabel('BVI Score (Biological Vulnerability)')
        plt.title('ERI vs BVI Correlation Analysis')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        self._save_fig('eri_bvi_correlation')
        plt.show()

        return correlation

    def plot_feature_importance(self, feature_names, importances,
                                 title='Feature Importance', top_n=20):
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            importances: Array of importance values
            title: Plot title
            top_n: Number of top features to show
        """
        # Select top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, color='skyblue', edgecolor='black')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self._save_fig(title.lower().replace(' ', '_'))
        plt.show()

    def create_dashboard(self, eri_scores_test, bvi_scores_test,
                        risk_categories, eri_cm, bvi_cm,
                        eri_threshold=0.5, bvi_threshold=0.5):
        """
        Create comprehensive dashboard with all visualizations.

        Args:
            eri_scores_test: ERI scores on test set
            bvi_scores_test: BVI scores on test set
            risk_categories: Risk category assignments
            eri_cm: ERI confusion matrix
            bvi_cm: BVI confusion matrix
            eri_threshold: ERI threshold
            bvi_threshold: BVI threshold
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. ERI Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(eri_scores_test, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(eri_threshold, color='red', linestyle='--', label='Threshold')
        ax1.set_xlabel('ERI Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('ERI Score Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. BVI Score Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(bvi_scores_test, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(bvi_threshold, color='red', linestyle='--', label='Threshold')
        ax2.set_xlabel('BVI Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('BVI Score Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Risk Categories Pie Chart
        ax3 = fig.add_subplot(gs[0, 2])
        category_names = ['Low Risk', 'Medium-Env', 'Medium-Bio', 'High Risk']
        counts = [np.sum(risk_categories == i) for i in range(4)]
        colors = ['green', 'yellow', 'orange', 'red']
        ax3.pie(counts, labels=category_names, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax3.set_title('Risk Distribution')

        # 4. ERI Confusion Matrix
        ax4 = fig.add_subplot(gs[1, 0])
        sns.heatmap(eri_cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
        ax4.set_title('ERI Confusion Matrix')
        ax4.set_ylabel('True')
        ax4.set_xlabel('Predicted')

        # 5. BVI Confusion Matrix
        ax5 = fig.add_subplot(gs[1, 1])
        sns.heatmap(bvi_cm, annot=True, fmt='d', cmap='Greens', ax=ax5, cbar=False)
        ax5.set_title('BVI Confusion Matrix')
        ax5.set_ylabel('True')
        ax5.set_xlabel('Predicted')

        # 6. Risk Space Scatter
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(eri_scores_test, bvi_scores_test,
                             c=risk_categories, cmap='RdYlGn_r',
                             alpha=0.6, edgecolors='black', linewidth=0.5)
        ax6.axhline(bvi_threshold, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(eri_threshold, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('ERI Score')
        ax6.set_ylabel('BVI Score')
        ax6.set_title('PEBS Risk Space')
        ax6.grid(alpha=0.3)

        # 7. Correlation (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        correlation = np.corrcoef(eri_scores_test, bvi_scores_test)[0, 1]
        ax7.scatter(eri_scores_test, bvi_scores_test, alpha=0.5)
        z = np.polyfit(eri_scores_test, bvi_scores_test, 1)
        p = np.poly1d(z)
        ax7.plot(eri_scores_test, p(eri_scores_test), "r--", alpha=0.8, linewidth=2,
                 label=f'Linear Fit (r={correlation:.3f})')
        ax7.set_xlabel('ERI Score (Environmental Risk)')
        ax7.set_ylabel('BVI Score (Biological Vulnerability)')
        ax7.set_title('ERI vs BVI Correlation Analysis')
        ax7.legend()
        ax7.grid(alpha=0.3)

        plt.suptitle('PEBS System - Comprehensive Performance Dashboard',
                     fontsize=16, fontweight='bold', y=0.995)

        self._save_fig('pebs_dashboard')
        plt.show()

        print(f"\n✅ Dashboard created")
        print(f"   ERI-BVI Correlation: {correlation:.4f}")
