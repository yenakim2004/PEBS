"""
Environmental Risk Index (ERI) Model.
Predicts alcohol risk from NSDUH survey data.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ERIModel:
    """
    Environmental Risk Index (ERI) Model.
    Uses Random Forest or Logistic Regression to predict alcohol risk
    from environmental/survey data.
    """

    def __init__(self, model_type='RandomForest', **kwargs):
        """
        Initialize ERI model.

        Args:
            model_type: Type of model ('RandomForest' or 'LogisticRegression')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False

        # Initialize model based on type
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 20),
                min_samples_leaf=kwargs.get('min_samples_leaf', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),
                verbose=kwargs.get('verbose', 0)
            )
        elif model_type == 'LogisticRegression':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),
                verbose=kwargs.get('verbose', 0)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train, verbose=True):
        """
        Train the ERI model.

        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Print training information

        Returns:
            Self
        """
        if verbose:
            print("🏋️  Training ERI model...")
            print(f"   Model type: {self.model_type}")
            print(f"   Training samples: {X_train.shape[0]:,}")
            print(f"   Features: {X_train.shape[1]}")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if verbose:
            train_pred = self.model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            print(f"\n✅ Training completed")
            print(f"   Training accuracy: {train_acc:.4f}")

        return self

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def get_eri_scores(self, X):
        """
        Get ERI scores (probability of positive class).

        Args:
            X: Features

        Returns:
            ERI scores (0-1)
        """
        proba = self.predict_proba(X)
        return proba[:, 1]

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        if verbose:
            print("\n📊 ERI Model Evaluation:")
            print(f"   Test Accuracy: {accuracy:.4f}")
            print(f"\n   Confusion Matrix:")
            print(f"   {cm}")
            print(f"\n   Classification Report:")
            print(classification_report(y_test, y_pred))

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def save(self, filepath):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ ERI model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded ERIModel instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ ERI model loaded from {filepath}")
        return model

    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance (for Random Forest).

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            Dictionary or None if not available
        """
        if self.model_type == 'RandomForest' and self.is_fitted:
            importances = self.model.feature_importances_

            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            # Sort by importance
            indices = np.argsort(importances)[::-1][:top_n]

            return {
                'feature_names': [feature_names[i] for i in indices],
                'importances': importances[indices]
            }
        else:
            return None
