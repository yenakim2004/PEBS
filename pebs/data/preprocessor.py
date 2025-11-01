"""
Data preprocessing module for NSDUH survey data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc


class NSDUHPreprocessor:
    """
    Preprocessor for NSDUH survey data.
    Handles missing values, feature selection, and train/test splitting.
    """

    def __init__(self, missing_threshold=0.5, test_size=0.2, random_state=42):
        """
        Initialize preprocessor.

        Args:
            missing_threshold: Remove columns with missing% > this threshold
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
        """
        self.missing_threshold = missing_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_column = None

    def identify_target(self, df, verbose=True):
        """
        Identify or create target variable for alcohol risk.

        Args:
            df: Input DataFrame
            verbose: Print information

        Returns:
            Series with target values
        """
        if verbose:
            print("🎯 Identifying target variable...")

        # Common NSDUH alcohol-related variables
        target_candidates = [
            'ABODLANG',  # Alcohol Abuse/Dependence
            'ABODAL2',   # Alcohol Abuse/Dependence (alternative)
            'ALCABDEP',  # Alcohol Abuse or Dependence
            'ALDEPEV',   # Alcohol Dependence Ever
            'ABUSALCO',  # Alcohol Abuse
            'ALCYR'      # Past Year Alcohol Use
        ]

        # Try to find existing target variable
        for candidate in target_candidates:
            if candidate in df.columns:
                self.target_column = candidate
                if verbose:
                    print(f"   Using '{candidate}' as target variable")
                    value_counts = df[candidate].value_counts()
                    print(f"   Distribution: {value_counts.to_dict()}")
                return df[candidate]

        # Fallback: Create binary target from alcohol use frequency
        if verbose:
            print("   ⚠️  Standard target variables not found")

        # Try to create target from available alcohol columns
        alcohol_keywords = ['alc', 'drink', 'alcohol']
        alcohol_cols = [col for col in df.columns
                        if any(keyword in col.lower() for keyword in alcohol_keywords)]

        if alcohol_cols:
            # Use first alcohol column as proxy
            proxy_col = alcohol_cols[0]
            if verbose:
                print(f"   Creating binary target from '{proxy_col}'")

            # Create binary classification (0 or 1)
            target = (df[proxy_col].fillna(0) > df[proxy_col].median()).astype(int)
            self.target_column = 'alcohol_risk'

            return target
        else:
            raise ValueError(
                "Could not identify or create target variable. "
                "Please ensure NSDUH data contains alcohol-related columns."
            )

    def remove_high_missing(self, df, verbose=True):
        """
        Remove columns with high percentage of missing values.

        Args:
            df: Input DataFrame
            verbose: Print information

        Returns:
            DataFrame with columns removed
        """
        if verbose:
            print(f"🔧 Removing columns with >{self.missing_threshold*100}% missing values...")

        missing_pct = df.isnull().sum() / len(df)
        valid_cols = missing_pct[missing_pct < self.missing_threshold].index.tolist()

        removed_count = len(df.columns) - len(valid_cols)

        if verbose:
            print(f"   Removed {removed_count} columns")
            print(f"   Remaining columns: {len(valid_cols)}")

        return df[valid_cols]

    def select_numeric_features(self, df, target_col=None, verbose=True):
        """
        Select numeric columns for modeling.

        Args:
            df: Input DataFrame
            target_col: Name of target column to exclude (optional)
            verbose: Print information

        Returns:
            DataFrame with numeric features only
        """
        if verbose:
            print("📊 Selecting numeric features...")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column if specified
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if verbose:
            print(f"   Selected {len(numeric_cols)} numeric features")

        self.feature_names = numeric_cols
        return df[numeric_cols]

    def fill_missing(self, df, method='median', verbose=True):
        """
        Fill remaining missing values.

        Args:
            df: Input DataFrame
            method: Imputation method ('median', 'mean', 'zero')
            verbose: Print information

        Returns:
            DataFrame with missing values filled
        """
        if verbose:
            missing_before = df.isnull().sum().sum()
            print(f"🔧 Filling missing values using {method} method...")
            print(f"   Missing values before: {missing_before}")

        df_filled = df.copy()

        if method == 'median':
            df_filled = df_filled.fillna(df_filled.median())
        elif method == 'mean':
            df_filled = df_filled.fillna(df_filled.mean())
        elif method == 'zero':
            df_filled = df_filled.fillna(0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill any remaining NaN with 0
        df_filled = df_filled.fillna(0)

        if verbose:
            missing_after = df_filled.isnull().sum().sum()
            print(f"   Missing values after: {missing_after}")

        return df_filled

    def split_data(self, X, y, stratify=True, verbose=True):
        """
        Split data into training and test sets.

        Args:
            X: Feature matrix
            y: Target vector
            stratify: Use stratified sampling
            verbose: Print information

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if verbose:
            print(f"✂️  Splitting data ({1-self.test_size:.0%} train / {self.test_size:.0%} test)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y if stratify else None,
            random_state=self.random_state
        )

        if verbose:
            print(f"✅ Split completed:")
            print(f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
            print(f"   Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
            print(f"   Features: {X_train.shape[1]}")

            if hasattr(y_train, 'value_counts'):
                print(f"\n   Train target distribution: {y_train.value_counts().to_dict()}")
                print(f"   Test target distribution:  {y_test.value_counts().to_dict()}")
            else:
                print(f"\n   Train target distribution: {np.bincount(y_train)}")
                print(f"   Test target distribution:  {np.bincount(y_test)}")

        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train, X_test, verbose=True):
        """
        Standardize features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features
            verbose: Print information

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if verbose:
            print("📏 Standardizing features...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if verbose:
            print(f"✅ Scaling completed")
            print(f"   Train shape: {X_train_scaled.shape}")
            print(f"   Test shape:  {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled

    def process(self, df, verbose=True):
        """
        Complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            verbose: Print information

        Returns:
            Dictionary with processed data:
            {
                'X_train', 'X_test', 'y_train', 'y_test',
                'X_train_scaled', 'X_test_scaled',
                'feature_names', 'target_column'
            }
        """
        if verbose:
            print("="*80)
            print("NSDUH PREPROCESSING PIPELINE")
            print("="*80)
            print(f"\nInput shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB\n")

        # Step 1: Identify target
        y = self.identify_target(df, verbose)

        # Step 2: Remove high missing columns
        df_clean = self.remove_high_missing(df, verbose)

        # Step 3: Select numeric features
        X = self.select_numeric_features(df_clean, self.target_column, verbose)

        # Step 4: Fill missing values
        X = self.fill_missing(X, verbose=verbose)

        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X.values, y.values, verbose=verbose
        )

        # Step 6: Standardize
        X_train_scaled, X_test_scaled = self.fit_transform(
            X_train, X_test, verbose=verbose
        )

        # Clean up memory
        del df_clean, X
        gc.collect()

        if verbose:
            print("\n🗑️  Cleaned up intermediate data from memory")
            print("\n" + "="*80)
            print("PREPROCESSING COMPLETE")
            print("="*80)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'scaler': self.scaler
        }
