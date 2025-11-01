"""
Data preprocessing module for NSDUH survey data.
Includes PCA dimensionality reduction for memory efficiency.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gc


class NSDUHPreprocessor:
    """
    Preprocessor for NSDUH survey data.
    Handles missing values, feature selection, and train/test splitting.
    """

    def __init__(self, missing_threshold=0.5, test_size=0.2, random_state=42, pca_config=None):
        """
        Initialize preprocessor.

        Args:
            missing_threshold: Remove columns with missing% > this threshold
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            pca_config: PCA configuration dictionary (None = no PCA)
                        {'enabled': True, 'n_components': 0.95, 'whiten': False}
        """
        self.missing_threshold = missing_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_config = pca_config or {}
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
        # Prioritize binary variables with balanced distribution
        target_candidates = [
            'ALCEVER',   # Ever used alcohol (binary: 0/1) - BEST
            'ALCYR',     # Past Year Alcohol Use (binary: 0/1)
            'ABODLANG',  # Alcohol Abuse/Dependence
            'ABODAL2',   # Alcohol Abuse/Dependence (alternative)
            'ALCABDEP',  # Alcohol Abuse or Dependence
            'ALDEPEV',   # Alcohol Dependence Ever
            'ABUSALCO',  # Alcohol Abuse
            'ALCTRY'     # Age tried alcohol (1,2,3,9) - 97% is 9, AVOID
        ]

        # Try to find existing target variable
        for candidate in target_candidates:
            if candidate in df.columns:
                self.target_column = candidate
                if verbose:
                    print(f"   Using '{candidate}' as target variable")
                    value_counts = df[candidate].value_counts()
                    print(f"   Distribution (before filtering): {value_counts.to_dict()}")
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

            # Safe target distribution display
            if hasattr(y_train, 'value_counts'):
                print(f"\n   Train target distribution: {y_train.value_counts().to_dict()}")
                print(f"   Test target distribution:  {y_test.value_counts().to_dict()}")
            else:
                # Use unique/count instead of bincount for safety
                train_unique, train_counts = np.unique(y_train, return_counts=True)
                test_unique, test_counts = np.unique(y_test, return_counts=True)
                print(f"\n   Train target distribution: {dict(zip(train_unique, train_counts))}")
                print(f"   Test target distribution:  {dict(zip(test_unique, test_counts))}")

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

    def apply_pca(self, X_train, X_test, verbose=True):
        """
        Apply PCA dimensionality reduction.

        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled)
            verbose: Print information

        Returns:
            Tuple of (X_train_pca, X_test_pca)
        """
        if not self.pca_config.get('enabled', False):
            if verbose:
                print("ℹ️  PCA disabled, skipping dimensionality reduction")
            return X_train, X_test

        n_components = self.pca_config.get('n_components', 0.95)
        whiten = self.pca_config.get('whiten', False)
        random_state = self.pca_config.get('random_state', self.random_state)

        if verbose:
            print(f"🔬 Applying PCA dimensionality reduction...")
            if isinstance(n_components, float):
                print(f"   Target variance explained: {n_components*100:.1f}%")
            else:
                print(f"   Target components: {n_components}")
            print(f"   Original features: {X_train.shape[1]}")

        # Initialize and fit PCA
        self.pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )

        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        if verbose:
            print(f"✅ PCA completed")
            print(f"   Reduced to: {X_train_pca.shape[1]} components")
            print(f"   Variance explained: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")
            print(f"   Dimension reduction: {X_train.shape[1]} → {X_train_pca.shape[1]} "
                  f"({(1 - X_train_pca.shape[1]/X_train.shape[1])*100:.1f}% reduction)")

            # Show top components
            if len(self.pca.explained_variance_ratio_) > 0:
                top_n = min(5, len(self.pca.explained_variance_ratio_))
                print(f"   Top {top_n} components explain:")
                for i in range(top_n):
                    print(f"      PC{i+1}: {self.pca.explained_variance_ratio_[i]*100:.2f}%")

        return X_train_pca, X_test_pca

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

        # Step 1.5: Filter out negative values (NSDUH missing codes: -9, -8, -7, etc.)
        if verbose:
            print("🔧 Filtering negative values from target variable...")
            print(f"   Original samples: {len(y):,}")

        valid_mask = y >= 0
        df = df[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        if verbose:
            print(f"   Valid samples: {len(y):,}")
            print(f"   Removed: {(~valid_mask).sum():,} samples with negative values")
            value_counts = y.value_counts()
            print(f"   Distribution (after filtering): {value_counts.to_dict()}\n")

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

        # Step 7: Apply PCA (if enabled)
        X_train_final, X_test_final = self.apply_pca(
            X_train_scaled, X_test_scaled, verbose=verbose
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
            'X_train_scaled': X_train_final,
            'X_test_scaled': X_test_final,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'scaler': self.scaler,
            'pca': self.pca
        }
