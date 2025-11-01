"""
Data loading modules for NSDUH and SMNI datasets.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


class NSDUHLoader:
    """
    Loader for NSDUH (National Survey on Drug Use and Health) data.
    Handles large TSV files with memory optimization for 16GB RAM systems.
    """

    def __init__(self, file_path, chunksize=10000, low_memory=True):
        """
        Initialize NSDUH loader.

        Args:
            file_path: Path to NSDUH TSV file
            chunksize: Number of rows to process at once (default: 10000)
            low_memory: Enable low memory mode (default: True)
        """
        self.file_path = file_path
        self.chunksize = chunksize
        self.low_memory = low_memory
        self.data = None

    def load(self, verbose=True):
        """
        Load NSDUH data with memory optimization.

        Args:
            verbose: Print loading progress (default: True)

        Returns:
            DataFrame with loaded data
        """
        if verbose:
            print(f"📊 Loading NSDUH dataset from {self.file_path}...")
            print(f"   Using chunksize: {self.chunksize}")

        try:
            # Load data
            self.data = pd.read_csv(
                self.file_path,
                sep='\t',
                low_memory=self.low_memory
            )

            if verbose:
                print(f"✅ Dataset loaded successfully")
                print(f"   Shape: {self.data.shape}")
                print(f"   Memory usage: {self.data.memory_usage(deep=True).sum() / 1e9:.2f} GB")

            return self.data

        except FileNotFoundError:
            raise FileNotFoundError(
                f"NSDUH data file not found at {self.file_path}. "
                f"Please run 'python download_data.py' first."
            )
        except Exception as e:
            raise Exception(f"Error loading NSDUH data: {e}")

    def get_info(self):
        """Get dataset information."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        info = {
            'shape': self.data.shape,
            'columns': len(self.data.columns),
            'rows': len(self.data),
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1e6,
            'dtypes': self.data.dtypes.value_counts().to_dict(),
            'missing_pct': (self.data.isnull().sum() / len(self.data) * 100).describe()
        }
        return info


class SMNILoader:
    """
    Loader for SMNI (EEG) dataset.
    Loads multiple CSV files from SMNI_CMI_TRAIN and SMNI_CMI_TEST directories.
    """

    def __init__(self, train_path, test_path):
        """
        Initialize SMNI loader.

        Args:
            train_path: Path to SMNI_CMI_TRAIN directory
            test_path: Path to SMNI_CMI_TEST directory
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

    def load_files(self, directory, file_count, dataset_name='data', verbose=True):
        """
        Load EEG CSV files from directory.

        Args:
            directory: Path to folder containing DataN.csv files
            file_count: Number of files to load
            dataset_name: Name for progress display
            verbose: Print loading progress

        Returns:
            Tuple of (data_list, labels)
        """
        data_list = []
        labels = []

        if verbose:
            print(f"📂 Loading {file_count} {dataset_name} files from {directory}")

        # Check if directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Directory not found: {directory}. "
                f"Please run 'python download_data.py' first."
            )

        # Load files with progress bar
        iterator = tqdm(range(1, file_count + 1), desc=f"Loading {dataset_name}") if verbose else range(1, file_count + 1)

        for i in iterator:
            try:
                file_path = os.path.join(directory, f'Data{i}.csv')
                df = pd.read_csv(file_path)

                # Extract label from filename pattern in 'name' column
                # Format: 'co2aXXXX' (control) or 'co2cXXXX' (alcoholic)
                if 'name' in df.columns and len(df) > 0:
                    name_value = df['name'].iloc[0]
                    if len(name_value) > 3:
                        label_char = name_value[3]  # Position 3 is 'a' or 'c'
                        label = 1 if label_char == 'c' else 0  # c=alcoholic=1, a=control=0
                    else:
                        label = 0  # Default to control
                else:
                    label = 0  # Default to control if 'name' column doesn't exist

                data_list.append(df)
                labels.append(label)

            except FileNotFoundError:
                if verbose:
                    print(f"   ⚠️  Warning: File Data{i}.csv not found, skipping...")
                continue
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error loading file {i}: {e}")
                continue

        if verbose:
            print(f"✅ Loaded {len(data_list)} files")
            if labels:
                print(f"   Label distribution: Control={labels.count(0)}, Alcoholic={labels.count(1)}")

        return data_list, labels

    def load_train(self, file_count=468, verbose=True):
        """
        Load training dataset.

        Args:
            file_count: Number of files in training set (default: 468)
            verbose: Print loading progress

        Returns:
            Tuple of (train_data, train_labels)
        """
        self.train_data, self.train_labels = self.load_files(
            self.train_path, file_count, 'TRAIN', verbose
        )
        return self.train_data, self.train_labels

    def load_test(self, file_count=480, verbose=True):
        """
        Load test dataset.

        Args:
            file_count: Number of files in test set (default: 480)
            verbose: Print loading progress

        Returns:
            Tuple of (test_data, test_labels)
        """
        self.test_data, self.test_labels = self.load_files(
            self.test_path, file_count, 'TEST', verbose
        )
        return self.test_data, self.test_labels

    def load_all(self, train_count=468, test_count=480, verbose=True):
        """
        Load both training and test datasets.

        Args:
            train_count: Number of files in training set
            test_count: Number of files in test set
            verbose: Print loading progress

        Returns:
            Dictionary with train_data, train_labels, test_data, test_labels
        """
        self.load_train(train_count, verbose)
        self.load_test(test_count, verbose)

        return {
            'train_data': self.train_data,
            'train_labels': self.train_labels,
            'test_data': self.test_data,
            'test_labels': self.test_labels
        }

    def get_info(self):
        """Get dataset information."""
        info = {}

        if self.train_data is not None:
            info['train'] = {
                'files': len(self.train_data),
                'sample_shape': self.train_data[0].shape if self.train_data else None,
                'labels': {
                    'control': self.train_labels.count(0),
                    'alcoholic': self.train_labels.count(1)
                }
            }

        if self.test_data is not None:
            info['test'] = {
                'files': len(self.test_data),
                'sample_shape': self.test_data[0].shape if self.test_data else None,
                'labels': {
                    'control': self.test_labels.count(0),
                    'alcoholic': self.test_labels.count(1)
                }
            }

        return info
