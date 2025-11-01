"""
EEG feature extraction module.
Extracts time-domain and frequency-domain features from EEG signals.
"""

import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import gc


class EEGFeatureExtractor:
    """
    Extract statistical and spectral features from EEG signals.

    Features extracted:
    - Time domain: mean, std, skewness, kurtosis, min, max, range
    - Frequency domain: band power (Delta, Theta, Alpha, Beta, Gamma)
    """

    def __init__(self, sampling_rate=256, bands=None):
        """
        Initialize EEG feature extractor.

        Args:
            sampling_rate: Sampling rate in Hz (default: 256)
            bands: Dictionary of frequency bands
                   Default: {delta: (0.5,4), theta: (4,8), alpha: (8,13),
                            beta: (13,30), gamma: (30,50)}
        """
        self.sampling_rate = sampling_rate

        # EEG frequency bands (Hz)
        if bands is None:
            self.bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
        else:
            self.bands = bands

        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self):
        """Generate feature names."""
        names = [
            'mean', 'std', 'skewness', 'kurtosis',
            'min', 'max', 'range'
        ]
        for band_name in self.bands.keys():
            names.append(f'{band_name}_power')
        return names

    def extract_time_features(self, signal):
        """
        Extract time-domain statistical features.

        Args:
            signal: 1D array of signal values

        Returns:
            Dictionary of time-domain features
        """
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'skewness': skew(signal),
            'kurtosis': kurtosis(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'range': np.ptp(signal)  # peak-to-peak
        }

    def extract_freq_features(self, signal):
        """
        Extract frequency-domain features using FFT.

        Args:
            signal: 1D array of signal values

        Returns:
            Dictionary of frequency-domain (band power) features
        """
        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1.0 / self.sampling_rate)

        # Compute power spectral density
        psd = np.abs(fft_vals) ** 2

        # Extract band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            # Find frequency indices for this band
            idx = np.logical_and(fft_freq >= low_freq, fft_freq < high_freq)
            # Sum power in this band
            band_powers[f'{band_name}_power'] = np.sum(psd[idx])

        return band_powers

    def extract_features_from_file(self, df, sensor_col=None):
        """
        Extract all features from a single EEG file.

        Args:
            df: DataFrame with EEG data
            sensor_col: Name of sensor value column (auto-detect if None)

        Returns:
            1D array of features
        """
        # Auto-detect sensor column
        if sensor_col is None:
            if 'sensor value' in df.columns:
                sensor_col = 'sensor value'
            elif 'value' in df.columns:
                sensor_col = 'value'
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                sensor_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        # Extract signal values
        signal = df[sensor_col].values

        # Extract time-domain features
        time_features = self.extract_time_features(signal)

        # Extract frequency-domain features
        freq_features = self.extract_freq_features(signal)

        # Combine all features
        features = list(time_features.values()) + list(freq_features.values())

        return np.array(features)

    def extract_features_from_dataset(self, data_list, description='data', verbose=True):
        """
        Extract features from list of EEG files.

        Args:
            data_list: List of DataFrames (EEG files)
            description: Description for progress bar
            verbose: Show progress bar

        Returns:
            2D array of features (n_samples × n_features)
        """
        feature_matrix = []

        iterator = tqdm(data_list, desc=f"Extracting features from {description}") if verbose else data_list

        for df in iterator:
            try:
                features = self.extract_features_from_file(df)
                feature_matrix.append(features)
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error extracting features: {e}")
                # Add zero features as placeholder
                features = np.zeros(len(self.feature_names))
                feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)

        if verbose and len(feature_matrix) > 0:
            print(f"✅ Feature extraction completed")
            print(f"   Shape: {feature_matrix.shape}")
            print(f"   Features: {len(self.feature_names)} ({', '.join(self.feature_names[:5])}...)")

        return feature_matrix

    def get_feature_names(self):
        """Get list of feature names."""
        return self.feature_names

    def get_feature_info(self):
        """Get information about extracted features."""
        info = {
            'total_features': len(self.feature_names),
            'time_domain': 7,
            'frequency_domain': len(self.bands),
            'sampling_rate': self.sampling_rate,
            'frequency_bands': self.bands,
            'feature_names': self.feature_names
        }
        return info
