# PEBS: Personalized Evidence-Based Screening System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system for alcoholism risk assessment using environmental (survey) and biological (EEG) data.

## Overview

PEBS (Personalized Evidence-Based Screening) is a 2-step risk prediction system that combines:

- **Step 1A**: Environmental Risk Index (ERI) from NSDUH survey data
- **Step 1B**: Biological Vulnerability Index (BVI) from EEG biomarkers
- **Step 2**: Risk Classification combining ERI + BVI into 4 categories

### Risk Categories

| Category | Name | Description |
|----------|------|-------------|
| 0 | Low Risk | Low environmental and biological risk factors |
| 1 | Medium-Environmental | High environmental risk, low biological vulnerability |
| 2 | Medium-Biological | Low environmental risk, high biological vulnerability |
| 3 | High Risk | Both environmental and biological risk factors elevated |

## Features

- Complete end-to-end pipeline from raw data to predictions
- Memory-optimized for 16GB RAM systems
- Modular architecture with separate ERI and BVI models
- Comprehensive visualization and evaluation
- Command-line interface for easy execution
- Configurable via YAML file

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 16GB recommended (minimum 8GB)
- **Storage**: ~15GB for datasets
- **OS**: Linux, macOS, or Windows

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/username/PEBS.git
cd PEBS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### 3. Download Datasets

```bash
python download_data.py
```

Follow the instructions to download:
- **NSDUH**: National Survey on Drug Use and Health
- **SMNI**: EEG Database from UCI ML Repository

## Quick Start

### Training

Train the complete PEBS system:

```bash
python train.py
```

This will:
1. Load and preprocess NSDUH data (949K samples)
2. Load SMNI EEG data (948 files)
3. Extract EEG features (time + frequency domain)
4. Train ERI model (Random Forest)
5. Train BVI model (Logistic Regression)
6. Perform risk classification
7. Generate visualizations
8. Save all models to `models/`

**Estimated runtime**: 30-60 minutes on full dataset

### Making Predictions

Predict risk for new samples:

```bash
python predict.py --nsduh sample_survey.csv --eeg sample_eeg.csv
```

Batch prediction:

```bash
python predict.py --nsduh batch_surveys.csv --eeg batch_features.csv --batch
```

## Project Structure

```
PEBS/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── config.yaml                  # Configuration file
├── download_data.py             # Data download script
├── train.py                     # Training pipeline
├── predict.py                   # Prediction script
│
├── pebs/                        # Main package
│   ├── __init__.py
│   ├── data/                    # Data loading and preprocessing
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/                # Feature extraction
│   │   └── eeg_extractor.py
│   ├── models/                  # ML models
│   │   ├── eri_model.py
│   │   ├── bvi_model.py
│   │   └── risk_classifier.py
│   └── utils/                   # Utilities
│       ├── visualization.py
│       └── metrics.py
│
├── data/                        # Data directory (gitignored)
│   ├── raw/
│   │   ├── NSDUH_2002_2018_Tab.tsv
│   │   ├── SMNI_CMI_TRAIN/      # 468 EEG files
│   │   └── SMNI_CMI_TEST/       # 480 EEG files
│   └── processed/
│
├── models/                      # Trained models (gitignored)
│   ├── eri_model.pkl
│   ├── bvi_model.pkl
│   ├── risk_classifier.pkl
│   ├── scaler_nsduh.pkl
│   ├── scaler_eeg.pkl
│   └── eeg_extractor.pkl
│
└── figures/                     # Visualizations (gitignored)
```

## Configuration

Edit `config.yaml` to customize:

- **Data paths**: NSDUH and SMNI locations
- **Model parameters**: Random Forest, Logistic Regression settings
- **Thresholds**: ERI and BVI classification thresholds
- **Memory settings**: Chunk sizes for low-memory mode
- **Visualization**: Figure settings

Example:

```yaml
models:
  eri:
    type: RandomForest
    n_estimators: 100
    max_depth: 10

  risk:
    eri_threshold: 0.5
    bvi_threshold: 0.5

memory:
  nsduh_chunksize: 10000
  low_memory_mode: true
```

## Datasets

### NSDUH (National Survey on Drug Use and Health)

- **Source**: SAMHDA (https://www.datafiles.samhsa.gov/)
- **Size**: ~12GB, 949,285 rows × 3,662 columns
- **Format**: Tab-separated values (TSV)
- **Content**: Comprehensive survey on substance use

### SMNI (EEG Database)

- **Source**: UCI ML Repository
- **Size**: 948 CSV files (468 train + 480 test)
- **Format**: 64-channel EEG recordings
- **Sampling Rate**: 256 Hz
- **Content**: EEG signals from control and alcoholic subjects

## Usage Examples

### Basic Training

```bash
# Train with default settings
python train.py

# Train with custom config
python train.py --config my_config.yaml

# Quiet mode (less output)
python train.py --quiet
```

### Advanced Prediction

```python
from pebs.models.eri_model import ERIModel
from pebs.models.bvi_model import BVIModel
from pebs.models.risk_classifier import RiskClassifier

# Load models
eri_model = ERIModel.load('models/eri_model.pkl')
bvi_model = BVIModel.load('models/bvi_model.pkl')
risk_classifier = RiskClassifier.load('models/risk_classifier.pkl')

# Get scores
eri_score = eri_model.get_eri_scores(nsduh_features)[0]
bvi_score = bvi_model.get_bvi_scores(eeg_features)[0]

# Classify risk
result = risk_classifier.classify_single(eri_score, bvi_score)
print(f"Risk Category: {result['name']}")
```

### Custom Feature Extraction

```python
from pebs.features.eeg_extractor import EEGFeatureExtractor
import pandas as pd

# Initialize extractor
extractor = EEGFeatureExtractor(
    sampling_rate=256,
    bands={'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13)}
)

# Extract features
eeg_data = pd.read_csv('sample_eeg.csv')
features = extractor.extract_features_from_file(eeg_data)
```

## Performance

### Model Accuracy

Typical performance on test set:

- **ERI Model**: 85-90% accuracy
- **BVI Model**: 80-85% accuracy
- **Risk Classification**: 4-class categorization

### Runtime

On a system with 16GB RAM and modern CPU:

- **Training**: 30-60 minutes (full dataset)
- **Prediction (single)**: < 1 second
- **Prediction (batch, 1000 samples)**: ~10 seconds

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. Increase `nsduh_chunksize` in `config.yaml`
2. Enable `low_memory_mode: true`
3. Close other applications
4. Consider using a subset of data for testing

### Missing Data

If datasets are not found:

```bash
# Check data status
python download_data.py --check

# Follow download instructions
python download_data.py
```

### Model Loading Errors

If models fail to load:

1. Ensure training completed successfully
2. Check `models/` directory contains all required files
3. Re-run training: `python train.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pebs2024,
  title={PEBS: Personalized Evidence-Based Screening System},
  author={PEBS Development Team},
  year={2024},
  url={https://github.com/username/PEBS}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NSDUH dataset: Substance Abuse and Mental Health Data Archive (SAMHDA)
- SMNI EEG dataset: UCI Machine Learning Repository
- Research foundations: Evidence-based alcoholism screening methodologies

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or issues:

- Open an issue on GitHub
- Email: pebs-dev@example.com

---

**Disclaimer**: This tool is for research and educational purposes only. It should not be used as the sole basis for clinical decisions. Always consult qualified healthcare professionals for medical advice.
