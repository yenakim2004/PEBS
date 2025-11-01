"""
Main training script for PEBS system.
Trains ERI, BVI models and performs risk classification.
"""

import os
import yaml
import argparse
import pickle
import numpy as np
import gc
from datetime import datetime

from pebs.data.loader import NSDUHLoader, SMNILoader
from pebs.data.preprocessor import NSDUHPreprocessor
from pebs.features.eeg_extractor import EEGFeatureExtractor
from pebs.models.eri_model import ERIModel
from pebs.models.bvi_model import BVIModel
from pebs.models.risk_classifier import RiskClassifier
from pebs.utils.visualization import Visualizer
from pebs.utils.metrics import Metrics
from sklearn.preprocessing import StandardScaler


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_pebs_system(config, verbose=True):
    """
    Complete PEBS training pipeline.

    Args:
        config: Configuration dictionary
        verbose: Print detailed information

    Returns:
        Dictionary with all trained models and results
    """
    start_time = datetime.now()

    print("\n" + "="*80)
    print("PEBS TRAINING PIPELINE")
    print("="*80)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # ========================================================================
    # STEP 1: Load NSDUH Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1/9: LOADING NSDUH DATA")
    print("="*80)
    print("⏱️  Estimated time: 30-60 seconds\n")

    nsduh_loader = NSDUHLoader(
        file_path=config['data']['nsduh_path'],
        chunksize=config['memory'].get('nsduh_chunksize', 10000),
        low_memory=config['memory'].get('low_memory_mode', True),
        selected_columns=config['data'].get('nsduh_selected_columns', None)
    )

    nsduh_data = nsduh_loader.load(verbose=verbose, use_chunks=True)

    # ========================================================================
    # STEP 2: Preprocess NSDUH Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2/9: PREPROCESSING NSDUH DATA")
    print("="*80)
    print("⏱️  Estimated time: 1-3 minutes\n")

    preprocessor = NSDUHPreprocessor(
        missing_threshold=config['preprocessing']['missing_threshold'],
        test_size=config['data']['train_test_split'],
        random_state=config['data']['random_state'],
        pca_config=config['preprocessing'].get('pca', None)
    )

    processed_data = preprocessor.process(nsduh_data, verbose=verbose)

    # Extract processed data
    X_nsduh_train_scaled = processed_data['X_train_scaled']
    X_nsduh_test_scaled = processed_data['X_test_scaled']
    y_nsduh_train = processed_data['y_train']
    y_nsduh_test = processed_data['y_test']
    scaler_nsduh = processed_data['scaler']

    # Clean up
    del nsduh_data
    gc.collect()

    # ========================================================================
    # STEP 3: Load SMNI EEG Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3/9: LOADING SMNI EEG DATA")
    print("="*80)
    print("⏱️  Estimated time: 10-20 seconds (948 files)\n")

    smni_loader = SMNILoader(
        train_path=config['data']['smni_train_path'],
        test_path=config['data']['smni_test_path']
    )

    smni_data = smni_loader.load_all(verbose=verbose)

    train_data = smni_data['train_data']
    train_labels = smni_data['train_labels']
    test_data = smni_data['test_data']
    test_labels = smni_data['test_labels']

    # ========================================================================
    # STEP 4: Extract EEG Features
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4/9: EXTRACTING EEG FEATURES")
    print("="*80)
    print("⏱️  Estimated time: 1-2 minutes\n")

    extractor = EEGFeatureExtractor(
        sampling_rate=config['eeg']['sampling_rate'],
        bands=config['eeg']['bands']
    )

    X_eeg_train = extractor.extract_features_from_dataset(train_data, 'TRAIN', verbose=verbose)
    y_eeg_train = np.array(train_labels)

    X_eeg_test = extractor.extract_features_from_dataset(test_data, 'TEST', verbose=verbose)
    y_eeg_test = np.array(test_labels)

    # Standardize EEG features
    print("\n📏 Standardizing EEG features...")
    scaler_eeg = StandardScaler()
    X_eeg_train_scaled = scaler_eeg.fit_transform(X_eeg_train)
    X_eeg_test_scaled = scaler_eeg.transform(X_eeg_test)
    print("✅ EEG features standardized")

    # Clean up
    del train_data, test_data
    gc.collect()

    # ========================================================================
    # STEP 5: Train ERI Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5/9: TRAINING ERI MODEL (Environmental Risk Index)")
    print("="*80)
    print("⏱️  Estimated time: 1-2 minutes\n")

    eri_config = config['models']['eri']
    eri_model = ERIModel(
        model_type=eri_config['type'],
        **{k: v for k, v in eri_config.items() if k != 'type'}
    )

    eri_model.train(X_nsduh_train_scaled, y_nsduh_train, verbose=verbose)
    eri_results = eri_model.evaluate(X_nsduh_test_scaled, y_nsduh_test, verbose=verbose)

    # Get ERI scores
    eri_train_scores = eri_model.get_eri_scores(X_nsduh_train_scaled)
    eri_test_scores = eri_model.get_eri_scores(X_nsduh_test_scaled)

    results['eri_model'] = eri_model
    results['eri_results'] = eri_results
    results['eri_train_scores'] = eri_train_scores
    results['eri_test_scores'] = eri_test_scores

    # ========================================================================
    # STEP 6: Train BVI Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6/9: TRAINING BVI MODEL (Biological Vulnerability Index)")
    print("="*80)
    print("⏱️  Estimated time: 30-60 seconds\n")

    bvi_config = config['models']['bvi']
    bvi_model = BVIModel(
        model_type=bvi_config['type'],
        **{k: v for k, v in bvi_config.items() if k != 'type'}
    )

    bvi_model.train(X_eeg_train_scaled, y_eeg_train, verbose=verbose)
    bvi_results = bvi_model.evaluate(X_eeg_test_scaled, y_eeg_test, verbose=verbose)

    # Get BVI scores
    bvi_train_scores = bvi_model.get_bvi_scores(X_eeg_train_scaled)
    bvi_test_scores = bvi_model.get_bvi_scores(X_eeg_test_scaled)

    results['bvi_model'] = bvi_model
    results['bvi_results'] = bvi_results
    results['bvi_train_scores'] = bvi_train_scores
    results['bvi_test_scores'] = bvi_test_scores

    # ========================================================================
    # STEP 7: Risk Classification
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7/9: RISK CLASSIFICATION")
    print("="*80)
    print("⏱️  Estimated time: <10 seconds\n")

    risk_config = config['models']['risk']
    risk_classifier = RiskClassifier(
        eri_threshold=risk_config['eri_threshold'],
        bvi_threshold=risk_config['bvi_threshold']
    )

    # Align test samples (use minimum length)
    n_samples = min(len(eri_test_scores), len(bvi_test_scores))
    eri_test_aligned = eri_test_scores[:n_samples]
    bvi_test_aligned = bvi_test_scores[:n_samples]

    # Classify
    risk_categories = risk_classifier.classify(eri_test_aligned, bvi_test_aligned)
    risk_distribution = risk_classifier.get_distribution(risk_categories, verbose=verbose)

    results['risk_classifier'] = risk_classifier
    results['risk_categories'] = risk_categories
    results['risk_distribution'] = risk_distribution

    # ========================================================================
    # STEP 8: Visualization
    # ========================================================================
    if config['visualization']['save_figures']:
        print("\n" + "="*80)
        print("STEP 8/9: GENERATING VISUALIZATIONS")
        print("="*80)
        print("⏱️  Estimated time: 10-20 seconds\n")

        visualizer = Visualizer(
            save_figures=True,
            figures_path=config['visualization']['figures_path'],
            dpi=config['visualization']['dpi'],
            format=config['visualization']['format']
        )

        # Create comprehensive dashboard
        visualizer.create_dashboard(
            eri_test_aligned,
            bvi_test_aligned,
            risk_categories,
            eri_results['confusion_matrix'],
            bvi_results['confusion_matrix'],
            risk_config['eri_threshold'],
            risk_config['bvi_threshold']
        )

    # ========================================================================
    # STEP 9: Save Models
    # ========================================================================
    if config['training']['save_models']:
        print("\n" + "="*80)
        print("STEP 9/9: SAVING MODELS")
        print("="*80)
        print("⏱️  Estimated time: <10 seconds\n")

        models_path = config['training']['models_path']
        os.makedirs(models_path, exist_ok=True)

        # Save models
        eri_model.save(os.path.join(models_path, 'eri_model.pkl'))
        bvi_model.save(os.path.join(models_path, 'bvi_model.pkl'))
        risk_classifier.save(os.path.join(models_path, 'risk_classifier.pkl'))

        # Save scalers
        with open(os.path.join(models_path, 'scaler_nsduh.pkl'), 'wb') as f:
            pickle.dump(scaler_nsduh, f)
        print(f"✅ NSDUH scaler saved to {os.path.join(models_path, 'scaler_nsduh.pkl')}")

        with open(os.path.join(models_path, 'scaler_eeg.pkl'), 'wb') as f:
            pickle.dump(scaler_eeg, f)
        print(f"✅ EEG scaler saved to {os.path.join(models_path, 'scaler_eeg.pkl')}")

        # Save feature extractor
        with open(os.path.join(models_path, 'eeg_extractor.pkl'), 'wb') as f:
            pickle.dump(extractor, f)
        print(f"✅ EEG extractor saved to {os.path.join(models_path, 'eeg_extractor.pkl')}")

        # Save PCA (if used)
        if processed_data['pca'] is not None:
            with open(os.path.join(models_path, 'pca_nsduh.pkl'), 'wb') as f:
                pickle.dump(processed_data['pca'], f)
            print(f"✅ PCA model saved to {os.path.join(models_path, 'pca_nsduh.pkl')}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print("\n📊 Final Results Summary:")
    print(f"   ERI Model Test Accuracy: {eri_results['accuracy']:.4f}")
    print(f"   BVI Model Test Accuracy: {bvi_results['accuracy']:.4f}")
    print(f"   Risk Categories: {len(risk_distribution)} classes")
    print(f"   High Risk Samples: {risk_distribution['high_risk_count']} ({risk_distribution['high_risk_percentage']:.1f}%)")
    print("\n✅ All models saved to: " + config['training']['models_path'])
    print("="*80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train PEBS system (ERI + BVI + Risk Classification)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("   Using default config.yaml")
        args.config = 'config.yaml'

    config = load_config(args.config)

    # Check if data exists
    if not os.path.exists(config['data']['nsduh_path']):
        print("\n❌ ERROR: NSDUH data not found!")
        print(f"   Expected location: {config['data']['nsduh_path']}")
        print("   Please run: python download_data.py")
        return

    if not os.path.exists(config['data']['smni_train_path']):
        print("\n❌ ERROR: SMNI training data not found!")
        print(f"   Expected location: {config['data']['smni_train_path']}")
        print("   Please run: python download_data.py")
        return

    # Train system
    try:
        results = train_pebs_system(config, verbose=not args.quiet)
        print("\n✅ Training pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return

    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
