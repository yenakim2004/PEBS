"""
Prediction script for PEBS system.
Makes predictions on new samples using trained models.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

from pebs.models.eri_model import ERIModel
from pebs.models.bvi_model import BVIModel
from pebs.models.risk_classifier import RiskClassifier


def load_models(models_path='models/'):
    """
    Load all trained models and scalers.

    Args:
        models_path: Path to models directory

    Returns:
        Dictionary with all loaded models and scalers
    """
    print("📦 Loading trained models...")

    models = {}

    # Load ERI model
    eri_path = os.path.join(models_path, 'eri_model.pkl')
    models['eri_model'] = ERIModel.load(eri_path)

    # Load BVI model
    bvi_path = os.path.join(models_path, 'bvi_model.pkl')
    models['bvi_model'] = BVIModel.load(bvi_path)

    # Load Risk Classifier
    risk_path = os.path.join(models_path, 'risk_classifier.pkl')
    models['risk_classifier'] = RiskClassifier.load(risk_path)

    # Load scalers
    with open(os.path.join(models_path, 'scaler_nsduh.pkl'), 'rb') as f:
        models['scaler_nsduh'] = pickle.load(f)
    print(f"✅ NSDUH scaler loaded")

    with open(os.path.join(models_path, 'scaler_eeg.pkl'), 'rb') as f:
        models['scaler_eeg'] = pickle.load(f)
    print(f"✅ EEG scaler loaded")

    # Load EEG feature extractor
    with open(os.path.join(models_path, 'eeg_extractor.pkl'), 'rb') as f:
        models['eeg_extractor'] = pickle.load(f)
    print(f"✅ EEG extractor loaded")

    print("\n✅ All models loaded successfully\n")

    return models


def predict_single_sample(nsduh_features, eeg_dataframe, models, verbose=True):
    """
    Predict risk for a single sample.

    Args:
        nsduh_features: 1D array or DataFrame of NSDUH survey features
        eeg_dataframe: DataFrame with EEG signal data
        models: Dictionary with all trained models and scalers
        verbose: Print prediction details

    Returns:
        Dictionary with prediction results
    """
    # Extract features from EEG
    if verbose:
        print("🔬 Extracting EEG features...")

    eeg_features = models['eeg_extractor'].extract_features_from_file(eeg_dataframe)

    # Prepare NSDUH features
    if isinstance(nsduh_features, pd.DataFrame):
        nsduh_features = nsduh_features.values

    # Ensure correct shape
    if nsduh_features.ndim == 1:
        nsduh_features = nsduh_features.reshape(1, -1)
    if eeg_features.ndim == 1:
        eeg_features = eeg_features.reshape(1, -1)

    # Standardize features
    if verbose:
        print("📏 Standardizing features...")

    nsduh_scaled = models['scaler_nsduh'].transform(nsduh_features)
    eeg_scaled = models['scaler_eeg'].transform(eeg_features)

    # Get ERI score
    if verbose:
        print("🧮 Calculating ERI score...")

    eri_score = models['eri_model'].get_eri_scores(nsduh_scaled)[0]

    # Get BVI score
    if verbose:
        print("🧮 Calculating BVI score...")

    bvi_score = models['bvi_model'].get_bvi_scores(eeg_scaled)[0]

    # Classify risk
    if verbose:
        print("🎯 Classifying risk...")

    risk_result = models['risk_classifier'].classify_single(eri_score, bvi_score)

    if verbose:
        print_prediction_report(risk_result)

    return risk_result


def predict_batch(nsduh_features, eeg_features, models, verbose=True):
    """
    Predict risk for multiple samples.

    Args:
        nsduh_features: 2D array of NSDUH survey features
        eeg_features: 2D array of EEG features (already extracted)
        models: Dictionary with all trained models and scalers
        verbose: Print prediction summary

    Returns:
        Dictionary with batch prediction results
    """
    if verbose:
        print(f"\n🔮 Predicting risk for {len(nsduh_features)} samples...")

    # Standardize features
    nsduh_scaled = models['scaler_nsduh'].transform(nsduh_features)
    eeg_scaled = models['scaler_eeg'].transform(eeg_features)

    # Get scores
    eri_scores = models['eri_model'].get_eri_scores(nsduh_scaled)
    bvi_scores = models['bvi_model'].get_bvi_scores(eeg_scaled)

    # Classify risks
    risk_categories = models['risk_classifier'].classify(eri_scores, bvi_scores)

    if verbose:
        print("\n✅ Batch prediction completed")
        print(f"\n📊 Results Summary:")
        distribution = models['risk_classifier'].get_distribution(risk_categories, verbose=False)
        for i, info in distribution.items():
            print(f"   {info['name']:25s}: {info['count']:4d} ({info['percentage']:5.1f}%)")

    return {
        'eri_scores': eri_scores,
        'bvi_scores': bvi_scores,
        'risk_categories': risk_categories,
        'distribution': distribution
    }


def print_prediction_report(result):
    """
    Print formatted prediction report.

    Args:
        result: Prediction result dictionary
    """
    print("\n" + "="*80)
    print("PEBS RISK ASSESSMENT REPORT")
    print("="*80)

    print(f"\n🎯 Risk Scores:")
    print(f"   Environmental Risk Index (ERI): {result['eri_score']:.4f}")
    print(f"   Biological Vulnerability Index (BVI): {result['bvi_score']:.4f}")

    print(f"\n📊 Risk Classification:")
    print(f"   Category: {result['category']} - {result['name']}")

    print(f"\n💡 Interpretation:")
    print(f"   {result['description']}")

    print(f"\n🔍 Risk Factors:")
    print(f"   Environmental Risk: {'⚠️  HIGH' if result['eri_high'] else '✅ Low'}")
    print(f"   Biological Risk:    {'⚠️  HIGH' if result['bvi_high'] else '✅ Low'}")

    # Recommendations based on category
    print(f"\n📋 Recommendations:")
    if result['category'] == 0:
        print("   ✅ Continue healthy lifestyle and preventive measures.")
    elif result['category'] == 1:
        print("   ⚠️  Focus on environmental interventions:")
        print("      - Address social and environmental risk factors")
        print("      - Consider counseling and social support")
    elif result['category'] == 2:
        print("   ⚠️  Focus on biological/genetic factors:")
        print("      - Consider biological vulnerability assessment")
        print("      - Evaluate family history and genetic predisposition")
    else:  # category == 3
        print("   ⚠️⚠️  COMPREHENSIVE INTERVENTION RECOMMENDED:")
        print("      - Both environmental and biological risk factors present")
        print("      - Multifaceted treatment approach advised")
        print("      - Consider professional assessment and intervention")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained PEBS models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from CSV files
  python predict.py --nsduh sample_survey.csv --eeg sample_eeg.csv

  # Batch prediction
  python predict.py --nsduh batch_surveys.csv --eeg batch_eeg_features.csv --batch

  # Use custom models path
  python predict.py --nsduh sample.csv --eeg sample.csv --models-path ./my_models/

Note:
  - NSDUH file should contain survey features (same format as training data)
  - EEG file should contain either:
    * Raw EEG signal data (for single prediction)
    * Pre-extracted EEG features (for batch prediction with --batch flag)
        """
    )

    parser.add_argument('--nsduh', type=str, required=True,
                       help='Path to NSDUH features CSV file')
    parser.add_argument('--eeg', type=str, required=True,
                       help='Path to EEG data CSV file')
    parser.add_argument('--models-path', type=str, default='models/',
                       help='Path to trained models directory (default: models/)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch prediction mode (EEG file contains pre-extracted features)')
    parser.add_argument('--output', type=str,
                       help='Save predictions to CSV file (optional)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.nsduh):
        print(f"❌ ERROR: NSDUH file not found: {args.nsduh}")
        return

    if not os.path.exists(args.eeg):
        print(f"❌ ERROR: EEG file not found: {args.eeg}")
        return

    # Check if models exist
    required_models = [
        'eri_model.pkl', 'bvi_model.pkl', 'risk_classifier.pkl',
        'scaler_nsduh.pkl', 'scaler_eeg.pkl', 'eeg_extractor.pkl'
    ]

    for model_file in required_models:
        model_path = os.path.join(args.models_path, model_file)
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Required model file not found: {model_path}")
            print("   Please run: python train.py")
            return

    # Load models
    try:
        models = load_models(args.models_path)
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        return

    # Load input data
    print("📂 Loading input data...")
    nsduh_data = pd.read_csv(args.nsduh)
    eeg_data = pd.read_csv(args.eeg)
    print(f"   NSDUH shape: {nsduh_data.shape}")
    print(f"   EEG shape: {eeg_data.shape}\n")

    # Make predictions
    try:
        if args.batch:
            # Batch prediction
            result = predict_batch(
                nsduh_data.values,
                eeg_data.values,
                models,
                verbose=not args.quiet
            )

            # Save results if requested
            if args.output:
                output_df = pd.DataFrame({
                    'eri_score': result['eri_scores'],
                    'bvi_score': result['bvi_scores'],
                    'risk_category': result['risk_categories'],
                    'risk_name': [models['risk_classifier'].get_category_name(cat)
                                  for cat in result['risk_categories']]
                })
                output_df.to_csv(args.output, index=False)
                print(f"\n💾 Predictions saved to: {args.output}")

        else:
            # Single prediction
            result = predict_single_sample(
                nsduh_data,
                eeg_data,
                models,
                verbose=not args.quiet
            )

            # Save result if requested
            if args.output:
                output_df = pd.DataFrame([result])
                output_df.to_csv(args.output, index=False)
                print(f"\n💾 Prediction saved to: {args.output}")

        print("\n✅ Prediction completed successfully!")

    except Exception as e:
        print(f"\n❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
