"""
Data download script for PEBS project.
Downloads NSDUH and SMNI datasets from public sources.
"""

import os
import argparse


def download_nsduh():
    """
    Download NSDUH dataset.

    Note: NSDUH data requires manual download from SAMHDA website.
    """
    print("="*80)
    print("NSDUH DATASET")
    print("="*80)
    print("\nThe NSDUH (National Survey on Drug Use and Health) dataset requires")
    print("manual download due to data use agreements.\n")
    print("📥 Download Instructions:")
    print("   1. Visit: https://www.datafiles.samhsa.gov/dataset/nsduh-2002-2018-ds0001")
    print("   2. Accept the data use agreement")
    print("   3. Download: NSDUH_2002_2018_Tab.tsv")
    print("   4. Place the file in: data/raw/NSDUH_2002_2018_Tab.tsv\n")
    print("💡 Alternative: If you have the data file, manually place it in data/raw/")
    print("="*80 + "\n")


def download_smni():
    """
    Download SMNI EEG dataset.

    Note: SMNI data requires manual download from UCI repository.
    """
    print("="*80)
    print("SMNI EEG DATASET")
    print("="*80)
    print("\nThe SMNI (EEG) dataset can be downloaded from UCI Machine Learning Repository.\n")
    print("📥 Download Instructions:")
    print("   1. Visit: https://archive.ics.uci.edu/ml/datasets/EEG+Database")
    print("   2. Download the dataset")
    print("   3. Extract files to:")
    print("      - data/raw/SMNI_CMI_TRAIN/ (468 files: Data1.csv to Data468.csv)")
    print("      - data/raw/SMNI_CMI_TEST/ (480 files: Data1.csv to Data480.csv)\n")
    print("💡 Alternative download source:")
    print("   - Kaggle: https://www.kaggle.com/datasets/nnair25/Alcoholics")
    print("   - GitHub: https://github.com/datasets (search for 'EEG alcoholic')\n")
    print("="*80 + "\n")


def setup_directories():
    """Create necessary directories."""
    directories = [
        'data/raw',
        'data/raw/SMNI_CMI_TRAIN',
        'data/raw/SMNI_CMI_TEST',
        'data/processed',
        'models',
        'figures'
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    print()


def check_data_status():
    """Check which datasets are already available."""
    print("="*80)
    print("DATA STATUS CHECK")
    print("="*80 + "\n")

    # Check NSDUH
    nsduh_path = 'data/raw/NSDUH_2002_2018_Tab.tsv'
    if os.path.exists(nsduh_path):
        size_mb = os.path.getsize(nsduh_path) / (1024 * 1024)
        print(f"✅ NSDUH dataset found: {nsduh_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ NSDUH dataset not found: {nsduh_path}")

    # Check SMNI Train
    smni_train_path = 'data/raw/SMNI_CMI_TRAIN'
    if os.path.exists(smni_train_path):
        train_files = len([f for f in os.listdir(smni_train_path) if f.endswith('.csv')])
        print(f"✅ SMNI TRAIN dataset found: {train_files} files in {smni_train_path}")
    else:
        print(f"❌ SMNI TRAIN dataset not found: {smni_train_path}")

    # Check SMNI Test
    smni_test_path = 'data/raw/SMNI_CMI_TEST'
    if os.path.exists(smni_test_path):
        test_files = len([f for f in os.listdir(smni_test_path) if f.endswith('.csv')])
        print(f"✅ SMNI TEST dataset found: {test_files} files in {smni_test_path}")
    else:
        print(f"❌ SMNI TEST dataset not found: {smni_test_path}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download PEBS datasets (NSDUH and SMNI EEG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py              # Setup directories and show instructions
  python download_data.py --check      # Check data status only
  python download_data.py --help       # Show this help message

Note:
  Both datasets require manual download due to licensing and size.
  This script will guide you through the download process.
        """
    )

    parser.add_argument('--check', action='store_true',
                       help='Check data status without showing download instructions')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PEBS DATA DOWNLOAD SCRIPT")
    print("="*80 + "\n")

    # Setup directories
    setup_directories()

    # Check status
    check_data_status()

    # Show download instructions unless --check is used
    if not args.check:
        download_nsduh()
        download_smni()

        print("="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Download the datasets following the instructions above")
        print("2. Place the files in the correct directories")
        print("3. Run: python download_data.py --check  (to verify)")
        print("4. Run: python train.py  (to start training)\n")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
