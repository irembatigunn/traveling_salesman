"""
Kaggle Dataset Downloader
=========================
Standalone script to download the TSPLIB dataset from Kaggle.

Prerequisites:
    1. Install kagglehub: pip install kagglehub
    2. Configure Kaggle credentials:
       - Sign in to kaggle.com
       - Go to Account -> API -> Create New Token
       - This downloads a kaggle.json file
       - Place it in ~/.kaggle/kaggle.json (Linux/Mac)
       - Or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)

Usage:
    python data/download_dataset.py
"""

import os
import sys


def download():
    """Download the TSPLIB dataset from Kaggle and print its location."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Install it with: pip install kagglehub")
        sys.exit(1)

    print("Downloading TSPLIB dataset from Kaggle...")
    print("(This requires valid Kaggle API credentials)")
    print()

    try:
        path = kagglehub.dataset_download(
            "ziya07/traveling-salesman-problem-tsplib-dataset"
        )
        print(f"[SUCCESS] Dataset downloaded to: {path}")
        print()

        # List downloaded files
        print("Files in dataset:")
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        print()
        print("Make sure your Kaggle credentials are configured.")
        print("See: https://github.com/Kaggle/kagglehub#authenticate")
        sys.exit(1)


if __name__ == "__main__":
    download()
