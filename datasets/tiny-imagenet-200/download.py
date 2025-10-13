#!/usr/bin/env python3
"""
Tiny-ImageNet-200 Dataset Download Script

This script downloads and prepares the Tiny-ImageNet-200 dataset for the 
XAI robustness benchmark experiments.

Dataset Details:
- 200 classes (subset of ImageNet)
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
- Image size: 64×64 RGB
- Total size: ~250MB

Usage:
    python download.py [--data-dir DATA_DIR]

Note: This dataset is used for intermediate complexity evaluation between
      CIFAR-10 and MS-COCO-2017 in our robustness benchmark.
"""

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path


def download_file(url, destination):
    """Download file with progress bar."""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
    print()  # New line after progress


def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def organize_validation_images(data_dir):
    """
    Organize Tiny-ImageNet validation images into class folders.
    
    By default, validation images are in a flat directory. This function
    organizes them into class-specific folders for easier loading.
    """
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    
    if not os.path.exists(val_annotations):
        print("Validation annotations not found. Skipping organization.")
        return
    
    print("Organizing validation images into class folders...")
    
    # Read annotations
    with open(val_annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_id = parts[1]
            
            # Create class directory if not exists
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            # Move image to class directory
            src = os.path.join(val_dir, 'images', img_name)
            dst = os.path.join(class_dir, img_name)
            
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)
    
    print("Validation images organized!")


def verify_dataset(data_dir):
    """Verify dataset integrity."""
    dataset_dir = os.path.join(data_dir, 'tiny-imagenet-200')
    
    checks = {
        'train': os.path.join(dataset_dir, 'train'),
        'val': os.path.join(dataset_dir, 'val'),
        'test': os.path.join(dataset_dir, 'test'),
        'wnids.txt': os.path.join(dataset_dir, 'wnids.txt'),
        'words.txt': os.path.join(dataset_dir, 'words.txt'),
    }
    
    print("\nVerifying dataset...")
    all_good = True
    for name, path in checks.items():
        if os.path.exists(path):
            print(f"✓ {name} found")
        else:
            print(f"✗ {name} missing")
            all_good = False
    
    if all_good:
        # Count classes in train
        train_dir = checks['train']
        n_classes = len([d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
        print(f"\n✓ Dataset verified! Found {n_classes} training classes.")
        print(f"✓ Dataset location: {dataset_dir}")
    else:
        print("\n✗ Dataset verification failed!")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(
        description='Download Tiny-ImageNet-200 dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to download dataset to (default: ./data)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if dataset already exists'
    )
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URL
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = data_dir / 'tiny-imagenet-200.zip'
    dataset_path = data_dir / 'tiny-imagenet-200'
    
    # Check if dataset already exists
    if dataset_path.exists() and args.skip_download:
        print(f"Dataset already exists at {dataset_path}")
        verify_dataset(data_dir)
        return
    
    # Download dataset
    if not zip_path.exists():
        print(f"Downloading Tiny-ImageNet-200 from {url}")
        print("This may take a few minutes (~250MB)...")
        download_file(url, str(zip_path))
    else:
        print(f"Using existing zip file: {zip_path}")
    
    # Extract dataset
    if not dataset_path.exists():
        extract_zip(str(zip_path), str(data_dir))
    else:
        print("Dataset already extracted.")
    
    # Organize validation images
    organize_validation_images(str(data_dir))
    
    # Verify dataset
    if verify_dataset(str(data_dir)):
        print("\n" + "="*60)
        print("SUCCESS! Tiny-ImageNet-200 is ready to use!")
        print("="*60)
        print(f"\nDataset location: {dataset_path}")
        print("\nYou can now run experiments with:")
        print("  python scripts/run_experiments.py --dataset tiny-imagenet-200")
    
    # Optional: Remove zip file to save space
    if zip_path.exists():
        response = input("\nRemove zip file to save space? (y/n): ")
        if response.lower() == 'y':
            os.remove(zip_path)
            print("Zip file removed.")


if __name__ == '__main__':
    main()

