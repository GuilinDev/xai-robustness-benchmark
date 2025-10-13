#!/usr/bin/env python3
"""
Unified Data Loader
Handles loading images from all three datasets with consistent interface
"""

import os
import json
from typing import List, Dict, Any


class UnifiedDataLoader:
    """Unified data loader for all datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        # Use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_paths = {
            'tiny-imagenet-200': os.path.join(base_dir, 'datasets/tiny-imagenet-200'),
            'cifar-10': os.path.join(base_dir, 'datasets/cifar-10'),
            'ms-coco-2017': os.path.join(base_dir, 'datasets/ms-coco-2017')
        }
        
    def get_image_list(self, dataset_name: str) -> List[str]:
        """
        Get list of image paths for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of absolute image paths
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.dataset_paths[dataset_name]
        list_file = os.path.join(dataset_dir, 'lists', 'selected_images.txt')
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Image list not found: {list_file}")
        
        # Read image list
        image_paths = []
        with open(list_file, 'r') as f:
            for line in f:
                img_name = line.strip()
                if img_name:
                    img_path = os.path.join(dataset_dir, 'images', img_name)
                    if os.path.exists(img_path):
                        image_paths.append(img_path)
                    else:
                        print(f"Warning: Image not found: {img_path}")
        
        return image_paths
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.dataset_paths[dataset_name]
        image_list = self.get_image_list(dataset_name)
        
        info = {
            'name': dataset_name,
            'path': dataset_dir,
            'num_images': len(image_list),
            'image_format': self._get_image_format(dataset_name)
        }
        
        # Add dataset-specific information
        if dataset_name == 'cifar-10':
            info['image_size'] = (32, 32)
            info['num_classes'] = 10
        elif dataset_name == 'tiny-imagenet-200':
            info['image_size'] = (64, 64)
            info['num_classes'] = 200
        elif dataset_name == 'ms-coco-2017':
            info['image_size'] = 'variable'
            info['num_classes'] = 80
        
        return info
    
    def _get_image_format(self, dataset_name: str) -> str:
        """Get image file format for dataset"""
        if dataset_name == 'cifar-10':
            return 'png'
        else:
            return 'jpg'
    
    def verify_all_datasets(self) -> Dict[str, bool]:
        """
        Verify that all datasets are properly downloaded
        
        Returns:
            Dictionary mapping dataset names to verification status
        """
        status = {}
        
        for dataset_name in self.dataset_paths:
            try:
                image_list = self.get_image_list(dataset_name)
                expected_count = 496 if dataset_name == 'tiny-imagenet-200' else 500
                
                if len(image_list) >= expected_count - 5:  # Allow small deviation
                    status[dataset_name] = True
                    print(f"✓ {dataset_name}: {len(image_list)} images found")
                else:
                    status[dataset_name] = False
                    print(f"✗ {dataset_name}: Only {len(image_list)} images found (expected ~{expected_count})")
            except Exception as e:
                status[dataset_name] = False
                print(f"✗ {dataset_name}: Error - {str(e)}")
        
        return status