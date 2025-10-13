#!/usr/bin/env python3
"""
Base XAI Evaluator Class
Ensures consistent evaluation across all XAI methods
"""

import os
import json
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from .metrics import UnifiedMetrics
from .corruptions import apply_corruption
from .data_loader import UnifiedDataLoader


class BaseXAIEvaluator(ABC):
    """Base class for all XAI method evaluators"""
    
    def __init__(self, config_path: str):
        """
        Initialize evaluator with configuration
        
        Args:
            config_path: Path to experiment configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.metrics = UnifiedMetrics()
        self.data_loader = UnifiedDataLoader(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.models = self._load_models()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load standard and robust models"""
        models = {}
        
        # Standard ResNet-50
        from torchvision import models as tv_models
        models['standard'] = tv_models.resnet50(pretrained=True)
        models['standard'].eval()
        models['standard'].to(self.device)
        
        # Robust ResNet-50
        from robustbench.utils import load_model
        models['robust'] = load_model(
            model_name='Salman2020Do_50_2',
            dataset='imagenet',
            threat_model='corruptions'
        )
        models['robust'].eval()
        models['robust'].to(self.device)
        
        return models
    
    @abstractmethod
    def get_attribution(self, image: torch.Tensor, model: torch.nn.Module, 
                       target_class: int = None) -> np.ndarray:
        """
        Get attribution/explanation for the image
        To be implemented by each XAI method
        
        Args:
            image: Input image tensor
            model: Model to explain
            target_class: Target class for explanation
            
        Returns:
            Attribution map as numpy array
        """
        raise NotImplementedError
    
    def evaluate_robustness(self, dataset_name: str, model_type: str = 'standard',
                          save_attributions: bool = False) -> Dict[str, Any]:
        """
        Main evaluation loop - consistent across all methods
        
        Args:
            dataset_name: Name of dataset to evaluate ('tiny-imagenet-200', 'cifar-10', 'ms-coco-2017')
            model_type: 'standard' or 'robust'
            save_attributions: Whether to save attribution maps
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.__class__.__name__} on {dataset_name} with {model_type} model")
        print(f"{'='*60}\n")
        
        model = self.models[model_type]
        results = {}
        
        # Get image list for this dataset
        image_list = self.data_loader.get_image_list(dataset_name)
        print(f"Found {len(image_list)} images for {dataset_name}")
        
        # Process each image
        for img_idx, img_path in enumerate(tqdm(image_list, desc="Processing images")):
            img_results = self._evaluate_single_image(
                img_path, model, dataset_name, save_attributions
            )
            
            # Store results
            rel_path = os.path.relpath(img_path, self.config['datasets'][dataset_name]['path'])
            results[rel_path] = img_results
            
            # Save intermediate results every 50 images
            if (img_idx + 1) % 50 == 0:
                self._save_intermediate_results(results, dataset_name, model_type, img_idx + 1)
        
        # Save final results
        output_file = self._get_output_filename(dataset_name, model_type)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    
    def _evaluate_single_image(self, img_path: str, model: torch.nn.Module,
                             dataset_name: str, save_attributions: bool) -> Dict[str, Any]:
        """Evaluate robustness for a single image"""
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        # Handle different image sizes
        if dataset_name == 'cifar-10':
            # CIFAR-10 images are 32x32, need special handling
            image = image.resize((224, 224), Image.BILINEAR)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get original prediction and attribution
        with torch.no_grad():
            output = model(image_tensor)
            pred_class = output.argmax(1).item()
            pred_conf = torch.softmax(output, dim=1).max().item()
        
        orig_attr = self.get_attribution(image_tensor, model, pred_class)
        
        # Store results for each corruption
        corruption_results = {}
        
        # Apply each corruption type and severity
        for corruption in self.config['corruptions']['types']:
            corruption_results[corruption] = {}
            
            for severity in self.config['corruptions']['severities']:
                # Apply corruption to PIL image
                corrupted_image = apply_corruption(image, corruption, severity)
                
                # Convert to tensor
                corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                
                # Get corrupted prediction and attribution
                with torch.no_grad():
                    corrupted_output = model(corrupted_tensor)
                    corrupted_pred = corrupted_output.argmax(1).item()
                    corrupted_conf = torch.softmax(corrupted_output, dim=1).max().item()
                
                corrupted_attr = self.get_attribution(corrupted_tensor, model, pred_class)
                
                # Calculate all metrics
                metrics = self.metrics.calculate_all_metrics(
                    original_attr=orig_attr,
                    corrupted_attr=corrupted_attr,
                    original_pred=pred_class,
                    corrupted_pred=corrupted_pred,
                    original_conf=pred_conf,
                    corrupted_conf=corrupted_conf,
                    original_output=output.cpu().numpy(),
                    corrupted_output=corrupted_output.cpu().numpy()
                )
                
                corruption_results[corruption][str(severity)] = {
                    'metrics': metrics,
                    'prediction_changed': pred_class != corrupted_pred,
                    'confidence_diff': pred_conf - corrupted_conf
                }
                
                # Save attributions if requested
                if save_attributions:
                    self._save_attribution(
                        corrupted_attr, img_path, corruption, severity,
                        dataset_name, model.name
                    )
        
        return {
            'original_prediction': pred_class,
            'original_confidence': pred_conf,
            'corruptions': corruption_results
        }
    
    def _save_intermediate_results(self, results: Dict, dataset_name: str,
                                 model_type: str, num_processed: int):
        """Save intermediate results for recovery"""
        temp_file = self._get_output_filename(dataset_name, model_type, temp=True)
        temp_file = temp_file.replace('.json', f'_temp_{num_processed}.json')
        
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved intermediate results ({num_processed} images) to {temp_file}")
    
    def _get_output_filename(self, dataset_name: str, model_type: str, temp: bool = False) -> str:
        """Get standardized output filename"""
        method_name = self.__class__.__name__.lower().replace('evaluator', '')
        
        output_dir = os.path.join(
            self.config['output_dir'],
            dataset_name,
            method_name
        )
        os.makedirs(output_dir, exist_ok=True)
        
        if temp:
            filename = f"{method_name}_robustness_{model_type}_temp.json"
        else:
            filename = f"{method_name}_robustness_{model_type}_results.json"
        
        return os.path.join(output_dir, filename)
    
    def _save_attribution(self, attribution: np.ndarray, img_path: str,
                        corruption: str, severity: int, dataset_name: str,
                        model_name: str):
        """Save attribution map for visualization"""
        # Implementation depends on specific needs
        pass
    
    def run_all_datasets(self, model_type: str = 'standard'):
        """Run evaluation on all configured datasets"""
        all_results = {}
        
        for dataset in self.config['datasets']:
            dataset_results = self.evaluate_robustness(
                dataset_name=dataset['name'],
                model_type=model_type,
                save_attributions=False
            )
            all_results[dataset['name']] = dataset_results
        
        return all_results