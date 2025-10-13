"""
Base Evaluator Class for XAI Robustness Evaluation
This provides a consistent framework that all XAI methods must follow.
"""

import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging
from datetime import datetime

from torchvision import transforms
from .metrics import UnifiedMetrics
from .corruptions import apply_corruption


class BaseXAIEvaluator(ABC):
    """
    Abstract base class for all XAI method evaluators.
    Ensures consistent evaluation protocol across different methods.
    """
    
    def __init__(self, config_path: str, method_name: str, model_type: str = "standard"):
        """
        Initialize the base evaluator.
        
        Args:
            config_path: Path to experiment configuration file
            method_name: Name of the XAI method (e.g., 'gradcam', 'shap')
            model_type: Type of model to use ('standard' or 'robust')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.method_name = method_name
        self.model_type = model_type
        self.method_config = self.config['methods'][method_name]
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.device = torch.device(self.config['computation']['device'])
        self.metrics = UnifiedMetrics(self.config['evaluation']['thresholds'])
        
        # Load model
        self.model = self._load_model()
        
        # Set up data transforms
        self.transform = self._get_transform()
        
        # Load sample list
        self.sample_list = self._load_sample_list()
        
        # Initialize results storage
        self.results = {}
        self.checkpoint_file = None
        
        self.logger.info(f"Initialized {method_name} evaluator for {model_type} model")
        
    def evaluate_robustness(self, dataset_name: str, save_attributions: bool = False) -> Dict:
        """
        Main evaluation loop - consistent across all methods.
        
        Args:
            dataset_name: Name of dataset to evaluate
            save_attributions: Whether to save attribution maps
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info(f"Starting robustness evaluation on {dataset_name}")
        
        # Set up checkpointing
        checkpoint_dir = os.path.join(self.config['experiment']['output_dir'], 
                                     'checkpoints', self.method_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, 
                                           f"{dataset_name}_{self.model_type}_checkpoint.json")
        
        # Load checkpoint if exists
        start_idx = self._load_checkpoint()
        
        # Get dataset configuration
        dataset_config = next(d for d in self.config['datasets'] if d['name'] == dataset_name)
        
        # Process images
        for idx, image_path in enumerate(tqdm(self.sample_list[start_idx:], 
                                             desc=f"Evaluating {self.method_name}")):
            actual_idx = start_idx + idx
            
            try:
                # Load and preprocess image
                image = self._load_image(image_path, dataset_config)
                
                # Get original attribution and prediction
                orig_attr, orig_pred, orig_probs, orig_top5 = self._get_attribution_and_prediction(image)
                
                # Initialize results for this image
                image_results = {}
                
                # Evaluate each corruption
                for corruption in self.config['corruptions']['all_types']:
                    corruption_results = {}
                    
                    for severity in self.config['corruptions']['severities']:
                        # Apply corruption
                        corrupted_image = apply_corruption(image, corruption, severity)
                        
                        # Get corrupted attribution and prediction
                        corr_attr, corr_pred, corr_probs, corr_top5 = \
                            self._get_attribution_and_prediction(corrupted_image)
                        
                        # Calculate metrics
                        metrics = self.metrics.calculate_all_metrics(
                            orig_attr, corr_attr,
                            orig_pred, corr_pred,
                            orig_probs, corr_probs,
                            orig_top5, corr_top5
                        )
                        
                        # Store results
                        corruption_results[str(severity)] = metrics
                        
                        # Optionally save attributions
                        if save_attributions and actual_idx < 5:  # Save first 5 examples
                            self._save_attribution_visualization(
                                image, corrupted_image,
                                orig_attr, corr_attr,
                                corruption, severity,
                                dataset_name
                            )
                    
                    image_results[corruption] = corruption_results
                
                # Store results
                self.results[image_path] = image_results
                
                # Checkpoint periodically
                if (actual_idx + 1) % self.config['computation']['checkpoint_frequency'] == 0:
                    self._save_checkpoint(actual_idx + 1)
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Save final results
        self._save_results(dataset_name)
        
        # Clean up checkpoint
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        self.logger.info(f"Evaluation complete. Processed {len(self.results)} images.")
        
        return self.results
    
    @abstractmethod
    def get_attribution(self, image: torch.Tensor) -> np.ndarray:
        """
        Get attribution map for an image. Must be implemented by each XAI method.
        
        Args:
            image: Preprocessed image tensor (C, H, W)
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        raise NotImplementedError("Each XAI method must implement get_attribution")
    
    def _get_attribution_and_prediction(self, image: PIL.Image) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Get attribution and prediction for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (attribution, prediction, probabilities, top5)
        """
        # Convert to tensor
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            top5 = torch.topk(output, k=5, dim=1)[1][0].cpu().numpy()
        
        # Get attribution
        attribution = self.get_attribution(image_tensor[0])
        
        return attribution, pred, probs[0].cpu().numpy(), top5
    
    def _load_model(self) -> torch.nn.Module:
        """Load the appropriate model based on configuration."""
        if self.model_type == "standard":
            import torchvision.models as models
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            from robustbench.utils import load_model
            model = load_model(
                model_name=self.config['models']['robust']['name'],
                dataset=self.config['models']['robust']['dataset'],
                threat_model=self.config['models']['robust']['threat_model']
            )
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the appropriate transform for the model."""
        # Standard ImageNet transform
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_sample_list(self) -> List[str]:
        """Load the sample list for the current dataset."""
        # For now, return empty list - will be implemented with actual sample lists
        return []
    
    def _load_image(self, image_path: str, dataset_config: Dict) -> PIL.Image:
        """Load an image from disk."""
        full_path = os.path.join(dataset_config['data_root'], image_path)
        return Image.open(full_path).convert('RGB')
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.method_name}_{self.model_type}_{timestamp}.log")
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"{self.method_name}_evaluator")
    
    def _save_checkpoint(self, processed_count: int):
        """Save checkpoint to resume interrupted evaluation."""
        checkpoint_data = {
            'processed_count': processed_count,
            'results': self.results,
            'method': self.method_name,
            'model_type': self.model_type
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        self.logger.info(f"Checkpoint saved at image {processed_count}")
    
    def _load_checkpoint(self) -> int:
        """Load checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.results = checkpoint_data['results']
            start_idx = checkpoint_data['processed_count']
            
            self.logger.info(f"Resuming from checkpoint at image {start_idx}")
            return start_idx
        
        return 0
    
    def _save_results(self, dataset_name: str):
        """Save final results to disk."""
        output_dir = os.path.join(self.config['experiment']['output_dir'], 
                                 'raw', self.method_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 
                                  f"{dataset_name}_{self.model_type}_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def _save_attribution_visualization(self, 
                                      original_image: PIL.Image,
                                      corrupted_image: PIL.Image,
                                      original_attr: np.ndarray,
                                      corrupted_attr: np.ndarray,
                                      corruption: str,
                                      severity: int,
                                      dataset_name: str):
        """Save visualization of attributions (optional)."""
        # Implementation depends on visualization requirements
        pass


def create_evaluator(method_name: str, config_path: str, model_type: str) -> BaseXAIEvaluator:
    """
    Factory function to create the appropriate evaluator.
    
    Args:
        method_name: Name of XAI method
        config_path: Path to configuration file
        model_type: Type of model ('standard' or 'robust')
        
    Returns:
        Configured evaluator instance
    """
    # Import method-specific evaluators
    if method_name == 'gradcam':
        from ..scripts.gradcam.gradcam_evaluator import GradCAMEvaluator
        return GradCAMEvaluator(config_path, model_type)
    elif method_name == 'shap':
        from ..scripts.shap.shap_evaluator import SHAPEvaluator
        return SHAPEvaluator(config_path, model_type)
    # Add other methods...
    else:
        raise ValueError(f"Unknown method: {method_name}")


if __name__ == "__main__":
    print("Base evaluator module loaded successfully.")