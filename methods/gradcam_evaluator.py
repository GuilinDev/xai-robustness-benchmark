#!/usr/bin/env python3
"""
GradCAM Evaluator
Implements GradCAM-specific attribution generation
"""

import torch
import numpy as np
from captum.attr import LayerGradCam

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.base_xai_evaluator import BaseXAIEvaluator


class GradCAMEvaluator(BaseXAIEvaluator):
    """GradCAM-specific evaluator"""
    
    def __init__(self, config_path: str):
        """Initialize GradCAM evaluator"""
        super().__init__(config_path)
        
        # GradCAM-specific setup
        self.target_layers = {
            'standard': 'layer4.2.conv3',  # Last conv layer for ResNet-50
            'robust': 'layer4.2.conv3'
        }
        
    def get_attribution(self, image: torch.Tensor, model: torch.nn.Module, 
                       target_class: int = None) -> np.ndarray:
        """
        Get GradCAM attribution for the image
        
        Args:
            image: Input image tensor (1, 3, 224, 224)
            model: Model to explain
            target_class: Target class for explanation
            
        Returns:
            Attribution map as numpy array (224, 224)
        """
        # Determine which model type we're using
        model_type = 'robust' if hasattr(model, 'model') else 'standard'
        
        # Get the target layer
        if model_type == 'robust':
            # RobustBench models wrap the actual model
            target_layer = model.model.layer4[2].conv3
        else:
            target_layer = model.layer4[2].conv3
        
        # Create GradCAM
        gradcam = LayerGradCam(model, target_layer)
        
        # Generate attribution
        attribution = gradcam.attribute(
            image,
            target=target_class,
            relu_attributions=True
        )
        
        # Convert to numpy and squeeze dimensions
        # GradCAM returns (1, 1, H, W), we want (H, W)
        attr_np = attribution.squeeze().cpu().numpy()
        
        # Upsample to original image size if needed
        if attr_np.shape != (224, 224):
            from scipy.ndimage import zoom
            zoom_factor = 224 / attr_np.shape[0]
            attr_np = zoom(attr_np, zoom_factor, order=1)
        
        # Normalize to [0, 1]
        if attr_np.max() > attr_np.min():
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
        
        return attr_np


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GradCAM robustness evaluation')
    parser.add_argument('--config', type=str, 
                       default='experiments/configs/experiment_config.json',
                       help='Path to experiment configuration')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tiny-imagenet-200', 'cifar-10', 'ms-coco-2017'],
                       help='Dataset to evaluate')
    parser.add_argument('--model', type=str, default='standard',
                       choices=['standard', 'robust'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GradCAMEvaluator(args.config)
    
    # Run evaluation
    results = evaluator.evaluate_robustness(
        dataset_name=args.dataset,
        model_type=args.model,
        save_attributions=False
    )
    
    print(f"\nEvaluation complete! Processed {len(results)} images.")


if __name__ == '__main__':
    main()