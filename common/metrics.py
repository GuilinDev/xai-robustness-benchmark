"""
Unified Metrics Module for XAI Robustness Evaluation
This module ensures consistent metric calculation across all XAI methods.
"""

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from typing import Union, Tuple, Dict, Optional
import warnings


class UnifiedMetrics:
    """
    Unified metrics calculator ensuring consistent evaluation across all XAI methods.
    All metrics are computed in the same way regardless of the XAI method being evaluated.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            config: Optional configuration dict with thresholds and parameters
        """
        self.config = config or {}
        self.attribution_threshold = self.config.get('attribution_threshold', 0.5)
        self.top_k_ratio = self.config.get('top_k_pixels', 0.2)
        self.mi_bins = self.config.get('mi_bins', 20)
        self.epsilon = 1e-8  # Small constant to avoid division by zero
        
    def calculate_all_metrics(self, 
                            original_attr: np.ndarray,
                            corrupted_attr: np.ndarray,
                            original_pred: int,
                            corrupted_pred: int,
                            original_probs: np.ndarray,
                            corrupted_probs: np.ndarray,
                            original_top5: np.ndarray,
                            corrupted_top5: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics for a single image pair.
        
        Args:
            original_attr: Original image attribution map
            corrupted_attr: Corrupted image attribution map
            original_pred: Original prediction class index
            corrupted_pred: Corrupted prediction class index
            original_probs: Original prediction probabilities
            corrupted_probs: Corrupted prediction probabilities
            original_top5: Original top-5 predictions
            corrupted_top5: Corrupted top-5 predictions
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Ensure attributions are numpy arrays and same shape
        original_attr = self._validate_attribution(original_attr)
        corrupted_attr = self._validate_attribution(corrupted_attr)
        
        if original_attr.shape != corrupted_attr.shape:
            corrupted_attr = self._resize_attribution(corrupted_attr, original_attr.shape)
        
        # Primary metrics
        cosine_sim = self.cosine_similarity(original_attr, corrupted_attr)
        mutual_info = self.mutual_information(original_attr, corrupted_attr)
        iou = self.intersection_over_union(original_attr, corrupted_attr)
        pred_change = self.prediction_change(original_pred, corrupted_pred)
        
        # Secondary metrics
        stability = self.calculate_stability(original_attr, corrupted_attr)
        kl_div = self.kl_divergence(original_probs, corrupted_probs)
        top5_dist = self.top5_distance(original_top5, corrupted_top5)
        conf_diff = self.confidence_difference(original_probs, corrupted_probs, 
                                              original_pred, corrupted_pred)
        
        # Derived metrics
        cer = self.corruption_error_rate(original_probs[original_pred], 
                                        corrupted_probs[corrupted_pred])
        
        return {
            'cosine_similarity': float(cosine_sim),
            'mutual_information': float(mutual_info),
            'iou': float(iou),
            'prediction_change': int(pred_change),
            'stability': float(stability),
            'kl_divergence': float(kl_div),
            'top5_distance': int(top5_dist),
            'confidence_difference': float(conf_diff),
            'corruption_error': float(cer)
        }
    
    def cosine_similarity(self, attr1: np.ndarray, attr2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two attribution maps.
        
        Args:
            attr1: First attribution map
            attr2: Second attribution map
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Flatten attributions
        attr1_flat = attr1.flatten()
        attr2_flat = attr2.flatten()
        
        # Calculate norms
        norm1 = np.linalg.norm(attr1_flat)
        norm2 = np.linalg.norm(attr2_flat)
        
        # Handle zero attributions
        if norm1 < self.epsilon or norm2 < self.epsilon:
            return 0.0
        
        # Normalize and compute dot product
        attr1_norm = attr1_flat / norm1
        attr2_norm = attr2_flat / norm2
        
        return np.clip(np.dot(attr1_norm, attr2_norm), -1.0, 1.0)
    
    def mutual_information(self, attr1: np.ndarray, attr2: np.ndarray) -> float:
        """
        Calculate mutual information between two attribution maps.
        Uses histogram binning for continuous values.
        
        Args:
            attr1: First attribution map
            attr2: Second attribution map
            
        Returns:
            Mutual information (non-negative)
        """
        # Flatten attributions
        attr1_flat = attr1.flatten()
        attr2_flat = attr2.flatten()
        
        # Handle constant attributions
        if np.std(attr1_flat) < self.epsilon or np.std(attr2_flat) < self.epsilon:
            return 0.0
        
        # Create histograms with same binning
        # First determine the combined range
        combined_min = min(attr1_flat.min(), attr2_flat.min())
        combined_max = max(attr1_flat.max(), attr2_flat.max())
        
        # Create bins
        bins = np.linspace(combined_min, combined_max, self.mi_bins + 1)
        
        # Compute histograms
        hist1, _ = np.histogram(attr1_flat, bins=bins)
        hist2, _ = np.histogram(attr2_flat, bins=bins)
        
        # Add small epsilon to avoid log(0)
        hist1 = hist1 + self.epsilon
        hist2 = hist2 + self.epsilon
        
        # Normalize to get probabilities
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Compute mutual information using sklearn
        # Note: We discretize into bins for MI calculation
        labels1 = np.digitize(attr1_flat, bins[:-1])
        labels2 = np.digitize(attr2_flat, bins[:-1])
        
        return mutual_info_score(labels1, labels2)
    
    def intersection_over_union(self, attr1: np.ndarray, attr2: np.ndarray) -> float:
        """
        Calculate IoU between binarized attribution maps.
        
        Args:
            attr1: First attribution map
            attr2: Second attribution map
            
        Returns:
            IoU in range [0, 1]
        """
        # Normalize attributions to [0, 1]
        attr1_norm = self._normalize_attribution(attr1)
        attr2_norm = self._normalize_attribution(attr2)
        
        # Two strategies for binarization:
        # 1. Threshold-based
        if self.config.get('iou_method', 'threshold') == 'threshold':
            mask1 = attr1_norm > self.attribution_threshold
            mask2 = attr2_norm > self.attribution_threshold
        else:
            # 2. Top-k based
            k1 = int(attr1_norm.size * self.top_k_ratio)
            k2 = int(attr2_norm.size * self.top_k_ratio)
            
            threshold1 = np.sort(attr1_norm.flatten())[-k1]
            threshold2 = np.sort(attr2_norm.flatten())[-k2]
            
            mask1 = attr1_norm >= threshold1
            mask2 = attr2_norm >= threshold2
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        # Handle edge case
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def prediction_change(self, pred1: int, pred2: int) -> int:
        """
        Check if prediction changed.
        
        Args:
            pred1: First prediction
            pred2: Second prediction
            
        Returns:
            1 if changed, 0 if same
        """
        return int(pred1 != pred2)
    
    def calculate_stability(self, attr1: np.ndarray, attr2: np.ndarray) -> float:
        """
        Calculate stability score based on pixel-wise correlation.
        
        Args:
            attr1: First attribution map
            attr2: Second attribution map
            
        Returns:
            Stability score in range [0, 1]
        """
        # Flatten attributions
        attr1_flat = attr1.flatten()
        attr2_flat = attr2.flatten()
        
        # Calculate Pearson correlation
        if np.std(attr1_flat) < self.epsilon or np.std(attr2_flat) < self.epsilon:
            return 0.0
        
        correlation = np.corrcoef(attr1_flat, attr2_flat)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            return 0.0
        
        # Convert to [0, 1] range
        return (correlation + 1.0) / 2.0
    
    def kl_divergence(self, probs1: np.ndarray, probs2: np.ndarray) -> float:
        """
        Calculate KL divergence between two probability distributions.
        
        Args:
            probs1: First probability distribution
            probs2: Second probability distribution
            
        Returns:
            KL divergence (non-negative)
        """
        # Add epsilon to avoid log(0)
        probs1 = probs1 + self.epsilon
        probs2 = probs2 + self.epsilon
        
        # Renormalize
        probs1 = probs1 / probs1.sum()
        probs2 = probs2 / probs2.sum()
        
        # Calculate KL divergence
        return entropy(probs1, probs2)
    
    def top5_distance(self, top5_1: np.ndarray, top5_2: np.ndarray) -> int:
        """
        Calculate the distance between two top-5 predictions.
        
        Args:
            top5_1: First top-5 predictions
            top5_2: Second top-5 predictions
            
        Returns:
            Number of different predictions (0-5)
        """
        # Convert to sets and find symmetric difference
        set1 = set(top5_1.tolist())
        set2 = set(top5_2.tolist())
        
        # Count unique elements
        return len(set1.symmetric_difference(set2)) // 2
    
    def confidence_difference(self, probs1: np.ndarray, probs2: np.ndarray,
                            pred1: int, pred2: int) -> float:
        """
        Calculate the confidence difference between predictions.
        
        Args:
            probs1: First probability distribution
            probs2: Second probability distribution
            pred1: First prediction
            pred2: Second prediction
            
        Returns:
            Absolute confidence difference
        """
        conf1 = probs1[pred1]
        conf2 = probs2[pred2]
        
        return abs(conf1 - conf2)
    
    def corruption_error_rate(self, conf_original: float, conf_corrupted: float) -> float:
        """
        Calculate corruption error rate.
        
        Args:
            conf_original: Original confidence
            conf_corrupted: Corrupted confidence
            
        Returns:
            Corruption error rate
        """
        error_original = 1.0 - conf_original
        error_corrupted = 1.0 - conf_corrupted
        
        # Relative increase in error
        if error_original < self.epsilon:
            return error_corrupted
        
        return error_corrupted / error_original
    
    def _validate_attribution(self, attr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert attribution to numpy and validate."""
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()
        
        if not isinstance(attr, np.ndarray):
            raise TypeError(f"Attribution must be numpy array or torch tensor, got {type(attr)}")
        
        # Handle different number of dimensions
        if attr.ndim == 4:  # Batch dimension
            attr = attr[0]
        if attr.ndim == 3 and attr.shape[0] in [1, 3]:  # Channel first
            attr = attr.mean(axis=0)
        elif attr.ndim == 3 and attr.shape[-1] in [1, 3]:  # Channel last
            attr = attr.mean(axis=-1)
        
        return attr
    
    def _normalize_attribution(self, attr: np.ndarray) -> np.ndarray:
        """Normalize attribution to [0, 1] range."""
        attr_min = attr.min()
        attr_max = attr.max()
        
        if attr_max - attr_min < self.epsilon:
            return np.zeros_like(attr)
        
        return (attr - attr_min) / (attr_max - attr_min)
    
    def _resize_attribution(self, attr: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Resize attribution to target shape using bilinear interpolation."""
        if attr.shape == target_shape:
            return attr
        
        # Use scipy zoom for resizing
        from scipy.ndimage import zoom
        
        zoom_factors = [t / s for t, s in zip(target_shape, attr.shape)]
        return zoom(attr, zoom_factors, order=1)


def test_metrics():
    """Test function to verify metrics calculation."""
    print("Testing Unified Metrics...")
    
    # Create sample data
    np.random.seed(42)
    attr1 = np.random.rand(224, 224)
    attr2 = attr1 + np.random.randn(224, 224) * 0.1  # Similar but with noise
    
    pred1, pred2 = 5, 7
    probs1 = np.random.rand(1000)
    probs1 = probs1 / probs1.sum()
    probs2 = np.random.rand(1000)
    probs2 = probs2 / probs2.sum()
    
    top5_1 = np.array([5, 10, 3, 7, 9])
    top5_2 = np.array([7, 10, 3, 5, 8])
    
    # Initialize metrics calculator
    metrics = UnifiedMetrics()
    
    # Calculate all metrics
    results = metrics.calculate_all_metrics(
        attr1, attr2, pred1, pred2, probs1, probs2, top5_1, top5_2
    )
    
    # Print results
    print("\nMetric Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMetrics test completed successfully!")


if __name__ == "__main__":
    test_metrics()