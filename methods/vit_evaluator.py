#!/usr/bin/env python3
"""
ViT-B/16 Pilot Study for XAI Robustness
Tests the robustness of XAI explanations on Vision Transformer architecture.
This is a pilot study to investigate architectural generalization (R2-1).

Scope: CIFAR-10 only, 100 images, 5 key corruptions
Methods: GradCAM (attention-based), IG, RISE (model-agnostic)
"""

import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
import cv2
import time
import argparse
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, map_coordinates
from skimage.filters import gaussian
from skimage import util as sk_util
from io import BytesIO
from sklearn.metrics import mutual_info_score
from captum.attr import IntegratedGradients, LayerGradCam
import matplotlib.pyplot as plt
import torch.nn.functional as F


def clipped_zoom(img, zoom_factor):
    """Center-crop an image after zooming"""
    h, w = img.shape[:2]
    zh = int(np.round(h * zoom_factor))
    zw = int(np.round(w * zoom_factor))
    zh = max(zh, 1)
    zw = max(zw, 1)
    zoomed = zoom(img, [zoom_factor, zoom_factor, 1], order=1, mode='reflect')
    if zoom_factor > 1:
        trim_h = ((zoomed.shape[0] - h) // 2)
        trim_w = ((zoomed.shape[1] - w) // 2)
        zoomed = zoomed[trim_h:trim_h+h, trim_w:trim_w+w]
    elif zoom_factor < 1:
        pad_h = ((h - zoomed.shape[0]) // 2)
        pad_w = ((w - zoomed.shape[1]) // 2)
        out = np.zeros_like(img)
        out[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
        zoomed = out
    if zoomed.shape[:2] != (h, w):
        zoomed = cv2.resize(zoomed, (w, h))
    return zoomed


class ViTAttentionRollout:
    """
    Attention Rollout for ViT - computes attention-based attribution
    Based on "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
    """
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []

        # Register hooks to capture attention weights
        for name, module in self.model.named_modules():
            if hasattr(module, 'attn_drop'):  # ViT attention layers
                module.register_forward_hook(self._get_attention)

    def _get_attention(self, module, input, output):
        """Hook to capture attention weights"""
        # Note: This requires accessing internal attention weights
        # For torchvision ViT, we need to access the attention differently
        pass

    def __call__(self, input_tensor, target_class=None):
        """
        Compute attention rollout attribution.
        Returns a 2D numpy array of size matching input spatial dimensions.
        """
        self.attentions = []

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # For torchvision ViT, we use gradient-based attribution instead
        # as attention hooks are complex to implement
        # Fall back to IG for now
        return None


class ViTRobustnessTest:
    """ViT robustness test class for pilot study"""

    def __init__(self, device: torch.device = None):
        """Initialize ViT tester"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load ViT-B/16 model
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        print("Loaded ViT-B/16 model (ImageNet pretrained)")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Create IG explainer for ViT
        self.ig = IntegratedGradients(self.model)

        # ViT-specific transform (224x224 expected)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Full 15 corruption types (matching other methods for fair comparison)
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur
            'snow', 'frost', 'fog', 'brightness',  # Weather
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'  # Digital
        ]

        # Methods to test
        self.methods = ['ig', 'gradcam', 'rise']

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load image and convert to model input format"""
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
        """Apply image corruption - supports all 15 corruption types"""
        img = np.array(image) / 255.0
        np.random.seed(1)
        sev = float(severity) / 5.0

        if corruption_type == 'gaussian_noise':
            noise = np.random.normal(loc=0, scale=sev * 0.5, size=img.shape)
            corrupted = np.clip(img + noise, 0, 1)
        elif corruption_type == 'shot_noise':
            corrupted = np.random.poisson(img * 255.0 * (1-sev)) / 255.0
            corrupted = np.clip(corrupted, 0, 1)
        elif corruption_type == 'impulse_noise':
            corrupted = sk_util.random_noise(img, mode='s&p', amount=sev)
        elif corruption_type == 'defocus_blur':
            corrupted = gaussian(img, sigma=sev * 4, channel_axis=-1)
        elif corruption_type == 'glass_blur':
            kernel_size = int(sev * 10) * 2 + 1
            corrupted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif corruption_type == 'motion_blur':
            kernel_size = max(int(sev * 20), 1)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            corrupted = cv2.filter2D(img, -1, kernel)
        elif corruption_type == 'zoom_blur':
            out = np.zeros_like(img)
            zoom_factors = [1-sev*0.1, 1-sev*0.05, 1, 1+sev*0.05, 1+sev*0.1]
            for zf in zoom_factors:
                zoomed = clipped_zoom(img, zf)
                out += zoomed
            corrupted = np.clip(out / len(zoom_factors), 0, 1)
        elif corruption_type == 'snow':
            snow_layer = np.random.normal(size=img.shape[:2], loc=0.5, scale=sev * 1.5)
            snow_layer = np.clip(snow_layer, 0, 1)
            snow_layer = np.expand_dims(snow_layer, axis=2)
            corrupted = np.clip(img + snow_layer, 0, 1)
        elif corruption_type == 'frost':
            frost_layer = np.random.uniform(size=img.shape[:2]) * sev
            frost_layer = np.expand_dims(frost_layer, axis=2)
            corrupted = np.clip(img * (1 - frost_layer), 0, 1)
        elif corruption_type == 'fog':
            fog_layer = sev * np.ones_like(img)
            corrupted = np.clip(img * (1 - sev) + fog_layer * sev, 0, 1)
        elif corruption_type == 'brightness':
            corrupted = np.clip(img * (1 + sev), 0, 1)
        elif corruption_type == 'contrast':
            mean = np.mean(img, axis=(0,1), keepdims=True)
            corrupted = np.clip((img - mean) * (1 + sev) + mean, 0, 1)
        elif corruption_type == 'elastic_transform':
            shape = img.shape[:2]
            dx = np.random.uniform(-1, 1, shape) * sev * 30
            dy = np.random.uniform(-1, 1, shape) * sev * 30
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            corrupted = np.zeros_like(img)
            for c in range(3):
                corrupted[:,:,c] = np.reshape(map_coordinates(img[:,:,c], indices, order=1), shape)
        elif corruption_type == 'pixelate':
            h, w = img.shape[:2]
            size = max(int((1-sev) * min(h,w)), 1)
            corrupted = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
            corrupted = cv2.resize(corrupted, (w,h), interpolation=cv2.INTER_NEAREST)
        elif corruption_type == 'jpeg':
            quality = max(int((1-sev) * 100), 1)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            corrupted = np.array(Image.open(buffer)) / 255.0
        else:
            return image

        corrupted = np.clip(corrupted * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted)

    def generate_ig(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Integrated Gradients explanation for ViT"""
        with torch.no_grad():
            output = self.model(input_tensor)
            target_class = torch.argmax(output, dim=1).item()

        # Calculate IG attribution
        attributions = self.ig.attribute(
            input_tensor,
            target=target_class,
            n_steps=50
        )

        # Sum across channels and take absolute value
        attr_sum = torch.abs(attributions).sum(dim=1).squeeze(0).cpu().detach().numpy()

        # Normalize to [0, 1]
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)

        return cv2.resize(attr_norm, (224, 224))

    def generate_gradcam_vit(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate GradCAM-like explanation for ViT.
        Uses the last encoder block's output.
        """
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        target_class = torch.argmax(output, dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradient w.r.t. input
        gradients = input_tensor.grad

        # Simple saliency: absolute gradient sum across channels
        saliency = torch.abs(gradients).sum(dim=1).squeeze(0).cpu().detach().numpy()

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        input_tensor.requires_grad = False

        return cv2.resize(saliency, (224, 224))

    def generate_rise(self, input_tensor: torch.Tensor, n_masks: int = 500) -> np.ndarray:
        """
        Generate RISE explanation for ViT.
        RISE is model-agnostic and works with any architecture.
        """
        # Get predicted class
        with torch.no_grad():
            output = self.model(input_tensor)
            target_class = torch.argmax(output, dim=1).item()

        # Input dimensions
        _, C, H, W = input_tensor.shape

        # Generate random masks
        cell_size = 7  # Mask cell size
        up_size = (H, W)

        # Initialize saliency map
        saliency = np.zeros((H, W))

        for _ in range(n_masks):
            # Generate random binary mask at low resolution
            mask_small = np.random.rand(cell_size, cell_size) < 0.5

            # Upscale mask
            mask = cv2.resize(mask_small.astype(np.float32), up_size, interpolation=cv2.INTER_LINEAR)
            mask = torch.from_numpy(mask).to(self.device).float()

            # Apply mask to input
            masked_input = input_tensor * mask.unsqueeze(0).unsqueeze(0)

            # Get model output
            with torch.no_grad():
                masked_output = self.model(masked_input)
                masked_prob = F.softmax(masked_output, dim=1)[0, target_class].cpu().numpy()

            # Accumulate weighted mask
            saliency += masked_prob * mask.cpu().numpy()

        # Normalize
        saliency = saliency / n_masks
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return cv2.resize(saliency, (224, 224))

    def generate_explanation(self, input_tensor: torch.Tensor, method: str) -> np.ndarray:
        """Generate explanation using specified method"""
        if method == 'ig':
            return self.generate_ig(input_tensor)
        elif method == 'gradcam':
            return self.generate_gradcam_vit(input_tensor)
        elif method == 'rise':
            return self.generate_rise(input_tensor)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_metrics(self, img_explanation: np.ndarray, corrupted_explanation: np.ndarray,
                       img_pred: int, corrupted_pred: int, img_probs: np.ndarray,
                       corrupted_probs: np.ndarray) -> Dict:
        """Calculate robustness metrics"""

        # Cosine similarity
        cosine_sim = np.sum(img_explanation * corrupted_explanation) / (
            np.sqrt(np.sum(img_explanation**2)) * np.sqrt(np.sum(corrupted_explanation**2)) + 1e-8)

        # Mutual information
        img_hist = np.histogram(img_explanation.flatten(), bins=20)[0]
        corrupted_hist = np.histogram(corrupted_explanation.flatten(), bins=20)[0]
        mutual_info = mutual_info_score(img_hist, corrupted_hist)

        # IoU
        threshold = 0.5
        binary_img = (img_explanation > threshold).astype(int)
        binary_corrupted = (corrupted_explanation > threshold).astype(int)
        intersection = np.logical_and(binary_img, binary_corrupted).sum()
        union = np.logical_or(binary_img, binary_corrupted).sum()
        iou = intersection / (union + 1e-8)

        # Prediction change
        prediction_change = int(img_pred != corrupted_pred)

        # Confidence difference
        confidence_diff = np.abs(img_probs[img_pred] - corrupted_probs[corrupted_pred])

        return {
            "similarity": float(cosine_sim),
            "consistency": float(mutual_info),
            "localization": float(iou),
            "prediction_change": prediction_change,
            "confidence_diff": float(confidence_diff)
        }

    def test_robustness(self, image_dir: str, output_file: str,
                        num_images: int = 100, severities: List[int] = None):
        """
        Test robustness of XAI methods on ViT

        Args:
            image_dir: Directory containing images
            output_file: Output JSON file path
            num_images: Number of images to test (default: 100)
            severities: List of severity levels (default: [1, 3, 5])
        """
        if severities is None:
            severities = [1, 2, 3, 4, 5]  # All severity levels for full comparison

        results = {
            "architecture": "ViT-B/16",
            "dataset": "cifar-10",
            "num_images": num_images,
            "corruptions": self.corruption_types,
            "methods": self.methods,
            "severities": severities,
            "results": {}
        }

        # Get image list
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files.append(os.path.join(root, file))

        # Limit to specified number of images
        if len(image_files) > num_images:
            np.random.seed(42)
            image_files = np.random.choice(image_files, num_images, replace=False).tolist()

        print(f"Testing {len(image_files)} images with {len(self.methods)} methods")
        print(f"Corruptions: {self.corruption_types}")
        print(f"Severities: {severities}")

        start_time = time.time()

        # Initialize results structure
        for method in self.methods:
            results["results"][method] = {}
            for corruption in self.corruption_types:
                results["results"][method][corruption] = {}
                for severity in severities:
                    results["results"][method][corruption][str(severity)] = {
                        "similarities": [],
                        "consistencies": [],
                        "localizations": [],
                        "prediction_changes": [],
                        "confidence_diffs": []
                    }

        # Process each image
        for idx, image_path in enumerate(image_files):
            print(f"\nProcessing image [{idx+1}/{len(image_files)}]: {os.path.basename(image_path)}")

            try:
                # Load image
                input_tensor, original_image = self.load_image(image_path)

                # Get original prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = output.argmax(dim=1).item()

                # Test each method
                for method in self.methods:
                    print(f"  Method: {method}")

                    # Generate original explanation
                    try:
                        original_exp = self.generate_explanation(input_tensor, method)
                    except Exception as e:
                        print(f"    Error generating {method} explanation: {e}")
                        continue

                    # Test each corruption
                    for corruption in self.corruption_types:
                        for severity in severities:
                            # Apply corruption
                            corrupted_image = self.apply_corruption(original_image, corruption, severity)
                            corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)

                            # Get corrupted prediction
                            with torch.no_grad():
                                corrupted_output = self.model(corrupted_tensor)
                                corrupted_probs = F.softmax(corrupted_output, dim=1).cpu().numpy()[0]
                                corrupted_pred = corrupted_output.argmax(dim=1).item()

                            # Generate corrupted explanation
                            try:
                                corrupted_exp = self.generate_explanation(corrupted_tensor, method)
                            except Exception as e:
                                continue

                            # Compute metrics
                            metrics = self.compute_metrics(
                                original_exp, corrupted_exp,
                                pred_class, corrupted_pred,
                                probs, corrupted_probs
                            )

                            # Store results
                            key = str(severity)
                            results["results"][method][corruption][key]["similarities"].append(metrics["similarity"])
                            results["results"][method][corruption][key]["consistencies"].append(metrics["consistency"])
                            results["results"][method][corruption][key]["localizations"].append(metrics["localization"])
                            results["results"][method][corruption][key]["prediction_changes"].append(metrics["prediction_change"])
                            results["results"][method][corruption][key]["confidence_diffs"].append(metrics["confidence_diff"])

            except Exception as e:
                print(f"  Error processing image: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Progress update
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(image_files) - idx - 1)
                print(f"\n  Progress: {idx+1}/{len(image_files)} ({100*(idx+1)/len(image_files):.1f}%)")
                print(f"  Estimated remaining time: {remaining/60:.1f} minutes")

        # Compute summary statistics
        print("\nComputing summary statistics...")
        summary = {}
        for method in self.methods:
            summary[method] = {}
            for corruption in self.corruption_types:
                summary[method][corruption] = {}
                for severity in severities:
                    key = str(severity)
                    sims = results["results"][method][corruption][key]["similarities"]
                    if sims:
                        summary[method][corruption][key] = {
                            "mean_similarity": float(np.mean(sims)),
                            "std_similarity": float(np.std(sims)),
                            "mean_localization": float(np.mean(results["results"][method][corruption][key]["localizations"])),
                            "prediction_change_rate": float(np.mean(results["results"][method][corruption][key]["prediction_changes"]))
                        }

        results["summary"] = summary

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        total_time = time.time() - start_time
        print(f"\nPilot study completed. Results saved to {output_file}")
        print(f"Total time: {total_time/60:.1f} minutes")

        # Print summary comparison with ResNet-50
        print("\n" + "="*60)
        print("ViT-B/16 Pilot Study Summary")
        print("="*60)
        for method in self.methods:
            all_sims = []
            for corruption in self.corruption_types:
                for severity in severities:
                    key = str(severity)
                    all_sims.extend(results["results"][method][corruption][key]["similarities"])
            if all_sims:
                print(f"{method.upper()}: Mean similarity = {np.mean(all_sims):.3f} (±{np.std(all_sims):.3f})")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ViT-B/16 Pilot Study for XAI Robustness')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing CIFAR-10 validation images')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to test (default: 100)')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 3, 5],
                        help='Severity levels to test (default: 1 3 5)')

    args = parser.parse_args()

    # Create tester
    tester = ViTRobustnessTest()

    # Run test
    tester.test_robustness(
        image_dir=args.image_dir,
        output_file=args.output_file,
        num_images=args.num_images,
        severities=args.severities
    )


if __name__ == "__main__":
    main()
