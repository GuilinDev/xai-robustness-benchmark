#!/usr/bin/env python3
"""
ViT Extra Methods: SmoothGrad + SHAP on ViT-B/16
Supplements the main ViT experiment (IG, GradCAM, RISE) with 2 additional methods.
Uses Captum's NoiseTunnel (SmoothGrad) and GradientShap (SHAP).
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
import gc
from typing import Dict, List, Tuple
from scipy.ndimage import zoom, map_coordinates
from skimage.filters import gaussian
from skimage import util as sk_util
from io import BytesIO
from sklearn.metrics import mutual_info_score
from captum.attr import Saliency, NoiseTunnel, GradientShap
import torch.nn.functional as F


def clipped_zoom(img, zoom_factor):
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


class ViTExtraMethodsTest:
    """Test SmoothGrad and SHAP on ViT-B/16"""

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load ViT-B/16
        print("Loading ViT-B/16...")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize explainers
        print("Initializing SmoothGrad (NoiseTunnel + Saliency)...")
        self.saliency = Saliency(self.model)
        self.smoothgrad = NoiseTunnel(self.saliency)

        print("Initializing SHAP (Captum GradientShap)...")
        self.shap = GradientShap(self.model)
        self.shap_baseline = torch.zeros((1, 3, 224, 224)).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Full 15 corruption types
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'
        ]

        self.methods = ['smoothgrad', 'shap']
        self.severity_levels = [1, 2, 3, 4, 5]
        print(f"Methods: {self.methods}")
        print(f"Corruptions: {len(self.corruption_types)}, Severities: {self.severity_levels}")
        print("Initialization complete!")

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
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

    def generate_explanation(self, input_tensor: torch.Tensor, method: str, target_class: int) -> np.ndarray:
        """Generate explanation using specified method"""
        if method == 'smoothgrad':
            attr = self.smoothgrad.attribute(
                input_tensor, target=target_class,
                nt_samples=30, stdevs=0.1, nt_type='smoothgrad'
            )
        elif method == 'shap':
            attr = self.shap.attribute(
                input_tensor, baselines=self.shap_baseline,
                target=target_class, n_samples=30, stdevs=0.09
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        attr_sum = attr.abs().sum(dim=1).squeeze(0).cpu().detach().numpy()
        del attr
        attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        return attr_sum

    def compute_metrics(self, img_exp: np.ndarray, corr_exp: np.ndarray,
                       img_pred: int, corr_pred: int,
                       img_probs: np.ndarray, corr_probs: np.ndarray) -> Dict:
        cosine_sim = np.sum(img_exp * corr_exp) / (
            np.sqrt(np.sum(img_exp**2)) * np.sqrt(np.sum(corr_exp**2)) + 1e-8)

        img_hist = np.histogram(img_exp.flatten(), bins=20)[0]
        corr_hist = np.histogram(corr_exp.flatten(), bins=20)[0]
        mutual_info = mutual_info_score(img_hist, corr_hist)

        threshold = 0.5
        binary_img = (img_exp > threshold).astype(int)
        binary_corr = (corr_exp > threshold).astype(int)
        intersection = np.logical_and(binary_img, binary_corr).sum()
        union = np.logical_or(binary_img, binary_corr).sum()
        iou = intersection / (union + 1e-8)

        prediction_change = int(img_pred != corr_pred)
        confidence_diff = abs(float(img_probs[img_pred]) - float(corr_probs[corr_pred]))

        return {
            "similarity": float(cosine_sim),
            "consistency": float(mutual_info),
            "localization": float(iou),
            "prediction_change": prediction_change,
            "confidence_diff": confidence_diff,
        }

    def test_robustness(self, image_dir: str, output_file: str, temp_file: str = None,
                       num_images: int = 500):
        results = {}

        if temp_file and os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                results = json.load(f)
                print(f"Loaded {len(results)} previous results")

        # Get image files
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.JPEG')):
                    image_files.append(os.path.join(root, file))

        if len(image_files) > num_images:
            np.random.seed(42)
            image_files = np.random.choice(image_files, num_images, replace=False).tolist()

        total_images = len(image_files)
        total_corruptions = len(self.corruption_types) * len(self.severity_levels)

        print(f"\n{'='*60}")
        print(f"ViT Extra Methods: SmoothGrad + SHAP")
        print(f"Images: {total_images}, Corruptions: {total_corruptions}")
        print(f"{'='*60}\n")

        start_time = time.time()
        completed = sum(1 for img in image_files if img in results)

        for idx, image_path in enumerate(image_files):
            if image_path in results:
                continue

            image_start = time.time()
            elapsed = time.time() - start_time

            if completed > 0:
                avg_time = elapsed / completed
                remaining = avg_time * (total_images - completed)
                eta = f"{remaining/60:.1f}min ({remaining/3600:.1f}h)"
            else:
                eta = "calculating..."

            print(f"\n[{idx+1}/{total_images}] ({(idx+1)/total_images*100:.1f}%) ETA: {eta}")
            print(f"  {os.path.basename(image_path)}")

            try:
                results[image_path] = {}
                input_tensor, original_image = self.load_image(image_path)

                with torch.no_grad():
                    output = self.model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = int(output.argmax(dim=1).item())

                for method in self.methods:
                    results[image_path][method] = {}

                    # Original explanation
                    original_exp = self.generate_explanation(input_tensor, method, pred_class)

                    for corruption_type in self.corruption_types:
                        results[image_path][method][corruption_type] = {"results": []}

                        for severity in self.severity_levels:
                            corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                            corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                corr_output = self.model(corrupted_tensor)
                            corr_probs = F.softmax(corr_output, dim=1).cpu().numpy()[0]
                            corr_pred = int(corr_output.argmax(dim=1).item())

                            corrupted_exp = self.generate_explanation(corrupted_tensor, method, corr_pred)
                            del corrupted_tensor

                            metrics = self.compute_metrics(
                                original_exp, corrupted_exp,
                                pred_class, corr_pred, probs, corr_probs
                            )
                            del corrupted_exp
                            metrics["severity"] = severity
                            results[image_path][method][corruption_type]["results"].append(metrics)

                        # Clean up after each corruption type
                        torch.cuda.empty_cache()

                    del original_exp

                # Clean up after each image
                torch.cuda.empty_cache()
                gc.collect()

                # Save after each image
                if temp_file:
                    with open(temp_file, 'w') as f:
                        json.dump(results, f)

                completed += 1
                image_time = time.time() - image_start
                print(f"  Completed in {image_time:.1f}s")

            except Exception as e:
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()
                if temp_file:
                    with open(temp_file, 'w') as f:
                        json.dump(results, f)
                continue

        # Save final
        final_results = {
            "architecture": "ViT-B/16",
            "methods": self.methods,
            "num_images": completed,
            "corruptions": self.corruption_types,
            "severities": self.severity_levels,
            "per_image_results": results
        }

        with open(output_file, 'w') as f:
            json.dump(final_results, f)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETED! Saved to: {output_file}")
        print(f"Time: {total_time/60:.1f}min ({total_time/3600:.2f}h)")
        print(f"Average: {total_time/max(completed,1):.1f}s/image")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="ViT Extra Methods (SmoothGrad + SHAP)")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--temp_file', type=str)
    parser.add_argument('--num_images', type=int, default=500)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if args.temp_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.temp_file)), exist_ok=True)

    device = torch.device(args.device) if args.device else None
    tester = ViTExtraMethodsTest(device=device)
    tester.test_robustness(
        image_dir=args.image_dir,
        output_file=args.output_file,
        temp_file=args.temp_file,
        num_images=args.num_images
    )


if __name__ == "__main__":
    main()
