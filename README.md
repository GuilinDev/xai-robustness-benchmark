# XAI Robustness Benchmark

**Official implementation of "Benchmarking XAI Method Robustness under Natural Image Corruptions"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

## 📋 Overview

This repository provides a comprehensive benchmark for evaluating the robustness of Explainable AI (XAI) methods under natural image corruptions. We systematically assess **6 representative XAI methods** across **3 datasets**, **2 model architectures**, **15 corruption types**, and **5 severity levels**, totaling over **2.7 million comparisons**.

### Key Features

- ✅ **Comprehensive Evaluation Framework**: Unified evaluation pipeline for 6 XAI methods
- ✅ **Standardized Corruption Protocol**: 15 corruption types following ImageNet-C benchmark
- ✅ **Multi-Dimensional Robustness Metrics**: 11 metrics across 3 analytical dimensions
- ✅ **Reproducible Sampling Strategy**: Fixed sample lists with balanced class distribution
- ✅ **Model-Agnostic Design**: Support for both standard and robust model variants

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/xai-robustness-benchmark.git
cd xai-robustness-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from common.base_xai_evaluator import XAIEvaluator
from common.corruptions import apply_corruption
from common.metrics import compute_robustness_metrics

# Initialize evaluator
evaluator = XAIEvaluator(
    method='gradcam',
    dataset='cifar10',
    model_type='standard'
)

# Run robustness evaluation
results = evaluator.evaluate_robustness(
    corruption_types=['gaussian_noise', 'defocus_blur'],
    severity_levels=[1, 3, 5]
)

# Compute metrics
metrics = compute_robustness_metrics(
    original_explanations=results['original'],
    corrupted_explanations=results['corrupted']
)
```

## 📊 Supported Components

### XAI Methods

| Category | Methods | Computational Efficiency |
|----------|---------|-------------------------|
| **Attribution-based** | GradCAM, Integrated Gradients (IG), LRP | High (< 3s/image) |
| **Perturbation-based** | LIME, RISE, Occlusion Sensitivity | Medium-Low (4-185s/image) |

### Datasets

- **CIFAR-10**: 1,000 images (100 per class) - Low complexity
- **Tiny-ImageNet-200**: 1,000 images (5 per class) - Medium complexity  
- **MS-COCO-2017**: 1,000 images (validation set) - High complexity

### Corruption Types

**Noise**: Gaussian, Shot, Impulse  
**Blur**: Defocus, Glass, Motion, Zoom  
**Weather**: Snow, Frost, Fog  
**Digital**: Brightness, Contrast, Elastic Transform, Pixelation, JPEG Compression

### Robustness Metrics

**Similarity-based**: Pearson Correlation, Cosine Similarity, SSIM, Consistency Rate  
**Localization-based**: IoU, Rank Correlation  
**Prediction-based**: KL Divergence, Confidence Difference, Mutual Information, Earth Mover's Distance, Top-k Intersection

## 📖 Reproducing Paper Results

### Step 1: Download Datasets

```bash
# CIFAR-10 (smallest, recommended for testing)
python datasets/cifar-10/download.py

# Tiny-ImageNet-200 (medium complexity)
python datasets/tiny-imagenet-200/download.py

# MS-COCO-2017 (largest, most realistic)
python datasets/ms-coco-2017/download.py

# Or download all at once
for dataset in cifar-10 tiny-imagenet-200 ms-coco-2017; do
    python datasets/$dataset/download.py
done
```

### Step 2: Run Experiments

```bash
# Run all experiments (full benchmark)
python scripts/run_all_experiments.py

# Run specific method
python scripts/run_method.py --method gradcam --dataset cifar10

# Run with custom configuration
python scripts/run_experiments.py --config configs/custom_config.yaml
```

### Step 3: Generate Results

```bash
# Analyze robustness results
python scripts/analyze_robustness_results.py

# Generate paper figures
python scripts/generate_paper_figures.py

# Summarize all results
python scripts/summarize_all_results.py
```

## 📈 Key Results

### Overall Robustness Ranking

| Rank | Method | Type | Mean Score | Std Dev |
|------|--------|------|------------|---------|
| 1 | LRP | Attribution | 0.994 | 0.074 |
| 2 | RISE | Perturbation | 0.904 | 0.054 |
| 3 | GradCAM | Attribution | 0.852 | 0.132 |
| 4 | LIME | Perturbation | 0.644 | 0.189 |
| 5 | Occlusion | Perturbation | 0.525 | 0.167 |
| 6 | IG | Attribution | 0.508 | 0.145 |

### Practical Recommendations

- **Highest Quality**: LRP (0.994 robustness, 0.89s/image)
- **Balanced Choice**: RISE (0.904 robustness, 62.15s/image)
- **Real-time Applications**: GradCAM (0.852 robustness, 0.12s/image)

## 🔧 Advanced Configuration

### Custom XAI Method Parameters

```yaml
# configs/experiment_config.yaml
xai_methods:
  gradcam:
    target_layer: 'layer4'
  integrated_gradients:
    n_steps: 50
  lime:
    num_samples: 1000
    num_features: 10
```

### Custom Corruption Settings

```yaml
corruptions:
  gaussian_noise:
    severities: [1, 2, 3, 4, 5]
  defocus_blur:
    severities: [3, 5]
```

## 📁 Project Structure

```
xai-robustness-benchmark/
├── common/                      # Core modules
│   ├── base_evaluator.py       # Base evaluator class
│   ├── base_xai_evaluator.py   # XAI-specific evaluator
│   ├── corruptions.py          # Corruption implementations
│   ├── metrics.py              # Robustness metrics
│   └── unified_data_loader.py  # Dataset loaders
├── methods/                     # XAI method implementations
│   ├── gradcam_evaluator.py
│   ├── ig_evaluator.py
│   ├── lrp_evaluator.py
│   ├── lime_evaluator.py
│   ├── rise_evaluator.py
│   └── occlusion_evaluator.py
├── datasets/                    # Dataset scripts and sample lists
│   ├── cifar-10/
│   ├── ms-coco-2017/
│   └── tiny-imagenet-200/
├── configs/                     # Configuration files
│   └── experiment_config.yaml
├── scripts/                     # Analysis and visualization scripts
│   ├── run_all_experiments.py
│   ├── analyze_robustness_results.py
│   ├── generate_paper_figures.py
│   └── summarize_all_results.py
├── results/                     # Experimental results
├── docs/                        # Documentation
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🛠️ Extending the Benchmark

### Adding a New XAI Method

```python
# methods/your_method_evaluator.py
from common.base_xai_evaluator import BaseXAIEvaluator

class YourMethodEvaluator(BaseXAIEvaluator):
    def generate_explanation(self, image, target_class):
        # Implement your XAI method here
        explanation = your_xai_method(image, target_class)
        return explanation
```

### Adding a New Corruption Type

```python
# common/corruptions.py
def apply_your_corruption(image, severity):
    # Implement your corruption here
    corrupted_image = your_corruption_function(image, severity)
    return corrupted_image
```

## 📚 Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@article{your2025xai,
  title={Benchmarking XAI Method Robustness under Natural Image Corruptions},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ImageNet-C benchmark: [Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261)
- RobustBench: [Croce et al., 2021](https://arxiv.org/abs/2010.09670)
- Captum library for XAI method implementations

## 📧 Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Email: your.email@example.com

## 🔄 Updates

- **v1.0.0** (2025-10): Initial release with paper submission
- Coming soon: Extended support for additional XAI methods and datasets

---

**Star ⭐ this repository if you find it helpful!**

