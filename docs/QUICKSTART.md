# Quick Start Guide

This guide will help you get started with the XAI Robustness Benchmark in 5 minutes.

## 🎯 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster computation)
- 20GB free disk space for datasets

## 📦 Installation

### Step 1: Clone and Setup

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

### Step 2: Download Datasets

```bash
# Option 1: Download CIFAR-10 (smallest, fastest)
python datasets/cifar-10/download.py

# Option 2: Download MS-COCO-2017 (medium size)
python datasets/ms-coco-2017/download.py

# Option 3: Manual download for Tiny-ImageNet-200
# See datasets/tiny-imagenet-200/README.md
```

## 🚀 Running Your First Evaluation

### Example 1: Single Method Evaluation

```python
from common.base_xai_evaluator import XAIEvaluator

# Initialize evaluator
evaluator = XAIEvaluator(
    method='gradcam',
    dataset='cifar10',
    model_type='standard'
)

# Run evaluation on a single corruption
results = evaluator.evaluate_robustness(
    corruption_types=['gaussian_noise'],
    severity_levels=[3]
)

# Print results
print(f"Robustness Score: {results['mean_similarity']:.3f}")
```

### Example 2: Compare Multiple Methods

```python
from scripts.run_comparison import compare_methods

# Compare GradCAM vs LIME
results = compare_methods(
    methods=['gradcam', 'lime'],
    dataset='cifar10',
    corruption_type='gaussian_noise',
    severity=3
)

# Visualize comparison
plot_comparison(results)
```

### Example 3: Full Benchmark Run

```bash
# Run complete benchmark (WARNING: Takes ~24 hours on single GPU)
python scripts/run_all_experiments.py --config configs/experiment_config.yaml

# Or run a smaller subset
python scripts/run_all_experiments.py --methods gradcam,lime --dataset cifar10 --severity 3
```

## 📊 Analyzing Results

### Generate Summary Statistics

```bash
# Summarize all results
python scripts/summarize_all_results.py

# Output: results/summary_statistics.json
```

### Generate Visualizations

```bash
# Generate all paper figures
python scripts/generate_paper_figures.py

# Output: figures/
# - corruption_heatmap.pdf
# - severity_curves.pdf
# - model_comparison.pdf
# - ranking_visualization.pdf
```

## 🔧 Common Tasks

### Task 1: Evaluate a Single Image

```python
import torch
from PIL import Image
from common.corruptions import apply_corruption
from methods.gradcam_evaluator import GradCAMEvaluator

# Load image
image = Image.open('path/to/image.jpg')

# Initialize method
evaluator = GradCAMEvaluator()

# Generate explanation for original image
explanation_orig = evaluator.generate_explanation(image)

# Apply corruption
corrupted_image = apply_corruption(image, 'gaussian_noise', severity=3)

# Generate explanation for corrupted image
explanation_corr = evaluator.generate_explanation(corrupted_image)

# Compute similarity
from common.metrics import compute_similarity
similarity = compute_similarity(explanation_orig, explanation_corr)
print(f"Similarity: {similarity:.3f}")
```

### Task 2: Custom Corruption Evaluation

```python
from common.base_xai_evaluator import XAIEvaluator

evaluator = XAIEvaluator(method='gradcam', dataset='cifar10')

# Evaluate specific corruptions
results = evaluator.evaluate_robustness(
    corruption_types=['gaussian_noise', 'defocus_blur', 'frost'],
    severity_levels=[1, 3, 5]
)

# Access detailed results
for corruption in results['per_corruption']:
    print(f"{corruption['name']}: {corruption['mean_score']:.3f}")
```

### Task 3: Batch Processing

```python
from common.batch_evaluator import BatchEvaluator

# Initialize batch evaluator
batch_eval = BatchEvaluator(
    methods=['gradcam', 'lime', 'rise'],
    datasets=['cifar10'],
    corruption_types=['gaussian_noise', 'shot_noise'],
    severity_levels=[1, 3, 5]
)

# Run batch evaluation
batch_eval.run(output_dir='results/batch_evaluation')
```

## 🎓 Understanding the Results

### Robustness Score Interpretation

- **0.9 - 1.0**: Excellent robustness (minimal degradation)
- **0.7 - 0.9**: Good robustness (moderate degradation)
- **0.5 - 0.7**: Fair robustness (noticeable degradation)
- **< 0.5**: Poor robustness (significant degradation)

### Key Metrics

1. **Similarity-based**: How similar are explanations?
   - Pearson Correlation
   - Cosine Similarity
   - SSIM

2. **Localization-based**: Are important regions preserved?
   - IoU (Intersection over Union)
   - Rank Correlation

3. **Prediction-based**: How does model behavior change?
   - Confidence Difference
   - KL Divergence

## 🐛 Troubleshooting

### Issue: Out of Memory

```python
# Solution: Reduce batch size
evaluator = XAIEvaluator(method='gradcam', batch_size=8)
```

### Issue: Slow LIME Evaluation

```python
# Solution: Reduce number of samples
evaluator = XAIEvaluator(
    method='lime',
    lime_num_samples=500  # Default: 1000
)
```

### Issue: Missing Dataset

```bash
# Solution: Download required dataset
python datasets/DATASET_NAME/download.py
```

## 📚 Next Steps

- Read the full [Documentation](../README.md)
- Explore [Advanced Configuration](ADVANCED.md)
- Check out [Examples](../examples/)
- Join the [Discussion](https://github.com/YOUR_USERNAME/xai-robustness-benchmark/discussions)

## 💡 Tips

1. **Start small**: Begin with CIFAR-10 and GradCAM
2. **Use GPU**: Enable CUDA for 10-100x speedup
3. **Cache results**: Enable caching to avoid recomputation
4. **Monitor progress**: Use tqdm progress bars
5. **Save intermediate**: Set checkpoints for long runs

## 📧 Need Help?

- Check [FAQ](FAQ.md)
- Open an [Issue](https://github.com/YOUR_USERNAME/xai-robustness-benchmark/issues)
- Email: your.email@example.com

---

Happy benchmarking! 🚀

