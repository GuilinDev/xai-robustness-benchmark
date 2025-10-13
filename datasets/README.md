# Datasets

This directory contains download scripts and sample lists for the three datasets used in the XAI robustness benchmark.

## 📊 Dataset Overview

| Dataset | Image Size | Classes | Samples | Complexity | Download Size |
|---------|-----------|---------|---------|------------|---------------|
| **CIFAR-10** | 32×32 | 10 | 1,000 | Low | ~170MB |
| **Tiny-ImageNet-200** | 64×64 | 200 | 1,000 | Medium | ~250MB |
| **MS-COCO-2017** | Variable | 80 | 1,000 | High | ~19GB |

**Total**: 3,000 images for comprehensive robustness evaluation

---

## 🚀 Quick Start

### Download All Datasets

```bash
# CIFAR-10 (smallest, fastest)
python datasets/cifar-10/download.py

# Tiny-ImageNet-200 (medium complexity)
python datasets/tiny-imagenet-200/download.py

# MS-COCO-2017 (largest, most realistic)
python datasets/ms-coco-2017/download.py
```

### Download Specific Dataset

```bash
# Download only CIFAR-10 for quick testing
cd datasets/cifar-10
python download.py
```

---

## 📁 Directory Structure

```
datasets/
├── README.md                       # This file
│
├── cifar-10/
│   ├── download.py                 # Download script
│   ├── README.md                   # Dataset documentation
│   └── lists/
│       ├── class_labels.txt        # 10 class names
│       └── selected_images.txt     # 1,000 selected samples
│
├── tiny-imagenet-200/
│   ├── download.py                 # Download script
│   ├── README.md                   # Dataset documentation
│   └── lists/
│       └── selected_images.txt     # 1,000 selected samples
│
└── ms-coco-2017/
    ├── download.py                 # Download script
    ├── README.md                   # Dataset documentation
    └── lists/
        ├── image_info.txt          # Image metadata
        └── selected_images.txt     # 1,000 selected samples
```

---

## 📖 Dataset Details

### CIFAR-10

**Purpose**: Baseline evaluation with simple object categories

- **Image Size**: 32×32 RGB
- **Classes**: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Samples**: 1,000 (100 per class)
- **Characteristics**: 
  - Simple objects on plain backgrounds
  - Clear class boundaries
  - Minimal visual complexity
- **Use Case**: Establish baseline robustness patterns

📄 **Documentation**: [cifar-10/README.md](cifar-10/README.md)

---

### Tiny-ImageNet-200

**Purpose**: Intermediate complexity evaluation

- **Image Size**: 64×64 RGB
- **Classes**: 200 (ImageNet subset)
- **Samples**: 1,000 (5 per class)
- **Characteristics**:
  - Moderate visual diversity
  - Real-world objects
  - Balanced complexity
- **Use Case**: Bridge gap between simple and complex datasets

📄 **Documentation**: [tiny-imagenet-200/README.md](tiny-imagenet-200/README.md)

---

### MS-COCO-2017

**Purpose**: Realistic deployment scenario evaluation

- **Image Size**: Variable (resized to 224×224 for experiments)
- **Classes**: 80 object categories
- **Samples**: 1,000 (from validation set)
- **Characteristics**:
  - Complex multi-object scenes
  - Natural backgrounds
  - Real-world context
  - High visual complexity
- **Use Case**: Test robustness under realistic conditions

📄 **Documentation**: [ms-coco-2017/README.md](ms-coco-2017/README.md)

---

## 🎯 Sampling Strategy

All datasets use **unified stratified sampling**:

1. **Fixed Seed**: `seed=42` for reproducibility
2. **Balanced Classes**: Equal samples per class where possible
3. **Validation Split**: All samples from validation/test sets
4. **Sample Lists**: Pre-generated and version-controlled

### Why 1,000 Images Per Dataset?

- ✅ **Statistical Significance**: Large enough for robust conclusions
- ✅ **Computational Efficiency**: Manageable experiment runtime
- ✅ **Fair Comparison**: Equal samples across datasets
- ✅ **Reproducibility**: Fixed lists ensure consistency

### Sample Distribution

```
CIFAR-10:           100 images × 10 classes   = 1,000 images
Tiny-ImageNet-200:    5 images × 200 classes  = 1,000 images
MS-COCO-2017:      ~12 images × 80 classes    = 1,000 images
```

---

## 🔄 Data Loading

### Using Unified Data Loader

```python
from common.unified_data_loader import load_dataset

# Load any dataset with consistent API
dataset = load_dataset(
    dataset_name='cifar10',  # or 'tiny-imagenet-200', 'ms-coco-2017'
    data_dir='./data',
    split='test',
    transform=None
)

# Iterate over samples
for image, label in dataset:
    print(f"Image shape: {image.shape}, Label: {label}")
```

### Custom Loading

Each dataset directory contains:
- `lists/selected_images.txt`: Fixed sample list
- Custom loading logic in `download.py`

---

## 💾 Storage Requirements

### Minimum (Compressed Only)

- CIFAR-10: ~170MB
- Tiny-ImageNet-200: ~250MB
- MS-COCO-2017: ~19GB
- **Total**: ~19.5GB

### Full (Extracted)

- CIFAR-10: ~170MB
- Tiny-ImageNet-200: ~500MB  
- MS-COCO-2017: ~25GB
- **Total**: ~25.7GB

### Recommendations

- **For testing**: Download CIFAR-10 only (~170MB)
- **For full benchmark**: Download all three (~26GB)
- **SSD recommended**: For faster I/O during experiments

---

## 🔧 Troubleshooting

### Download Issues

**Problem**: Download fails or is slow

**Solutions**:
1. Check internet connection
2. Try download script with `--retry` flag
3. Manual download from official sources:
   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
   - Tiny-ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip
   - MS-COCO: https://cocodataset.org/#download

### Disk Space Issues

**Problem**: Not enough disk space

**Solutions**:
1. Download datasets one at a time
2. Remove compressed files after extraction
3. Use external storage
4. Start with CIFAR-10 only for testing

### Dataset Not Found

**Problem**: Experiments fail with "dataset not found"

**Solutions**:
1. Verify dataset downloaded: `ls data/`
2. Check data directory path in config
3. Re-run download script
4. Check error messages in download script

---

## 📊 Benchmark Statistics

### Total Evaluations

```
3 datasets × 1,000 images × 15 corruptions × 5 severities × 6 XAI methods
= 1,350,000 explanation comparisons
```

### Computational Requirements

- **Storage**: ~26GB (all datasets)
- **GPU Memory**: 4-8GB (depending on method)
- **Runtime**: ~24-48 hours (full benchmark on single GPU)
- **CPU**: Works but 10-100× slower

---

## 📚 Citations

### CIFAR-10
```bibtex
@techreport{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex},
  year={2009}
}
```

### Tiny-ImageNet-200
```bibtex
@misc{le2015tinyimagenet,
  title={Tiny ImageNet Visual Recognition Challenge},
  author={Le, Ya and Yang, Xuan},
  year={2015}
}
```

### MS-COCO-2017
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and others},
  booktitle={ECCV},
  year={2014}
}
```

---

## 🤝 Contributing

To add a new dataset:

1. Create directory: `datasets/YOUR_DATASET/`
2. Add download script: `download.py`
3. Add documentation: `README.md`
4. Create sample list: `lists/selected_images.txt`
5. Update this README
6. Add to unified data loader

---

## 📧 Support

For dataset-related issues:
- Check individual dataset README files
- Open an issue on GitHub
- Email: your.email@example.com

---

**Ready to start?** Run `python datasets/cifar-10/download.py` for quick testing! 🚀

