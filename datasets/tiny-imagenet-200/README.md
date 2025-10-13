# Tiny-ImageNet-200 Dataset

## Overview

Tiny-ImageNet-200 is a subset of ImageNet with 200 classes, providing intermediate complexity between CIFAR-10 and full ImageNet. It's ideal for evaluating XAI robustness at moderate visual complexity.

## Dataset Characteristics

- **Classes**: 200 (subset of ImageNet-1k)
- **Images**: 
  - Training: 100,000 (500 per class)
  - Validation: 10,000 (50 per class)
  - Test: 10,000 (50 per class, no labels)
- **Image Size**: 64×64 RGB
- **Total Size**: ~250MB
- **Source**: Stanford CS231n

## Download Instructions

### Option 1: Automatic Download (Recommended)

```bash
python download.py
```

This will:
1. Download the dataset from Stanford (~250MB)
2. Extract all files
3. Organize validation images into class folders
4. Verify dataset integrity

### Option 2: Manual Download

1. Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
2. Extract to `./data/tiny-imagenet-200/`
3. Run organization script:
   ```bash
   python download.py --skip-download
   ```

## Directory Structure

After download and organization:

```
data/tiny-imagenet-200/
├── train/
│   ├── n01443537/          # Class ID (e.g., goldfish)
│   │   ├── images/
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── ...
│   │   │   └── n01443537_499.JPEG  # 500 images
│   │   └── n01443537_boxes.txt     # Bounding boxes
│   └── ... (200 classes)
│
├── val/
│   ├── n01443537/          # Organized by class
│   │   ├── val_*.JPEG      # 50 validation images
│   │   └── ...
│   └── ... (200 classes)
│
├── test/
│   └── images/             # 10,000 test images (no labels)
│
├── wnids.txt               # WordNet IDs (200 lines)
└── words.txt               # Class names mapping
```

## Sample Selection

For benchmark experiments, we use **1,000 images** sampled uniformly:
- **5 images per class** × 200 classes = 1,000 images
- Sampled from validation set
- Fixed seed (42) for reproducibility
- Stratified sampling ensures balanced class distribution

The sample list is provided in `lists/selected_images.txt`.

## Usage Example

```python
from common.unified_data_loader import load_dataset

# Load Tiny-ImageNet-200 with our unified loader
dataset = load_dataset(
    dataset_name='tiny-imagenet-200',
    data_dir='./data',
    split='val'
)

# Access images
for idx, (image, label) in enumerate(dataset):
    print(f"Image {idx}: shape={image.shape}, label={label}")
```

## Class Information

The 200 classes span various categories:
- Animals (dogs, cats, birds, fish, insects)
- Vehicles (cars, trucks, ships, aircraft)
- Objects (furniture, instruments, tools)
- Natural scenes (landscapes, plants)

Full class mappings are in `wnids.txt` and `words.txt`.

## Benchmark Role

In our robustness evaluation:
- **Position**: Intermediate complexity between CIFAR-10 and MS-COCO-2017
- **Purpose**: Evaluate how XAI methods scale with:
  - Increased number of classes (200 vs 10)
  - More diverse visual content
  - Higher resolution than CIFAR-10 (64×64 vs 32×32)

### Complexity Comparison

| Dataset | Image Size | Classes | Images | Complexity |
|---------|-----------|---------|--------|------------|
| CIFAR-10 | 32×32 | 10 | 1,000 | Low |
| **Tiny-ImageNet-200** | **64×64** | **200** | **1,000** | **Medium** |
| MS-COCO-2017 | Variable | 80 | 1,000 | High |

## Citation

If you use Tiny-ImageNet-200, please cite:

```bibtex
@misc{le2015tinyimagenet,
  title={Tiny ImageNet Visual Recognition Challenge},
  author={Le, Ya and Yang, Xuan},
  year={2015},
  howpublished={\url{https://tiny-imagenet.herokuapp.com/}}
}
```

## Troubleshooting

### Download Fails

If automatic download fails, manually download from:
- Primary: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- Mirror: Contact us for alternative links

### Disk Space

- Compressed: ~240MB
- Extracted: ~250MB
- Total needed: ~500MB (during extraction)

### Validation Images Not Organized

Run the organization step manually:
```python
from download import organize_validation_images
organize_validation_images('./data')
```

## References

- Stanford CS231n: http://cs231n.stanford.edu/
- ImageNet: http://www.image-net.org/
- Project Page: https://tiny-imagenet.herokuapp.com/

---

**Note**: For our benchmark, you only need the validation set (10,000 images), from which we sample 1,000 images for evaluation.

