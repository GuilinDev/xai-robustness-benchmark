#!/bin/bash
# Download script for Tiny-ImageNet-200 (500 selected images)

echo "=== Tiny-ImageNet-200 Download Script ==="
echo "This script downloads only 500 pre-selected images"

# Create temp directory
TEMP_DIR="temp_download"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Step 1: Download the full dataset (we need to extract our 500 images)
echo "Step 1: Downloading Tiny-ImageNet-200..."
if [ ! -f "tiny-imagenet-200.zip" ]; then
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
else
    echo "Dataset already downloaded."
fi

# Step 2: Extract
echo "Step 2: Extracting..."
unzip -q tiny-imagenet-200.zip

# Step 3: Create selected images list
echo "Step 3: Creating selection list..."
cd ..

# Create Python script to select 500 images
cat > select_images.py << 'EOF'
import os
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Get all validation images
val_dir = "temp_download/tiny-imagenet-200/val/images"
all_images = sorted(os.listdir(val_dir))

# Read val_annotations.txt to get class labels
annotations = {}
with open("temp_download/tiny-imagenet-200/val/val_annotations.txt", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        img_name = parts[0]
        class_id = parts[1]
        annotations[img_name] = class_id

# Group images by class
class_images = {}
for img, class_id in annotations.items():
    if class_id not in class_images:
        class_images[class_id] = []
    class_images[class_id].append(img)

# Select 2-3 images per class
selected_images = []
for class_id, images in sorted(class_images.items()):
    # Randomly select 2-3 images per class
    n_select = min(len(images), random.choice([2, 3]))
    selected = random.sample(images, n_select)
    selected_images.extend([(img, class_id) for img in selected])

# Limit to exactly 500 images
random.shuffle(selected_images)
selected_images = selected_images[:500]

# Sort for consistency
selected_images.sort()

# Copy selected images
print(f"Copying {len(selected_images)} images...")
os.makedirs("images", exist_ok=True)
os.makedirs("lists", exist_ok=True)

with open("lists/selected_images.txt", "w") as f_list:
    with open("lists/class_labels.txt", "w") as f_labels:
        for idx, (img_name, class_id) in enumerate(selected_images):
            # Copy image with new name
            src = os.path.join(val_dir, img_name)
            new_name = f"tiny_imagenet_{idx:05d}_{img_name}"
            dst = os.path.join("images", new_name)
            shutil.copy2(src, dst)
            
            # Write to lists
            f_list.write(f"{new_name}\n")
            f_labels.write(f"{new_name}\t{class_id}\n")
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/500 images")

print("Done! Created:")
print(f"  - images/ directory with 500 images")
print(f"  - lists/selected_images.txt")
print(f"  - lists/class_labels.txt")
EOF

# Step 4: Run selection script
echo "Step 4: Selecting 500 images..."
python3 select_images.py

# Step 5: Cleanup
echo "Step 5: Cleaning up..."
rm -rf $TEMP_DIR
rm select_images.py

echo "=== Download complete! ==="
echo "You now have exactly 500 Tiny-ImageNet images in ./images/"