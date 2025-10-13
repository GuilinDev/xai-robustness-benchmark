#!/usr/bin/env python3
"""
Download script for CIFAR-10 (500 selected images)
Downloads and extracts exactly 500 images (50 per class) from CIFAR-10 test set
"""

import os
import pickle
import random
import numpy as np
from PIL import Image
import urllib.request
import tarfile

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def download_cifar10():
    """Download CIFAR-10 dataset"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    if not os.path.exists(filename):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)
    
    print("Extracting...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    
    return "cifar-10-batches-py"

def unpickle(file):
    """Load CIFAR-10 batch"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_image(data, filename):
    """Convert CIFAR-10 array to image and save"""
    # Reshape from (3072,) to (3, 32, 32)
    img_array = data.reshape(3, 32, 32)
    # Transpose to (32, 32, 3)
    img_array = img_array.transpose(1, 2, 0)
    # Create PIL image
    img = Image.fromarray(img_array)
    img.save(filename)

def main():
    print("=== CIFAR-10 Download Script ===")
    print("This script downloads only 500 pre-selected images")
    
    # Step 1: Download CIFAR-10
    cifar_dir = download_cifar10()
    
    # Step 2: Load test batch
    print("Loading test batch...")
    test_batch = unpickle(os.path.join(cifar_dir, 'test_batch'))
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    # Step 3: Group by class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(test_labels):
        class_indices[label].append(idx)
    
    # Step 4: Select 50 images per class
    print("Selecting 500 images (50 per class)...")
    os.makedirs("images", exist_ok=True)
    os.makedirs("lists", exist_ok=True)
    
    selected_images = []
    
    with open("lists/selected_images.txt", "w") as f_list:
        with open("lists/class_labels.txt", "w") as f_labels:
            
            img_counter = 0
            for class_id in range(10):
                # Randomly select 50 images from this class
                indices = random.sample(class_indices[class_id], 50)
                
                for idx in indices:
                    # Get image data
                    img_data = test_data[idx]
                    class_name = CLASSES[class_id]
                    
                    # Create filename
                    filename = f"cifar10_{img_counter:05d}_{class_name}_{idx:05d}.png"
                    filepath = os.path.join("images", filename)
                    
                    # Save image
                    save_image(img_data, filepath)
                    
                    # Write to lists
                    f_list.write(f"{filename}\n")
                    f_labels.write(f"{filename}\t{class_id}\t{class_name}\n")
                    
                    img_counter += 1
                    
                print(f"  Processed class {class_name}: 50 images")
    
    # Step 5: Cleanup
    print("Cleaning up...")
    os.remove("cifar-10-python.tar.gz")
    import shutil
    shutil.rmtree(cifar_dir)
    
    print("=== Download complete! ===")
    print("You now have exactly 500 CIFAR-10 images in ./images/")
    print("  - 50 images per class")
    print("  - Converted to PNG format")
    print("  - Original resolution: 32x32")

if __name__ == "__main__":
    main()