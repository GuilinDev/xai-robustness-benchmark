#!/usr/bin/env python3
"""
Download MS COCO 2017 validation images and select 500 for experiments
MS COCO has 80 object categories and complex multi-object scenes
"""

import os
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# MS COCO configuration
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TEMP_DIR = "temp_download"
TARGET_COUNT = 500

def download_file(url, filename):
    """Download file with progress indicator"""
    print(f"Downloading {filename}...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        print(f"Progress: {percent:.1f}%", end='\r')
    
    urllib.request.urlretrieve(url, filename, reporthook=download_progress)
    print("\nDownload complete!")

def main():
    print("=== MS COCO 2017 Download Script ===")
    print(f"This will download MS COCO validation set and select {TARGET_COUNT} images")
    
    # Create directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("lists", exist_ok=True)
    
    # Download annotations first (smaller file)
    print("\nStep 1: Downloading annotations...")
    annotations_zip = os.path.join(TEMP_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(annotations_zip):
        download_file(COCO_ANNOTATIONS_URL, annotations_zip)
    
    # Extract annotations
    print("\nExtracting annotations...")
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        # Only extract validation annotations
        for file in zip_ref.namelist():
            if 'instances_val2017.json' in file:
                zip_ref.extract(file, TEMP_DIR)
    
    # Load annotations
    print("\nLoading annotations...")
    with open(os.path.join(TEMP_DIR, "annotations/instances_val2017.json"), 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Found {len(categories)} object categories")
    
    # Analyze images by complexity
    print("\nAnalyzing image complexity...")
    image_info = {}
    image_annotations = defaultdict(list)
    
    # Group annotations by image
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Calculate complexity metrics for each image
    for img in coco_data['images']:
        img_id = img['id']
        anns = image_annotations[img_id]
        
        if not anns:
            continue
            
        # Get unique object categories
        obj_categories = list(set([ann['category_id'] for ann in anns]))
        obj_names = [categories[cat_id] for cat_id in obj_categories]
        
        # Calculate total object area
        total_area = sum([ann['area'] for ann in anns])
        img_area = img['width'] * img['height']
        area_ratio = total_area / img_area if img_area > 0 else 0
        
        image_info[img_id] = {
            'filename': img['file_name'],
            'width': img['width'],
            'height': img['height'],
            'n_objects': len(anns),
            'n_categories': len(obj_categories),
            'categories': obj_names,
            'area_ratio': area_ratio,
            'complexity_score': len(anns) * len(obj_categories) * area_ratio
        }
    
    print(f"Analyzed {len(image_info)} images with annotations")
    
    # Select diverse images
    print("\nSelecting diverse multi-object images...")
    
    # Group by number of objects
    by_n_objects = defaultdict(list)
    for img_id, info in image_info.items():
        n = min(info['n_objects'], 10)  # Cap at 10+ objects
        by_n_objects[n].append(img_id)
    
    # Select images with preference for multi-object scenes
    selected_images = []
    
    # Target distribution (prefer multi-object scenes)
    target_dist = {
        1: 50,   # Single object
        2: 80,   # Two objects
        3: 80,   # Three objects
        4: 70,   # Four objects
        5: 60,   # Five objects
        6: 50,   # Six objects
        7: 40,   # Seven objects
        8: 30,   # Eight objects
        9: 20,   # Nine objects
        10: 20   # Ten or more objects
    }
    
    for n_obj, target_count in target_dist.items():
        if n_obj in by_n_objects:
            available = by_n_objects[n_obj]
            # Sort by complexity score to get most interesting images
            available.sort(key=lambda x: image_info[x]['complexity_score'], reverse=True)
            n_select = min(len(available), target_count)
            selected_images.extend(available[:n_select])
            print(f"Selected {n_select} images with {n_obj}{'+'if n_obj==10 else ''} objects")
    
    # If we need more, add from the most complex images
    if len(selected_images) < TARGET_COUNT:
        all_complex = sorted(image_info.keys(), 
                           key=lambda x: image_info[x]['complexity_score'], 
                           reverse=True)
        for img_id in all_complex:
            if img_id not in selected_images:
                selected_images.append(img_id)
                if len(selected_images) >= TARGET_COUNT:
                    break
    
    # Ensure exactly 500
    selected_images = selected_images[:TARGET_COUNT]
    print(f"\nTotal selected: {len(selected_images)} images")
    
    # Download validation images if not already present
    val_zip = os.path.join(TEMP_DIR, "val2017.zip")
    if not os.path.exists(val_zip):
        print(f"\nStep 2: Downloading validation images (~1GB)...")
        print("This may take several minutes...")
        download_file(COCO_VAL_URL, val_zip)
    else:
        print("\nStep 2: Validation images already downloaded")
    
    # Extract only selected images
    print("\nStep 3: Extracting selected images...")
    selected_filenames = {image_info[img_id]['filename'] for img_id in selected_images}
    
    with zipfile.ZipFile(val_zip, 'r') as zip_ref:
        # Get all files in the zip
        all_files = zip_ref.namelist()
        extracted_count = 0
        
        for file in all_files:
            if file.endswith('.jpg'):
                filename = os.path.basename(file)
                if filename in selected_filenames:
                    # Extract to temp location
                    zip_ref.extract(file, TEMP_DIR)
                    extracted_count += 1
                    if extracted_count % 50 == 0:
                        print(f"  Extracted {extracted_count}/{TARGET_COUNT} images")
    
    # Copy and rename images
    print("\nStep 4: Organizing selected images...")
    
    # Sort by image ID for consistent ordering
    selected_images.sort()
    
    with open("lists/selected_images.txt", "w") as f_list:
        with open("lists/image_info.txt", "w") as f_info:
            f_info.write("filename\tn_objects\tn_categories\tcategories\n")
            
            for idx, img_id in enumerate(selected_images):
                info = image_info[img_id]
                
                # Copy image with new name
                src = os.path.join(TEMP_DIR, "val2017", info['filename'])
                new_name = f"mscoco_{idx:05d}_{img_id:012d}.jpg"
                dst = os.path.join("images", new_name)
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    
                    # Write to lists
                    f_list.write(f"{new_name}\n")
                    f_info.write(f"{new_name}\t{info['n_objects']}\t{info['n_categories']}\t{','.join(info['categories'])}\n")
                    
                    if (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{TARGET_COUNT} images")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    
    # Object count distribution
    obj_dist = defaultdict(int)
    cat_freq = defaultdict(int)
    
    for img_id in selected_images:
        info = image_info[img_id]
        n = min(info['n_objects'], 10)
        obj_dist[n] += 1
        for cat in info['categories']:
            cat_freq[cat] += 1
    
    print("\nImages by object count:")
    for n in sorted(obj_dist.keys()):
        print(f"  {n}{'+'if n==10 else ''} objects: {obj_dist[n]} images")
    
    print("\nTop 20 most frequent object categories:")
    sorted_cats = sorted(cat_freq.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats[:20]:
        print(f"  {cat}: {count} images")
    
    # Cleanup
    print("\nStep 5: Cleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    
    print("\n=== Download Complete! ===")
    print(f"Successfully downloaded and organized {TARGET_COUNT} MS COCO images")
    print("Files saved to:")
    print("  - images/: 500 selected images")
    print("  - lists/selected_images.txt: Image filenames")
    print("  - lists/image_info.txt: Object annotations")

if __name__ == "__main__":
    main()