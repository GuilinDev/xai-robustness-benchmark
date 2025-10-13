#!/usr/bin/env python3
"""
Unified Corruption Implementation
Ensures consistent corruption application across all methods
"""

import numpy as np
from PIL import Image
from imagecorruptions import corrupt, get_corruption_names


# Standard ImageNet-C corruption types
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# Severity levels (1-5)
SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def apply_corruption(image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    """
    Apply corruption to an image
    
    Args:
        image: PIL Image
        corruption_type: Type of corruption from CORRUPTION_TYPES
        severity: Severity level (1-5)
        
    Returns:
        Corrupted PIL Image
    """
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"Invalid severity level: {severity}. Must be in {SEVERITY_LEVELS}")
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Apply corruption
    corrupted_array = corrupt(img_array, corruption_name=corruption_type, severity=severity)
    
    # Convert back to PIL
    corrupted_image = Image.fromarray(corrupted_array.astype(np.uint8))
    
    return corrupted_image


def get_corruption_info():
    """Get information about available corruptions"""
    return {
        'types': CORRUPTION_TYPES,
        'severities': SEVERITY_LEVELS,
        'total_corruptions': len(CORRUPTION_TYPES) * len(SEVERITY_LEVELS)
    }


def verify_corruptions():
    """Verify that all corruptions are working properly"""
    # Create a test image
    test_img = Image.new('RGB', (224, 224), color='red')
    
    failed_corruptions = []
    
    for corruption in CORRUPTION_TYPES:
        for severity in SEVERITY_LEVELS:
            try:
                corrupted = apply_corruption(test_img, corruption, severity)
                if corrupted.size != test_img.size:
                    failed_corruptions.append(f"{corruption}_s{severity}")
            except Exception as e:
                failed_corruptions.append(f"{corruption}_s{severity}: {str(e)}")
    
    if failed_corruptions:
        print(f"Failed corruptions: {failed_corruptions}")
        return False
    else:
        print("All corruptions verified successfully!")
        return True