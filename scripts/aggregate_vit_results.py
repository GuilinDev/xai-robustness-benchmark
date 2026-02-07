#!/usr/bin/env python3
"""
Aggregate ViT results from two different JSON formats into unified statistics.

Main format (ig, gradcam, rise):
  results -> method -> corruption -> severity -> {similarities: [...], ...}

Extra format (smoothgrad, shap):
  per_image_results -> image -> method -> corruption -> {results: [{similarity, ...}]}

Also computes SmoothGrad and SHAP CNN values for paper tables.
"""

import json
import os
from pathlib import Path


def load_vit_main(dataset):
    """Load ViT main results (ig, gradcam, rise) with severities 1,3,5."""
    fpath = f'results/{dataset}/vit/vit_robustness_full_results.json'
    with open(fpath) as f:
        return json.load(f)


def load_vit_extra(dataset):
    """Load ViT extra results (smoothgrad, shap) with severities 1-5."""
    fpath = f'results/{dataset}/vit/vit_extra_methods_results.json'
    with open(fpath) as f:
        return json.load(f)


def extract_sims_main(data, method):
    """Extract all similarity values from main format."""
    sims = []
    for corruption in data['corruptions']:
        if corruption not in data['results'].get(method, {}):
            continue
        for sev, sev_data in data['results'][method][corruption].items():
            sims.extend(sev_data['similarities'])
    return sims


def extract_sims_extra(data, method):
    """Extract all similarity values from extra format."""
    sims = []
    for img_path, img_data in data['per_image_results'].items():
        if method not in img_data:
            continue
        for corruption, corr_data in img_data[method].items():
            if 'results' in corr_data:
                for result in corr_data['results']:
                    if 'similarity' in result:
                        sims.append(result['similarity'])
    return sims


def extract_sims_cnn(data):
    """Extract similarity values from CNN per-image format."""
    sims = []
    if data is None:
        return sims
    if 'corruption_results' in data:
        for corruption, corr_data in data['corruption_results'].items():
            if 'cosine_similarity' in corr_data:
                sims.extend(corr_data['cosine_similarity'])
        return sims
    for img_path, img_data in data.items():
        if not isinstance(img_data, dict):
            continue
        for corruption, corr_data in img_data.items():
            if isinstance(corr_data, dict) and 'results' in corr_data:
                for result in corr_data['results']:
                    if 'similarity' in result:
                        sims.append(result['similarity'])
    return sims


def compute_stats(values):
    """Compute mean and std of a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return {'mean': round(mean, 4), 'std': round(std, 4), 'n': n}


def main():
    datasets = ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']
    vit_main_methods = ['ig', 'gradcam', 'rise']
    vit_extra_methods = ['smoothgrad', 'shap']
    all_vit_methods = vit_main_methods + vit_extra_methods

    output = {
        'vit': {'per_dataset': {}, 'overall': {}},
        'cnn_new_methods': {'per_dataset': {}, 'overall': {}},
    }

    # --- ViT results ---
    vit_all_sims = {m: [] for m in all_vit_methods}

    for dataset in datasets:
        output['vit']['per_dataset'][dataset] = {}

        # Main methods
        main_data = load_vit_main(dataset)
        for method in vit_main_methods:
            sims = extract_sims_main(main_data, method)
            vit_all_sims[method].extend(sims)
            output['vit']['per_dataset'][dataset][method] = compute_stats(sims)

        # Extra methods
        extra_data = load_vit_extra(dataset)
        for method in vit_extra_methods:
            sims = extract_sims_extra(extra_data, method)
            vit_all_sims[method].extend(sims)
            output['vit']['per_dataset'][dataset][method] = compute_stats(sims)

    for method in all_vit_methods:
        output['vit']['overall'][method] = compute_stats(vit_all_sims[method])

    # --- CNN SmoothGrad and SHAP (captum) results ---
    cnn_patterns = {
        'smoothgrad': 'smoothgrad/smoothgrad_robustness',
        'shap': 'shap/shap_captum',
    }

    for method, pattern in cnn_patterns.items():
        output['cnn_new_methods']['per_dataset'][method] = {}
        for model_type in ['standard', 'robust']:
            all_sims = []
            for dataset in datasets:
                fpath = f'results/{dataset}/{pattern}_{model_type}_results.json'
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    sims = extract_sims_cnn(data)
                    all_sims.extend(sims)
                except FileNotFoundError:
                    print(f"  Warning: {fpath} not found")
            output['cnn_new_methods']['per_dataset'][method][model_type] = compute_stats(all_sims)

        # Overall (standard + robust combined)
        combined = []
        for model_type in ['standard', 'robust']:
            stats = output['cnn_new_methods']['per_dataset'][method][model_type]
            # We need to re-aggregate from files
            for dataset in datasets:
                fpath = f'results/{dataset}/{pattern}_{model_type}_results.json'
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    combined.extend(extract_sims_cnn(data))
                except FileNotFoundError:
                    pass
        output['cnn_new_methods']['overall'][method] = compute_stats(combined)

    # --- Save output ---
    out_dir = Path('results/analysis/vit')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'vit_combined_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # --- Print summary ---
    print("=== ViT Overall Mean Similarities ===")
    for method in all_vit_methods:
        stats = output['vit']['overall'][method]
        print(f"  {method:>12s}: {stats['mean']:.3f} +/- {stats['std']:.3f}  (n={stats['n']})")

    print("\n=== CNN New Methods (Standard / Robust) ===")
    for method in cnn_patterns:
        std_stats = output['cnn_new_methods']['per_dataset'][method]['standard']
        rob_stats = output['cnn_new_methods']['per_dataset'][method]['robust']
        improvement = (rob_stats['mean'] - std_stats['mean']) / std_stats['mean'] * 100 if std_stats['mean'] else 0
        print(f"  {method:>12s}: std={std_stats['mean']:.3f} +/- {std_stats['std']:.3f}, "
              f"rob={rob_stats['mean']:.3f} +/- {rob_stats['std']:.3f}, "
              f"delta={improvement:+.1f}%")

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
