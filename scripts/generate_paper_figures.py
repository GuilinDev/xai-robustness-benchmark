#!/usr/bin/env python3
"""
Generate paper figures for Applied Intelligence major revision.
8 XAI methods with pastel color palette.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-paper')

# Global pastel color palette for 8 methods
PASTEL_COLORS = {
    'gradcam': '#7EB8DA',
    'ig': '#FFB87A',
    'lrp': '#82D9A5',
    'smoothgrad': '#C4A7E7',
    'lime': '#FF9B9B',
    'rise': '#A8D5BA',
    'occlusion': '#F0B6D4',
    'shap': '#B8B8B8',
}

# Canonical method order and display names
METHOD_LIST = ['gradcam', 'ig', 'lrp', 'smoothgrad', 'lime', 'occlusion', 'rise', 'shap']
METHOD_NAMES = {
    'gradcam': 'GradCAM', 'ig': 'IG', 'lrp': 'LRP', 'smoothgrad': 'SmoothGrad',
    'lime': 'LIME', 'occlusion': 'Occlusion', 'rise': 'RISE', 'shap': 'SHAP',
}
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
DATASETS = ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']


def load_results():
    """Load all experiment results (8 methods)."""
    results = {}
    model_types = ['standard', 'robust']

    for method in METHOD_LIST:
        results[method] = {}
        for dataset in DATASETS:
            results[method][dataset] = {}
            for model_type in model_types:
                file_path = f'results/{dataset}/{method}/{method}_robustness_{model_type}_results.json'
                # Also try shap_captum format
                if method == 'shap':
                    captum_path = f'results/{dataset}/shap/shap_captum_{model_type}_results.json'
                    if Path(captum_path).exists():
                        file_path = captum_path
                try:
                    with open(file_path, 'r') as f:
                        results[method][dataset][model_type] = json.load(f)
                except FileNotFoundError:
                    print(f"Warning: {file_path} not found, skipping...")
                    results[method][dataset][model_type] = None

    return results


def get_corruption_types(results):
    """Get list of corruption types from results."""
    for method in ['gradcam', 'ig', 'lrp', 'smoothgrad', 'lime', 'occlusion', 'rise']:
        if method in results and results[method]['cifar-10']['standard'] is not None:
            data = results[method]['cifar-10']['standard']
            for img_path, img_data in data.items():
                if isinstance(img_data, dict) and 'results' not in img_data:
                    return list(img_data.keys())
                elif isinstance(img_data, dict) and isinstance(list(img_data.values())[0], dict):
                    first_key = list(img_data.keys())[0]
                    if 'results' in img_data[first_key]:
                        return list(img_data.keys())
    return ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg']


def extract_similarities(data, method, corruption, severity_idx=2):
    """Extract similarity scores from various result formats."""
    similarities = []

    if data is None:
        return similarities

    # Handle SHAP aggregated format
    if 'corruption_results' in data:
        if corruption in data['corruption_results']:
            corr_data = data['corruption_results'][corruption]
            if 'cosine_similarity' in corr_data:
                sims = corr_data['cosine_similarity']
                step = len(sims) // 5 if len(sims) >= 5 else 1
                if len(sims) > severity_idx * step:
                    similarities.extend(sims[severity_idx * step:(severity_idx + 1) * step])
                else:
                    similarities.extend(sims)
        return similarities

    # Handle per-image format: {image_path: {corruption: {results: [...]}}}
    for img_path, img_data in data.items():
        if not isinstance(img_data, dict):
            continue
        if corruption in img_data:
            corr_data = img_data[corruption]
            if 'results' in corr_data and len(corr_data['results']) > severity_idx:
                result = corr_data['results'][severity_idx]
                if 'similarity' in result:
                    similarities.append(result['similarity'])

    return similarities


def extract_all_similarities(data, method):
    """Extract all similarity scores from a result dict."""
    similarities = []
    if data is None:
        return similarities

    # Handle SHAP aggregated format
    if 'corruption_results' in data:
        for corruption, corr_data in data['corruption_results'].items():
            if 'cosine_similarity' in corr_data:
                similarities.extend(corr_data['cosine_similarity'])
        return similarities

    # Handle per-image format
    for img_path, img_data in data.items():
        if not isinstance(img_data, dict):
            continue
        for corruption, corr_data in img_data.items():
            if isinstance(corr_data, dict) and 'results' in corr_data:
                for result in corr_data['results']:
                    if 'similarity' in result:
                        similarities.append(result['similarity'])
    return similarities


def create_corruption_heatmap(results, output_dir='paper/figures'):
    """Create corruption sensitivity heatmap with pastel colormap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    corruptions = get_corruption_types(results)
    heatmap_data = []

    # Filter to methods that have data
    available_methods = []
    available_names = []

    for method in METHOD_LIST:
        if method not in results:
            continue
        has_data = False
        for dataset in DATASETS:
            for model_type in ['standard', 'robust']:
                if results[method][dataset][model_type] is not None:
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            available_methods.append(method)
            available_names.append(METHOD_NAMES[method])

    for method in available_methods:
        method_scores = []
        for corruption in corruptions:
            similarities = []
            for dataset in DATASETS:
                for model_type in ['standard', 'robust']:
                    sims = extract_similarities(
                        results[method][dataset][model_type],
                        method, corruption, severity_idx=2
                    )
                    similarities.extend(sims)
            method_scores.append(np.mean(similarities) if similarities else 0)
        heatmap_data.append(method_scores)

    fig, ax = plt.subplots(figsize=(14, 6))

    df = pd.DataFrame(heatmap_data,
                     index=available_names,
                     columns=[c.replace('_', ' ').title() for c in corruptions])

    # Pastel-friendly diverging colormap
    cmap = sns.light_palette("#82D9A5", as_cmap=True)
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlGnBu',
                vmin=0.3, vmax=1.0, cbar_kws={'label': 'Similarity Score'},
                ax=ax, linewidths=0.5, linecolor='white')

    ax.set_title('XAI Method Robustness to Different Corruption Types (Severity Level 3)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Corruption Type', fontsize=14)
    ax.set_ylabel('XAI Method', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/corruption_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/corruption_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/corruption_heatmap.pdf")


def create_severity_curves(results, output_dir='paper/figures'):
    """Create severity progression curves for all 8 methods with pastel colors."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    corruption = 'gaussian_noise'
    dataset = 'cifar-10'
    model_type = 'standard'

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(METHOD_LIST):
        if method not in results or results[method][dataset][model_type] is None:
            continue

        severities = []
        similarities = []

        for severity_idx in range(5):
            sims = extract_similarities(
                results[method][dataset][model_type],
                method, corruption, severity_idx=severity_idx
            )

            if sims:
                severities.append(severity_idx + 1)
                similarities.append(np.mean(sims))

        if severities and similarities:
            label = METHOD_NAMES[method]
            ax.plot(severities, similarities, marker=MARKERS[i], label=label,
                    linewidth=2.5, markersize=8, color=PASTEL_COLORS[method])

    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Similarity Score', fontsize=14)
    ax.set_title('Explanation Robustness vs. Gaussian Noise Severity', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim([0.30, 1.02])
    ax.grid(True, alpha=0.3)

    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/severity_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/severity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/severity_curves.pdf")


def create_model_comparison(results, output_dir='paper/figures'):
    """Create standard vs robust model comparison for all 8 methods with pastel bars."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    available_methods = []
    standard_scores = []
    robust_scores = []

    for method in METHOD_LIST:
        if method not in results:
            continue
        std_similarities = []
        rob_similarities = []

        for dataset in DATASETS:
            std_sims = extract_all_similarities(results[method][dataset]['standard'], method)
            rob_sims = extract_all_similarities(results[method][dataset]['robust'], method)
            std_similarities.extend(std_sims)
            rob_similarities.extend(rob_sims)

        if std_similarities or rob_similarities:
            available_methods.append(method)
            standard_scores.append(np.mean(std_similarities) if std_similarities else 0)
            robust_scores.append(np.mean(rob_similarities) if rob_similarities else 0)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(available_methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_scores, width, label='Standard Model', color='#7EB8DA', edgecolor='white')
    bars2 = ax.bar(x + width/2, robust_scores, width, label='Robust Model', color='#C4A7E7', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('XAI Method', fontsize=14)
    ax.set_ylabel('Average Similarity Score', fontsize=14)
    ax.set_title('Explanation Robustness: Standard vs. Robust Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in available_methods])
    ax.legend()
    ax.set_ylim([0.0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/model_comparison.pdf")


def create_ranking_visualization(results, output_dir='paper/figures'):
    """Create method ranking visualization for all 8 methods with pastel colors."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    method_scores = {}

    print("\nComputing robustness scores for all 8 methods...")
    for method in METHOD_LIST:
        if method not in results:
            continue
        all_similarities = []
        for dataset in DATASETS:
            for model_type in ['standard', 'robust']:
                sims = extract_all_similarities(results[method][dataset][model_type], method)
                all_similarities.extend(sims)

        if all_similarities:
            mean_score = np.mean(all_similarities)
            std_score = np.std(all_similarities)
            method_scores[method] = (mean_score, std_score)
            print(f"  {METHOD_NAMES[method]}: {mean_score:.3f} +/- {std_score:.3f}")
        else:
            print(f"  {METHOD_NAMES[method]}: No data available")

    # Sort by score (high to low)
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1][0], reverse=True)

    methods = [m[0] for m in sorted_methods]
    scores = [m[1][0] for m in sorted_methods]
    stds = [m[1][1] for m in sorted_methods]
    method_labels = [METHOD_NAMES[m] for m in methods]

    print("\nRanking:")
    for i, (method, (score, std)) in enumerate(sorted_methods, 1):
        print(f"  #{i} {METHOD_NAMES[method]}: {score:.3f}")

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(methods))

    # Use per-method pastel colors
    bar_colors = [PASTEL_COLORS[m] for m in methods]

    bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor='white', linewidth=0.5)

    for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f} +/- {std:.3f}', va='center', ha='left', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_labels)
    ax.set_xlabel('Robustness Score (Mean Similarity)', fontsize=14)
    ax.set_title('Overall Robustness Ranking of XAI Methods', fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.15])

    ax.grid(True, alpha=0.3, axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(methods)):
        ax.text(0.05, i, f'#{i+1}', fontweight='bold', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ranking_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ranking_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/ranking_visualization.pdf")


def create_vit_comparison_figure(output_dir='paper/figures'):
    """Create CNN vs ViT comparison grouped bar chart from aggregated ViT results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load aggregated results
    vit_results_path = 'results/analysis/vit/vit_combined_results.json'
    if not Path(vit_results_path).exists():
        print(f"Warning: {vit_results_path} not found. Run aggregate_vit_results.py first.")
        return

    with open(vit_results_path) as f:
        agg = json.load(f)

    # Methods available in ViT study
    vit_methods = ['ig', 'gradcam', 'rise', 'smoothgrad', 'shap']
    method_labels = [METHOD_NAMES[m] for m in vit_methods]

    cnn_scores = []
    vit_scores = []

    for method in vit_methods:
        vit_stats = agg['vit']['overall'].get(method, {})
        vit_scores.append(vit_stats.get('mean', 0))

        # CNN standard model scores (from per_dataset new methods or load directly)
        if method in ['smoothgrad', 'shap']:
            cnn_stats = agg['cnn_new_methods']['per_dataset'].get(method, {}).get('standard', {})
            cnn_scores.append(cnn_stats.get('mean', 0))
        else:
            # For ig, gradcam, rise - compute CNN standard from original results
            cnn_sims = []
            for dataset in DATASETS:
                patterns = {
                    'ig': f'results/{dataset}/ig/ig_robustness_standard_results.json',
                    'gradcam': f'results/{dataset}/gradcam/gradcam_robustness_standard_results.json',
                    'rise': f'results/{dataset}/rise/rise_robustness_standard_results.json',
                }
                fpath = patterns.get(method)
                if fpath and Path(fpath).exists():
                    with open(fpath) as f:
                        data = json.load(f)
                    cnn_sims.extend(extract_all_similarities(data, method))
            cnn_scores.append(np.mean(cnn_sims) if cnn_sims else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(vit_methods))
    width = 0.35

    bars_cnn = ax.bar(x - width/2, cnn_scores, width, label='CNN (ResNet-50)',
                      color='#7EB8DA', edgecolor='white')
    bars_vit = ax.bar(x + width/2, vit_scores, width, label='ViT (ViT-B/16)',
                      color='#FFB87A', edgecolor='white')

    for bars in [bars_cnn, bars_vit]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('XAI Method', fontsize=14)
    ax.set_ylabel('Mean Similarity Score', fontsize=14)
    ax.set_title('Explanation Robustness: CNN vs. ViT Architecture', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim([0.0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/vit_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/vit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/vit_comparison.pdf")


def main():
    print("Loading experiment results...")
    results = load_results()

    print("\nGenerating paper figures (8 methods, pastel palette)...")
    create_corruption_heatmap(results)
    create_severity_curves(results)
    create_model_comparison(results)
    create_ranking_visualization(results)
    create_vit_comparison_figure()

    print("\nAll figures saved to paper/figures/")


if __name__ == "__main__":
    main()
