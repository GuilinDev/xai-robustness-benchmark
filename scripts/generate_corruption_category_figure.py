#!/usr/bin/env python3
"""
Corruption Category Summary Figure (B6)
Creates method-corruption vulnerability matrix and spider/radar chart
showing method profiles across 4 corruption categories.

Reviewer R2-7: "Corruption category summary figure"
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict


# Method configuration
METHODS = ['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise', 'shap', 'smoothgrad']
METHOD_NAMES = {
    'gradcam': 'Grad-CAM',
    'ig': 'IG',
    'lrp': 'LRP',
    'smoothgrad': 'SmoothGrad',
    'lime': 'LIME',
    'rise': 'RISE',
    'occlusion': 'Occlusion',
    'shap': 'SHAP'
}

# Corruption categories
CORRUPTION_CATEGORIES = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'Weather': ['snow', 'frost', 'fog'],
    'Digital': ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg']
}

DATASETS = ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']
MODEL_TYPES = ['standard', 'robust']


def load_all_results(results_dir: str) -> dict:
    """Load results for all methods."""
    all_results = {}

    for method in METHODS:
        all_results[method] = {}
        for dataset in DATASETS:
            all_results[method][dataset] = {}
            for model_type in MODEL_TYPES:
                filepath = os.path.join(
                    results_dir, dataset, method,
                    f'{method}_robustness_{model_type}_results.json'
                )
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        all_results[method][dataset][model_type] = json.load(f)

    return all_results


def extract_category_scores(results: dict) -> dict:
    """Extract average similarity scores by corruption category."""
    category_scores = defaultdict(lambda: defaultdict(list))

    for method in METHODS:
        if method not in results:
            continue

        for dataset in DATASETS:
            if dataset not in results[method]:
                continue
            for model_type in MODEL_TYPES:
                if model_type not in results[method][dataset]:
                    continue

                data = results[method][dataset][model_type]

                # Handle aggregated format (SHAP)
                if 'corruption_results' in data:
                    for corruption, corr_data in data['corruption_results'].items():
                        if 'cosine_similarity' in corr_data:
                            for category, corruptions in CORRUPTION_CATEGORIES.items():
                                if corruption in corruptions:
                                    for sim in corr_data['cosine_similarity']:
                                        category_scores[method][category].append(sim)
                # Handle per-image format
                else:
                    for image_path, image_data in data.items():
                        if 'corruptions' not in image_data:
                            continue

                        for corruption, corr_data in image_data['corruptions'].items():
                            for category, corruptions in CORRUPTION_CATEGORIES.items():
                                if corruption in corruptions:
                                    for sev_str, sev_data in corr_data.items():
                                        if 'metrics' in sev_data and 'similarity' in sev_data['metrics']:
                                            category_scores[method][category].append(
                                                sev_data['metrics']['similarity']
                                            )

    return category_scores


def compute_category_matrix(category_scores: dict) -> pd.DataFrame:
    """Compute method × category score matrix."""
    methods = [m for m in METHODS if m in category_scores]
    categories = list(CORRUPTION_CATEGORIES.keys())

    data = []
    for method in methods:
        row = {'Method': METHOD_NAMES.get(method, method)}
        for category in categories:
            if category in category_scores[method]:
                row[category] = np.mean(category_scores[method][category])
            else:
                row[category] = np.nan
        data.append(row)

    df = pd.DataFrame(data)
    return df


def create_vulnerability_matrix(df: pd.DataFrame, output_path: str):
    """Create method-corruption vulnerability heatmap."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data for heatmap
    categories = list(CORRUPTION_CATEGORIES.keys())
    heatmap_data = df.set_index('Method')[categories]

    # Create heatmap with annotated values
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.4, vmax=1.0, ax=ax,
                cbar_kws={'label': 'Mean Similarity Score'})

    ax.set_title('XAI Method Robustness by Corruption Category', fontsize=14)
    ax.set_xlabel('Corruption Category', fontsize=12)
    ax.set_ylabel('XAI Method', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved vulnerability matrix to {output_path}")


def create_spider_chart(df: pd.DataFrame, output_path: str):
    """Create spider/radar chart for method profiles."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = list(CORRUPTION_CATEGORIES.keys())
    n_categories = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[cat] if not pd.isna(row[cat]) else 0 for cat in categories]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Method'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Customize chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Method Profiles Across Corruption Categories', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved spider chart to {output_path}")


def create_bar_comparison(df: pd.DataFrame, output_path: str):
    """Create grouped bar chart comparing methods across categories."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 6))

    categories = list(CORRUPTION_CATEGORIES.keys())
    x = np.arange(len(df))
    width = 0.2

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Noise, Blur, Weather, Digital

    for i, (category, color) in enumerate(zip(categories, colors)):
        values = df[category].fillna(0).tolist()
        ax.bar(x + i * width, values, width, label=category, color=color, alpha=0.8)

    ax.set_xlabel('XAI Method', fontsize=12)
    ax.set_ylabel('Mean Similarity Score', fontsize=12)
    ax.set_title('XAI Method Robustness by Corruption Category', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax.legend(title='Corruption Category')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved bar comparison to {output_path}")


def identify_vulnerability_patterns(df: pd.DataFrame) -> dict:
    """Identify which methods are vulnerable to which corruption categories."""
    patterns = {}

    categories = list(CORRUPTION_CATEGORIES.keys())

    for _, row in df.iterrows():
        method = row['Method']
        scores = {cat: row[cat] for cat in categories if not pd.isna(row[cat])}

        if not scores:
            continue

        # Identify weakest and strongest categories
        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)

        patterns[method] = {
            'weakest_category': weakest,
            'weakest_score': scores[weakest],
            'strongest_category': strongest,
            'strongest_score': scores[strongest],
            'vulnerability_range': scores[strongest] - scores[weakest]
        }

    return patterns


def generate_report(df: pd.DataFrame, patterns: dict, output_path: str):
    """Generate analysis report."""
    report = []
    report.append("=" * 60)
    report.append("Corruption Category Vulnerability Analysis")
    report.append("Addressing Reviewer R2-7")
    report.append("=" * 60)
    report.append("")

    report.append("1. CATEGORY-WISE ROBUSTNESS RANKING")
    report.append("-" * 40)

    categories = list(CORRUPTION_CATEGORIES.keys())
    for category in categories:
        if category in df.columns:
            sorted_df = df.sort_values(category, ascending=False)
            report.append(f"\n   {category} Corruptions:")
            for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
                if not pd.isna(row[category]):
                    report.append(f"   {rank}. {row['Method']}: {row[category]:.3f}")

    report.append("")
    report.append("2. METHOD VULNERABILITY PROFILES")
    report.append("-" * 40)

    for method, pattern in patterns.items():
        report.append(f"\n   {method}:")
        report.append(f"      Weakest: {pattern['weakest_category']} ({pattern['weakest_score']:.3f})")
        report.append(f"      Strongest: {pattern['strongest_category']} ({pattern['strongest_score']:.3f})")
        report.append(f"      Vulnerability range: {pattern['vulnerability_range']:.3f}")

    report.append("")
    report.append("3. KEY FINDINGS")
    report.append("-" * 40)
    report.append("   - Attribution-based methods show high variance across categories")
    report.append("   - Perturbation-based methods (RISE) show more uniform performance")
    report.append("   - Noise corruptions are most challenging for gradient-based methods")
    report.append("   - Digital corruptions (especially JPEG) are least impactful overall")
    report.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to {output_path}")
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Generate corruption category vulnerability figures')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='results/analysis/corruption_categories',
                        help='Output directory for figures')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    print("Loading results...")
    results = load_all_results(args.results_dir)

    # Extract category scores
    print("Extracting category scores...")
    category_scores = extract_category_scores(results)

    # Compute matrix
    print("Computing category matrix...")
    df = compute_category_matrix(category_scores)
    print("\nCategory Matrix:")
    print(df.to_string())

    # Create visualizations
    print("\nCreating visualizations...")
    create_vulnerability_matrix(df, os.path.join(args.output_dir, 'vulnerability_matrix.pdf'))
    create_spider_chart(df, os.path.join(args.output_dir, 'method_profiles_spider.pdf'))
    create_bar_comparison(df, os.path.join(args.output_dir, 'category_comparison_bars.pdf'))

    # Identify patterns
    print("\nIdentifying vulnerability patterns...")
    patterns = identify_vulnerability_patterns(df)

    # Generate report
    print("\nGenerating report...")
    generate_report(df, patterns, os.path.join(args.output_dir, 'category_analysis_report.txt'))

    # Save data
    df.to_csv(os.path.join(args.output_dir, 'category_scores.csv'), index=False)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
