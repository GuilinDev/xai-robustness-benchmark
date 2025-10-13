#!/usr/bin/env python3
"""
Summarize all 36 XAI robustness experiment results
"""

import json
import os
import pandas as pd
from pathlib import Path

def load_results(filepath):
    """Load and parse results JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics(data):
    """Extract average metrics from results"""
    if 'average_metrics' in data:
        return data['average_metrics']
    elif 'overall_metrics' in data:
        return data['overall_metrics']
    else:
        # Try to find metrics in the data
        for key in data:
            if 'cosine_similarity' in str(data):
                # Extract from various formats
                break
        return {}

def main():
    base_dir = '/home/guilin/allProjects/xai/experiments/results'
    
    # Methods and configurations
    methods = ['gradcam', 'ig', 'lrp', 'lime', 'shap', 'occlusion']
    datasets = ['tiny-imagenet-200', 'cifar-10', 'ms-coco-2017']
    model_types = ['standard', 'robust']
    
    # Collect all results
    results_table = []
    
    for dataset in datasets:
        for method in methods:
            for model_type in model_types:
                filepath = f"{base_dir}/{dataset}/{method}/{method}_robustness_{model_type}_results.json"
                
                if os.path.exists(filepath):
                    try:
                        data = load_results(filepath)
                        metrics = extract_metrics(data)
                        
                        # Extract key metrics
                        row = {
                            'Dataset': dataset,
                            'Method': method.upper(),
                            'Model': model_type.capitalize(),
                            'Cosine Similarity': 0,
                            'Prediction Change': 0,
                            'File Size (KB)': os.path.getsize(filepath) / 1024
                        }
                        
                        # Try different metric keys
                        if 'cosine_similarity' in metrics:
                            if isinstance(metrics['cosine_similarity'], dict):
                                row['Cosine Similarity'] = metrics['cosine_similarity'].get('mean', 0)
                            else:
                                row['Cosine Similarity'] = metrics['cosine_similarity']
                        
                        if 'prediction_change' in metrics:
                            if isinstance(metrics['prediction_change'], dict):
                                row['Prediction Change'] = metrics['prediction_change'].get('mean', 0)
                            else:
                                row['Prediction Change'] = metrics['prediction_change']
                        
                        results_table.append(row)
                        print(f"✓ {dataset}/{method}/{model_type}")
                    except Exception as e:
                        print(f"✗ Error processing {filepath}: {e}")
                else:
                    print(f"✗ Missing: {filepath}")
    
    # Create DataFrame
    df = pd.DataFrame(results_table)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY OF ALL 36 EXPERIMENTS")
    print("="*70)
    
    print(f"\nTotal experiments completed: {len(df)}/36")
    
    # Group by dataset
    print("\n--- By Dataset ---")
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Experiments: {len(dataset_df)}/12")
        print(f"  Avg Cosine Similarity: {dataset_df['Cosine Similarity'].mean():.3f}")
        print(f"  Avg Prediction Change: {dataset_df['Prediction Change'].mean():.3f}")
    
    # Group by method
    print("\n--- By Method ---")
    for method in methods:
        method_df = df[df['Method'] == method.upper()]
        print(f"\n{method.upper()}:")
        print(f"  Experiments: {len(method_df)}/6")
        print(f"  Avg Cosine Similarity: {method_df['Cosine Similarity'].mean():.3f}")
        print(f"  Avg Prediction Change: {method_df['Prediction Change'].mean():.3f}")
    
    # Group by model type
    print("\n--- By Model Type ---")
    for model_type in model_types:
        model_df = df[df['Model'] == model_type.capitalize()]
        print(f"\n{model_type.capitalize()}:")
        print(f"  Experiments: {len(model_df)}/18")
        print(f"  Avg Cosine Similarity: {model_df['Cosine Similarity'].mean():.3f}")
        print(f"  Avg Prediction Change: {model_df['Prediction Change'].mean():.3f}")
    
    # Save to CSV
    output_file = '/home/guilin/allProjects/xai/experiments/all_results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Create a pivot table for better visualization
    print("\n" + "="*70)
    print("COSINE SIMILARITY MATRIX (Higher is better)")
    print("="*70)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        dataset_df = df[df['Dataset'] == dataset]
        pivot = dataset_df.pivot_table(
            values='Cosine Similarity', 
            index='Method', 
            columns='Model',
            aggfunc='mean'
        )
        print(pivot.round(3))
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("All 36 experiments (6 methods × 2 models × 3 datasets) finished")
    print("="*70)

if __name__ == "__main__":
    main()