#!/usr/bin/env python3
"""
详细分析XAI方法的鲁棒性结果
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd

def load_and_analyze_results():
    """加载并分析所有实验结果"""
    
    methods = ['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise']
    datasets = ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']
    model_types = ['standard', 'robust']
    
    # 存储每个方法的所有相似度值
    method_similarities = {method: [] for method in methods}
    
    # 存储详细统计
    detailed_stats = {}
    
    for method in methods:
        print(f"\n=== Analyzing {method.upper()} ===")
        method_data = []
        
        for dataset in datasets:
            for model_type in model_types:
                file_path = f'results/{dataset}/{method}/{method}_robustness_{model_type}_results.json'
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 统计该文件中的数据
                    similarities = []
                    for img_path, img_data in data.items():
                        for corruption, corruption_data in img_data.items():
                            for result in corruption_data['results']:
                                sim = result['similarity']
                                similarities.append(sim)
                                method_data.append({
                                    'dataset': dataset,
                                    'model': model_type,
                                    'corruption': corruption,
                                    'severity': result.get('severity', 'unknown'),
                                    'similarity': sim
                                })
                    
                    print(f"  {dataset}/{model_type}: {len(similarities)} measurements, "
                          f"mean={np.mean(similarities):.3f}, std={np.std(similarities):.3f}")
                    
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
        
        # 计算该方法的总体统计
        if method_data:
            df = pd.DataFrame(method_data)
            
            # 总体统计
            all_sims = df['similarity'].values
            method_similarities[method] = all_sims
            
            mean_sim = np.mean(all_sims)
            std_sim = np.std(all_sims)
            median_sim = np.median(all_sims)
            
            print(f"\n  Overall Statistics for {method.upper()}:")
            print(f"    Total samples: {len(all_sims)}")
            print(f"    Mean: {mean_sim:.4f}")
            print(f"    Std: {std_sim:.4f}")
            print(f"    Median: {median_sim:.4f}")
            print(f"    Min: {np.min(all_sims):.4f}")
            print(f"    Max: {np.max(all_sims):.4f}")
            
            # 按严重程度分组
            severity_stats = df.groupby('severity')['similarity'].agg(['mean', 'std', 'count'])
            print(f"\n  By Severity:")
            print(severity_stats)
            
            # 按数据集分组
            dataset_stats = df.groupby('dataset')['similarity'].agg(['mean', 'std', 'count'])
            print(f"\n  By Dataset:")
            print(dataset_stats)
            
            # 按模型类型分组
            model_stats = df.groupby('model')['similarity'].agg(['mean', 'std', 'count'])
            print(f"\n  By Model Type:")
            print(model_stats)
            
            detailed_stats[method] = {
                'mean': mean_sim,
                'std': std_sim,
                'median': median_sim,
                'count': len(all_sims)
            }
    
    # 打印最终排名
    print("\n" + "="*50)
    print("FINAL RANKING (by mean similarity):")
    print("="*50)
    
    ranking = sorted(detailed_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for rank, (method, stats) in enumerate(ranking, 1):
        print(f"#{rank} {method.upper():10s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(median={stats['median']:.4f}, n={stats['count']})")
    
    # 创建结果表格（用于论文）
    print("\n" + "="*50)
    print("TABLE FORMAT FOR PAPER:")
    print("="*50)
    print("Method\t\tMean ± Std\tMedian\tSamples")
    print("-"*50)
    for rank, (method, stats) in enumerate(ranking, 1):
        method_name = method.upper() if method != 'ig' else 'IG'
        print(f"{method_name:10s}\t{stats['mean']:.2f} ± {stats['std']:.2f}\t"
              f"{stats['median']:.2f}\t{stats['count']}")
    
    return detailed_stats, method_similarities

def check_data_quality(method_similarities):
    """检查数据质量和异常值"""
    print("\n" + "="*50)
    print("DATA QUALITY CHECK:")
    print("="*50)
    
    for method, sims in method_similarities.items():
        if len(sims) > 0:
            sims = np.array(sims)
            
            # 检查异常值
            q1 = np.percentile(sims, 25)
            q3 = np.percentile(sims, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = sims[(sims < lower_bound) | (sims > upper_bound)]
            
            print(f"\n{method.upper()}:")
            print(f"  Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
            print(f"  Normal range: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(f"  Outliers: {len(outliers)} out of {len(sims)} ({100*len(outliers)/len(sims):.1f}%)")
            
            if len(outliers) > 0:
                print(f"  Outlier values: min={np.min(outliers):.3f}, max={np.max(outliers):.3f}")

def main():
    print("Analyzing XAI Robustness Results")
    print("="*50)
    
    # 分析结果
    detailed_stats, method_similarities = load_and_analyze_results()
    
    # 检查数据质量
    check_data_quality(method_similarities)
    
    # 保存统计结果
    with open('results/robustness_statistics.json', 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to results/robustness_statistics.json")

if __name__ == "__main__":
    main()