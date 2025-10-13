#!/usr/bin/env python3
"""
生成论文所需的图表
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_results():
    """加载所有实验结果"""
    results = {}
    methods = ['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise']
    datasets = ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']
    model_types = ['standard', 'robust']
    
    for method in methods:
        results[method] = {}
        for dataset in datasets:
            results[method][dataset] = {}
            for model_type in model_types:
                file_path = f'results/{dataset}/{method}/{method}_robustness_{model_type}_results.json'
                with open(file_path, 'r') as f:
                    results[method][dataset][model_type] = json.load(f)
    
    return results

def create_corruption_heatmap(results, output_dir='paper/figures'):
    """创建corruption sensitivity热力图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有corruption类型
    first_method = list(results.keys())[0]
    first_dataset = 'cifar-10'
    first_model = 'standard'
    first_img = list(results[first_method][first_dataset][first_model].keys())[0]
    corruptions = list(results[first_method][first_dataset][first_model][first_img].keys())
    
    # 计算每个方法在每种corruption下的平均相似度
    heatmap_data = []
    
    for method in ['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise']:
        method_scores = []
        for corruption in corruptions:
            similarities = []
            for dataset in ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']:
                for model_type in ['standard', 'robust']:
                    for img_path, img_data in results[method][dataset][model_type].items():
                        if corruption in img_data:
                            # 取severity 3的结果（中等程度）
                            if len(img_data[corruption]['results']) >= 3:
                                similarities.append(img_data[corruption]['results'][2]['similarity'])
            method_scores.append(np.mean(similarities))
        heatmap_data.append(method_scores)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 创建DataFrame
    df = pd.DataFrame(heatmap_data, 
                     index=['GradCAM', 'IG', 'LRP', 'LIME', 'Occlusion', 'RISE'],
                     columns=[c.replace('_', ' ').title() for c in corruptions])
    
    # 绘制热力图
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'Similarity Score'},
                ax=ax)
    
    ax.set_title('XAI Method Robustness to Different Corruption Types (Severity Level 3)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Corruption Type', fontsize=14)
    ax.set_ylabel('XAI Method', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/corruption_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/corruption_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/corruption_heatmap.pdf")

def create_severity_curves(results, output_dir='paper/figures'):
    """创建severity progression曲线"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 选择代表性的corruption: gaussian_noise
    corruption = 'gaussian_noise'
    dataset = 'cifar-10'
    model_type = 'standard'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义颜色和标记样式，确保所有方法都可见
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, method in enumerate(['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise']):
        severities = []
        similarities = []
        
        for severity in range(1, 6):
            severity_similarities = []
            for img_path, img_data in results[method][dataset][model_type].items():
                if corruption in img_data and len(img_data[corruption]['results']) >= severity:
                    severity_similarities.append(
                        img_data[corruption]['results'][severity-1]['similarity']
                    )
            
            if severity_similarities:
                severities.append(severity)
                similarities.append(np.mean(severity_similarities))
        
        # 只绘制有数据的方法
        if severities and similarities:
            label = method.upper() if method != 'ig' else 'IG'
            ax.plot(severities, similarities, marker=markers[i], label=label, 
                    linewidth=2.5, markersize=8, color=colors[i])
    
    ax.set_xlabel('Corruption Severity', fontsize=14)  # 与其他图保持一致
    ax.set_ylabel('Similarity Score', fontsize=14)     # 与其他图保持一致
    ax.set_title('Explanation Robustness vs. Gaussian Noise Severity', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim([0.35, 1.02])  # 扩展y轴范围，包含所有数据点（Occlusion最低约0.36）
    ax.grid(True, alpha=0.3)
    
    # 调整图例位置到右下角
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, ncol=1)
    
    # 去掉顶部边框，避免与LRP线重叠
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/severity_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/severity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/severity_curves.pdf")

def create_model_comparison(results, output_dir='paper/figures'):
    """创建标准vs鲁棒模型对比图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 计算每个方法在两种模型下的平均相似度
    methods = ['gradcam', 'ig', 'lrp', 'lime', 'occlusion', 'rise']
    standard_scores = []
    robust_scores = []
    
    for method in methods:
        # Standard model scores
        std_similarities = []
        rob_similarities = []
        
        for dataset in ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']:
            for img_path, img_data in results[method][dataset]['standard'].items():
                for corruption, corruption_data in img_data.items():
                    for result in corruption_data['results']:
                        std_similarities.append(result['similarity'])
            
            for img_path, img_data in results[method][dataset]['robust'].items():
                for corruption, corruption_data in img_data.items():
                    for result in corruption_data['results']:
                        rob_similarities.append(result['similarity'])
        
        standard_scores.append(np.mean(std_similarities))
        robust_scores.append(np.mean(rob_similarities))
    
    # 创建对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, standard_scores, width, label='Standard Model', color='#2E86AB')
    bars2 = ax.bar(x + width/2, robust_scores, width, label='Robust Model', color='#A23B72')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('XAI Method', fontsize=14)  # 与其他图保持一致
    ax.set_ylabel('Average Similarity Score', fontsize=14)  # 与其他图保持一致
    ax.set_title('Explanation Robustness: Standard vs. Robust Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m != 'ig' else 'IG' for m in methods])
    ax.legend()
    ax.set_ylim([0.0, 1.1])  # 扩展范围，包含所有数据并留出空间给标签
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/model_comparison.pdf")

def create_ranking_visualization(results, output_dir='paper/figures'):
    """创建方法排名可视化"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 首先计算所有方法的得分
    method_scores = {}
    
    print("\n计算各方法的鲁棒性得分...")
    for method in ['rise', 'occlusion', 'gradcam', 'ig', 'lime', 'lrp']:
        all_similarities = []
        for dataset in ['cifar-10', 'tiny-imagenet-200', 'ms-coco-2017']:
            for model_type in ['standard', 'robust']:
                for img_path, img_data in results[method][dataset][model_type].items():
                    for corruption, corruption_data in img_data.items():
                        for result in corruption_data['results']:
                            all_similarities.append(result['similarity'])
        
        mean_score = np.mean(all_similarities)
        std_score = np.std(all_similarities)
        method_scores[method] = (mean_score, std_score)
        print(f"  {method.upper()}: {mean_score:.3f} ± {std_score:.3f}")
    
    # 根据得分排序（从高到低）
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    # 准备绘图数据
    methods = [m[0] for m in sorted_methods]
    scores = [m[1][0] for m in sorted_methods]
    stds = [m[1][1] for m in sorted_methods]
    method_labels = [m.upper() if m != 'ig' else 'IG' for m in methods]
    
    print("\n排名顺序:")
    for i, (method, (score, std)) in enumerate(sorted_methods, 1):
        print(f"  #{i} {method.upper()}: {score:.3f}")
    
    # 创建排名图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(methods))
    
    # 根据排名调整颜色深浅，前两名（perturbation）用绿色系，中间两名（attribution）用橙色系，最后两名用红色系
    color_shades = ['#27AE60', '#2ECC71', '#F39C12', '#E67E22', '#E74C3C', '#C0392B']
    
    bars = ax.barh(y_pos, scores, color=color_shades, alpha=0.8)
    
    # 添加数值标签（在条形图右侧）
    for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
        # 将数值标签放在条形图右端外侧
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f} ± {std:.3f}', va='center', ha='left', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_labels)
    ax.set_xlabel('Robustness Score (Mean Similarity ± Std)', fontsize=14)  # 与其他图保持一致
    ax.set_title('Overall Robustness Ranking of XAI Methods', fontsize=14, fontweight='bold')  # 与其他图保持一致
    ax.set_xlim([0.0, 1.1])  # 扩展范围，显示更自然的比例
    
    # 只保留x轴的网格线，更加细致
    ax.grid(True, alpha=0.3, axis='x')
    
    # 去掉顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加排名标签，位置更靠左避免重叠
    for i in range(len(methods)):
        ax.text(0.05, i, f'#{i+1}', fontweight='bold', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ranking_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ranking_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/ranking_visualization.pdf")

def main():
    print("加载实验结果...")
    results = load_results()
    
    print("\n生成论文图表...")
    create_corruption_heatmap(results)
    create_severity_curves(results)
    create_model_comparison(results)
    create_ranking_visualization(results)
    
    print("\n✓ 所有图表已生成到 paper/figures/ 目录")

if __name__ == "__main__":
    main()