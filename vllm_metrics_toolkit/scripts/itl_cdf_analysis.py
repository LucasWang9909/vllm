#!/usr/bin/env python3
"""
分析两个JSON文件中的itl_list_ms数据并绘制CDF对比图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from pathlib import Path

# 解决中文字体显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_json_file(file_path):
    """加载JSON文件"""
    print(f"正在加载文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"文件加载完成，包含 {data['metadata']['total_requests']} 个请求")
    return data

def extract_itl_data(json_data):
    """从JSON数据中提取所有itl_list_ms数据"""
    all_itl_values = []
    
    for request in json_data['requests']:
        if 'detailed_data' in request and 'itl_list_ms' in request['detailed_data']:
            itl_list = request['detailed_data']['itl_list_ms']
            all_itl_values.extend(itl_list)
    
    print(f"提取了 {len(all_itl_values)} 个ITL值")
    return np.array(all_itl_values)

def plot_violin_comparison(data1, data2, label1, label2, output_path=None):
    """绘制两组数据的小提琴图对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 准备数据
    data_combined = [data1, data2]
    labels = [label1, label2]
    colors = ['skyblue', 'lightcoral']
    
    # 绘制小提琴图
    violin_parts = ax1.violinplot(data_combined, positions=[1, 2], showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # 设置第一个子图属性
    ax1.set_xlabel('Benchmark', fontsize=12)
    ax1.set_ylabel('Inter-Token Latency (ms)', fontsize=12)
    ax1.set_title('ITL Distribution Comparison (Violin Plot)', fontsize=14, fontweight='bold')
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加箱线图叠加
    box_parts = ax1.boxplot(data_combined, positions=[1, 2], widths=0.1, 
                           patch_artist=True, showfliers=False)
    for i, patch in enumerate(box_parts['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
    
    # 第二个子图：对数刻度的小提琴图（更好地显示尾部分布）
    violin_parts2 = ax2.violinplot(data_combined, positions=[1, 2], showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for i, pc in enumerate(violin_parts2['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xlabel('Benchmark', fontsize=12)
    ax2.set_ylabel('Inter-Token Latency (ms) - Log Scale', fontsize=12)
    ax2.set_title('ITL Distribution Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 添加箱线图叠加（对数刻度）
    box_parts2 = ax2.boxplot(data_combined, positions=[1, 2], widths=0.1, 
                            patch_artist=True, showfliers=False)
    for i, patch in enumerate(box_parts2['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
    
    # 添加统计信息
    stats_text = f"""Statistics Summary:

{label1}:
  Samples: {len(data1):,}
  Mean: {np.mean(data1):.2f} ms
  Median: {np.median(data1):.2f} ms
  Std Dev: {np.std(data1):.2f} ms
  IQR: {np.percentile(data1, 75) - np.percentile(data1, 25):.2f} ms
  P95: {np.percentile(data1, 95):.2f} ms
  P99: {np.percentile(data1, 99):.2f} ms

{label2}:
  Samples: {len(data2):,}
  Mean: {np.mean(data2):.2f} ms
  Median: {np.median(data2):.2f} ms
  Std Dev: {np.std(data2):.2f} ms
  IQR: {np.percentile(data2, 75) - np.percentile(data2, 25):.2f} ms
  P95: {np.percentile(data2, 95):.2f} ms
  P99: {np.percentile(data2, 99):.2f} ms

Relative Differences ({label2} vs {label1}):
  Mean: {((np.mean(data2) - np.mean(data1)) / np.mean(data1) * 100):+.2f}%
  Median: {((np.median(data2) - np.median(data1)) / np.median(data1) * 100):+.2f}%
  P95: {((np.percentile(data2, 95) - np.percentile(data1, 95)) / np.percentile(data1, 95) * 100):+.2f}%
  P99: {((np.percentile(data2, 99) - np.percentile(data1, 99)) / np.percentile(data1, 99) * 100):+.2f}%"""
    
    # 在图片右侧添加统计信息
    fig.text(0.02, 0.95, stats_text, transform=fig.transFigure, 
             verticalalignment='top', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)  # 为统计信息留出空间
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"小提琴图已保存到: {output_path}")
    
    plt.show()
    
    return data_combined

def print_statistical_comparison(data1, data2, label1, label2):
    """打印详细的统计对比"""
    print("\n" + "="*60)
    print("详细统计对比")
    print("="*60)
    
    print(f"\n{label1}:")
    print(f"  样本数: {len(data1):,}")
    print(f"  均值: {np.mean(data1):.4f} ms")
    print(f"  中位数: {np.median(data1):.4f} ms")
    print(f"  标准差: {np.std(data1):.4f} ms")
    print(f"  最小值: {np.min(data1):.4f} ms")
    print(f"  最大值: {np.max(data1):.4f} ms")
    print(f"  P50: {np.percentile(data1, 50):.4f} ms")
    print(f"  P90: {np.percentile(data1, 90):.4f} ms")
    print(f"  P95: {np.percentile(data1, 95):.4f} ms")
    print(f"  P99: {np.percentile(data1, 99):.4f} ms")
    
    print(f"\n{label2}:")
    print(f"  样本数: {len(data2):,}")
    print(f"  均值: {np.mean(data2):.4f} ms")
    print(f"  中位数: {np.median(data2):.4f} ms")
    print(f"  标准差: {np.std(data2):.4f} ms")
    print(f"  最小值: {np.min(data2):.4f} ms")
    print(f"  最大值: {np.max(data2):.4f} ms")
    print(f"  P50: {np.percentile(data2, 50):.4f} ms")
    print(f"  P90: {np.percentile(data2, 90):.4f} ms")
    print(f"  P95: {np.percentile(data2, 95):.4f} ms")
    print(f"  P99: {np.percentile(data2, 99):.4f} ms")
    
    # 计算相对差异
    print(f"\n相对差异 ({label2} vs {label1}):")
    mean_diff = (np.mean(data2) - np.mean(data1)) / np.mean(data1) * 100
    median_diff = (np.median(data2) - np.median(data1)) / np.median(data1) * 100
    p95_diff = (np.percentile(data2, 95) - np.percentile(data1, 95)) / np.percentile(data1, 95) * 100
    p99_diff = (np.percentile(data2, 99) - np.percentile(data1, 99)) / np.percentile(data1, 99) * 100
    
    print(f"  均值差异: {mean_diff:+.2f}%")
    print(f"  中位数差异: {median_diff:+.2f}%")
    print(f"  P95差异: {p95_diff:+.2f}%")
    print(f"  P99差异: {p99_diff:+.2f}%")

def main():
    # 文件路径
    file1_path = "/home/vllm/vllm_metrics_toolkit/results/sim_vision/small_chunk_size/qwen_sim_text_benchmark_20250925_192901.json.json"
    file2_path = "/home/vllm/vllm_metrics_toolkit/results/sim_vision/qwen_sim_text_benchmark_20250925_160546.json.json"
    
    # 输出路径
    output_path = "./figures/itl_violin_comparison.png"
    
    print("开始分析ITL数据...")
    print("="*60)
    
    # 加载数据
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)
    
    # 提取ITL数据
    print("\n提取ITL数据...")
    itl_data1 = extract_itl_data(data1)
    itl_data2 = extract_itl_data(data2)
    
    # 创建标签（从文件名提取时间戳）
    label1 = "With chunk-size=128"
    label2 = "With chunk-size=2048"
    
    # 打印统计对比
    print_statistical_comparison(itl_data1, itl_data2, label1, label2)
    
    # 绘制小提琴图对比
    print(f"\n绘制小提琴图对比...")
    plot_violin_comparison(itl_data1, itl_data2, label1, label2, output_path)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()
