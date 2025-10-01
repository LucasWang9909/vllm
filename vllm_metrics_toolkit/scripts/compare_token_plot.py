#!/usr/bin/env python3
"""
累计Token生成对比图生成器

比较两个基准测试结果，在同一张图中绘制累计token生成曲线
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any
import sys

# 导入现有的分析器
sys.path.insert(0, str(Path(__file__).parent))
from cumulative_token_plot import CumulativeTokenAnalyzer


def plot_comparison(analyzer1: CumulativeTokenAnalyzer, 
                   analyzer2: CumulativeTokenAnalyzer,
                   label1: str = "File 1",
                   label2: str = "File 2",
                   output_file: str = None,
                   show_plot: bool = True,
                   time_range: Tuple[float, float] = None):
    """绘制两个测试的对比图
    
    Args:
        time_range: (start_time, end_time) 时间范围，单位为秒。如果为None则显示全部时间
    """
    
    print("📈 Generating comparison plot...")
    
    # 获取数据
    times1, cumulative1 = analyzer1.generate_cumulative_data()
    times2, cumulative2 = analyzer2.generate_cumulative_data()
    stats1 = analyzer1.get_statistics()
    stats2 = analyzer2.get_statistics()
    
    if not times1 or not times2:
        print("❌ No data to plot")
        return
    
    # 如果指定了时间范围，过滤数据
    if time_range is not None:
        start_time, end_time = time_range
        print(f"📍 Zooming to time range: {start_time}s - {end_time}s")
        
        # 过滤数据1
        filtered_indices1 = [i for i, t in enumerate(times1) if start_time <= t <= end_time]
        if filtered_indices1:
            times1 = [times1[i] for i in filtered_indices1]
            cumulative1 = [cumulative1[i] for i in filtered_indices1]
        
        # 过滤数据2
        filtered_indices2 = [i for i, t in enumerate(times2) if start_time <= t <= end_time]
        if filtered_indices2:
            times2 = [times2[i] for i in filtered_indices2]
            cumulative2 = [cumulative2[i] for i in filtered_indices2]
        
        if not times1 and not times2:
            print("❌ No data in specified time range")
            return
    
    # 创建图表
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 主图：累计token数对比
    color1 = '#1f77b4'  # 蓝色
    color2 = '#ff7f0e'  # 橙色
    
    ax1.plot(times1, cumulative1, linewidth=2.5, color=color1, alpha=0.8, label=label1)
    ax1.fill_between(times1, cumulative1, alpha=0.2, color=color1)
    
    ax1.plot(times2, cumulative2, linewidth=2.5, color=color2, alpha=0.8, label=label2)
    ax1.fill_between(times2, cumulative2, alpha=0.2, color=color2)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Cumulative Tokens Generated', fontsize=12)
    
    # 标题根据是否有时间范围而变化
    if time_range is not None:
        title = f'Cumulative Token Generation Comparison\n(Time Range: {time_range[0]}s - {time_range[1]}s)'
    else:
        title = 'Cumulative Token Generation Comparison'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')
    
    # 设置x轴范围
    if time_range is not None:
        ax1.set_xlim(time_range[0], time_range[1])
    else:
        max_time = max(max(times1) if times1 else 0, max(times2) if times2 else 0)
        ax1.set_xlim(0, max_time)
    
    # 添加统计信息对比
    textstr = f'''{label1}:
• Requests: {stats1["total_requests"]:,}
• Tokens: {stats1["total_tokens"]:,}
• Duration: {stats1["duration_seconds"]:.1f}s
• Avg: {stats1["average_tokens_per_second"]:.1f} tok/s
• Peak: {stats1["peak_tokens_per_second"]:.1f} tok/s

{label2}:
• Requests: {stats2["total_requests"]:,}
• Tokens: {stats2["total_tokens"]:,}
• Duration: {stats2["duration_seconds"]:.1f}s
• Avg: {stats2["average_tokens_per_second"]:.1f} tok/s
• Peak: {stats2["peak_tokens_per_second"]:.1f} tok/s'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, family='monospace')
    
    # 子图：吞吐量对比
    window_times1 = np.linspace(0, stats1["duration_seconds"], len(stats1["throughput_windows"]))
    window_times2 = np.linspace(0, stats2["duration_seconds"], len(stats2["throughput_windows"]))
    
    # 如果指定了时间范围，也过滤吞吐量数据
    if time_range is not None:
        start_time, end_time = time_range
        filtered_wt1 = [(t, v) for t, v in zip(window_times1, stats1["throughput_windows"]) if start_time <= t <= end_time]
        filtered_wt2 = [(t, v) for t, v in zip(window_times2, stats2["throughput_windows"]) if start_time <= t <= end_time]
        
        if filtered_wt1:
            window_times1, throughput1 = zip(*filtered_wt1)
        else:
            window_times1, throughput1 = [], []
            
        if filtered_wt2:
            window_times2, throughput2 = zip(*filtered_wt2)
        else:
            window_times2, throughput2 = [], []
    else:
        throughput1 = stats1["throughput_windows"]
        throughput2 = stats2["throughput_windows"]
    
    if window_times1 and throughput1:
        ax2.plot(window_times1, throughput1, 
                 linewidth=2, color=color1, marker='o', markersize=4, alpha=0.8,
                 label=f'{label1} (avg: {stats1["average_tokens_per_second"]:.1f} tok/s)')
        ax2.fill_between(window_times1, throughput1, 
                         alpha=0.2, color=color1)
    
    if window_times2 and throughput2:
        ax2.plot(window_times2, throughput2, 
                 linewidth=2, color=color2, marker='s', markersize=4, alpha=0.8,
                 label=f'{label2} (avg: {stats2["average_tokens_per_second"]:.1f} tok/s)')
        ax2.fill_between(window_times2, throughput2, 
                         alpha=0.2, color=color2)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax2.set_title('Token Generation Throughput Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 设置x轴范围
    if time_range is not None:
        ax2.set_xlim(time_range[0], time_range[1])
    
    # 添加平均线
    if window_times1 and throughput1:
        ax2.axhline(y=stats1["average_tokens_per_second"], color=color1, 
                    linestyle='--', alpha=0.5, linewidth=1)
    if window_times2 and throughput2:
        ax2.axhline(y=stats2["average_tokens_per_second"], color=color2, 
                    linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison plot saved to: {output_file}")
    
    # 显示图片
    if show_plot:
        plt.show()
    
    return fig


def print_comparison_stats(stats1: Dict[str, Any], stats2: Dict[str, Any], 
                          label1: str, label2: str):
    """打印详细的对比统计信息"""
    
    print("\n" + "="*70)
    print("📊 DETAILED COMPARISON")
    print("="*70)
    
    print(f"\n{label1}:")
    print(f"  Total Requests:      {stats1['total_requests']:,}")
    print(f"  Total Tokens:        {stats1['total_tokens']:,}")
    print(f"  Duration:            {stats1['duration_seconds']:.2f} seconds")
    print(f"  Average Throughput:  {stats1['average_tokens_per_second']:.2f} tokens/s")
    print(f"  Peak Throughput:     {stats1['peak_tokens_per_second']:.2f} tokens/s")
    print(f"  Minimum Throughput:  {stats1['min_tokens_per_second']:.2f} tokens/s")
    
    print(f"\n{label2}:")
    print(f"  Total Requests:      {stats2['total_requests']:,}")
    print(f"  Total Tokens:        {stats2['total_tokens']:,}")
    print(f"  Duration:            {stats2['duration_seconds']:.2f} seconds")
    print(f"  Average Throughput:  {stats2['average_tokens_per_second']:.2f} tokens/s")
    print(f"  Peak Throughput:     {stats2['peak_tokens_per_second']:.2f} tokens/s")
    print(f"  Minimum Throughput:  {stats2['min_tokens_per_second']:.2f} tokens/s")
    
    print(f"\n📈 PERFORMANCE DIFFERENCE:")
    
    # 计算差异
    tokens_diff = stats2['total_tokens'] - stats1['total_tokens']
    tokens_diff_pct = (tokens_diff / stats1['total_tokens'] * 100) if stats1['total_tokens'] > 0 else 0
    
    throughput_diff = stats2['average_tokens_per_second'] - stats1['average_tokens_per_second']
    throughput_diff_pct = (throughput_diff / stats1['average_tokens_per_second'] * 100) if stats1['average_tokens_per_second'] > 0 else 0
    
    duration_diff = stats2['duration_seconds'] - stats1['duration_seconds']
    duration_diff_pct = (duration_diff / stats1['duration_seconds'] * 100) if stats1['duration_seconds'] > 0 else 0
    
    print(f"  Total Tokens:        {tokens_diff:+,} ({tokens_diff_pct:+.1f}%)")
    print(f"  Avg Throughput:      {throughput_diff:+.2f} tokens/s ({throughput_diff_pct:+.1f}%)")
    print(f"  Duration:            {duration_diff:+.2f}s ({duration_diff_pct:+.1f}%)")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare cumulative token generation between two benchmark files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two benchmark files with auto-generated labels
  python compare_token_plot.py file1.json file2.json
  
  # Compare with custom labels
  python compare_token_plot.py file1.json file2.json --label1 "Text Only" --label2 "Vision Model"
  
  # Save to specific file without showing
  python compare_token_plot.py file1.json file2.json -o comparison.png --no-show
        """
    )
    
    parser.add_argument("file1", help="Path to first benchmark JSON file")
    parser.add_argument("file2", help="Path to second benchmark JSON file")
    parser.add_argument("--label1", default=None, help="Label for first file (default: filename)")
    parser.add_argument("--label2", default=None, help="Label for second file (default: filename)")
    parser.add_argument("--output", "-o", help="Output image file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--time-start", type=float, default=None, help="Start time in seconds (for zooming)")
    parser.add_argument("--time-end", type=float, default=None, help="End time in seconds (for zooming)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"❌ File not found: {file1_path}")
        return
    
    if not file2_path.exists():
        print(f"❌ File not found: {file2_path}")
        return
    
    # 生成标签
    label1 = args.label1 or file1_path.stem
    label2 = args.label2 or file2_path.stem
    
    # 加载数据
    print(f"📂 Loading first file: {file1_path}")
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {file1_path}: {e}")
        return
    
    print(f"📂 Loading second file: {file2_path}")
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {file2_path}: {e}")
        return
    
    # 创建分析器
    print("\n" + "="*70)
    print("Analyzing File 1...")
    print("="*70)
    analyzer1 = CumulativeTokenAnalyzer(data1)
    analyzer1.extract_token_events()
    
    print("\n" + "="*70)
    print("Analyzing File 2...")
    print("="*70)
    analyzer2 = CumulativeTokenAnalyzer(data2)
    analyzer2.extract_token_events()
    
    # 生成输出文件名
    output_file = args.output
    if not output_file:
        if args.time_start is not None and args.time_end is not None:
            output_file = f"token_comparison_{int(args.time_start)}-{int(args.time_end)}s.png"
        else:
            output_file = "token_comparison.png"
    
    # 准备时间范围
    time_range = None
    if args.time_start is not None and args.time_end is not None:
        time_range = (args.time_start, args.time_end)
    
    # 生成对比图
    try:
        plot_comparison(
            analyzer1, analyzer2,
            label1=label1,
            label2=label2,
            output_file=output_file,
            show_plot=not args.no_show,
            time_range=time_range
        )
        
        # 打印对比统计
        stats1 = analyzer1.get_statistics()
        stats2 = analyzer2.get_statistics()
        print_comparison_stats(stats1, stats2, label1, label2)
        
    except Exception as e:
        print(f"❌ Failed to generate comparison plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
