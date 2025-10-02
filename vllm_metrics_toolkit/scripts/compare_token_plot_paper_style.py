#!/usr/bin/env python3
"""
论文风格的累计Token生成对比图

模仿论文中的可视化风格，包含inset放大框
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


def plot_paper_style_comparison(analyzer1: CumulativeTokenAnalyzer, 
                                analyzer2: CumulativeTokenAnalyzer,
                                label1: str = "File 1",
                                label2: str = "File 2",
                                output_file: str = None,
                                show_plot: bool = True,
                                main_range: Tuple[float, float] = None,
                                inset_range: Tuple[float, float] = None,
                                stall_threshold: float = 0.06):
    """绘制论文风格的对比图，带有inset放大框
    
    Args:
        main_range: (start_time, end_time) 主图显示的时间范围
        inset_range: (start_time, end_time) inset放大框的时间范围
    """
    
    print("📈 Generating paper-style comparison plot...")

    stall_threshold = max(stall_threshold, 0.0)

    # 获取数据
    times1, cumulative1 = analyzer1.generate_cumulative_data()
    times2, cumulative2 = analyzer2.generate_cumulative_data()

    if not times1 or not times2:
        print("❌ No data to plot")
        return
    
    # 如果指定了主图时间范围，过滤数据
    if main_range is not None:
        main_start, main_end = main_range
        print(f"📍 Main plot range: {main_start}s - {main_end}s")
        
        # 过滤主图数据1
        main_indices1 = [i for i, t in enumerate(times1) if main_start <= t <= main_end]
        if main_indices1:
            times1 = [times1[i] for i in main_indices1]
            cumulative1 = [cumulative1[i] for i in main_indices1]
        
        # 过滤主图数据2
        main_indices2 = [i for i, t in enumerate(times2) if main_start <= t <= main_end]
        if main_indices2:
            times2 = [times2[i] for i in main_indices2]
            cumulative2 = [cumulative2[i] for i in main_indices2]
        
        if not times1 and not times2:
            print("❌ No data in specified main range")
            return
    
    # 设置论文风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 12
    
    # 创建单个图表
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 主图：累计token数对比
    color1 = '#4472C4'  # 蓝色 (类似论文中的 Sarathi-Serve)
    color2 = '#ED7D31'  # 橙色 (类似论文中的 vLLM)

    # 绘制主曲线
    line1, = ax.plot(times1, cumulative1, linewidth=2.5, color=color1, 
                     alpha=0.9, label=label1, linestyle='-')
    line2, = ax.plot(times2, cumulative2, linewidth=2.5, color=color2, 
                     alpha=0.9, label=label2, linestyle='-')

    stall_color = '#C00000'

    def find_stall_indices(times: List[float], threshold: float) -> List[int]:
        if not times or len(times) < 2 or threshold <= 0:
            return []
        return [i for i in range(len(times) - 1)
                if (times[i + 1] - times[i]) >= threshold]

    def highlight_stalls(ax_obj, times: List[float], cumulative: List[int],
                         stall_indices: List[int], add_label: bool) -> bool:
        if not stall_indices:
            return False

        label = "Stall ≥{:.0f} ms".format(stall_threshold * 1000) if add_label else None

        for idx, start_idx in enumerate(stall_indices):
            seg_label = label if idx == 0 else None
            ax_obj.plot(times[start_idx:start_idx + 2],
                        cumulative[start_idx:start_idx + 2],
                        color=stall_color,
                        linewidth=3,
                        alpha=0.85,
                        solid_capstyle='round',
                        zorder=4,
                        label=seg_label)
        return label is not None

    # 设置轴标签和标题
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Tokens Generated', fontsize=13, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置范围
    if main_range is not None:
        ax.set_xlim(main_range[0], main_range[1])
        # Y轴范围根据实际数据
        max_tokens = max(max(cumulative1) if cumulative1 else 0, max(cumulative2) if cumulative2 else 0)
        min_tokens = min(min(cumulative1) if cumulative1 else 0, min(cumulative2) if cumulative2 else 0)
        margin = (max_tokens - min_tokens) * 0.05
        ax.set_ylim(min_tokens - margin, max_tokens + margin)
    else:
        max_time = max(max(times1) if times1 else 0, max(times2) if times2 else 0)
        max_tokens = max(max(cumulative1) if cumulative1 else 0, max(cumulative2) if cumulative2 else 0)
        ax.set_xlim(0, max_time)
        ax.set_ylim(0, max_tokens * 1.05)
    
    # 标记停滞片段（token生成间隔超过阈值）
    stall_indices1 = find_stall_indices(times1, stall_threshold)
    stall_indices2 = find_stall_indices(times2, stall_threshold)

    total_stalls = len(stall_indices1) + len(stall_indices2)
    if total_stalls:
        print(f"🔴 Highlighting {total_stalls} stall segments (≥{stall_threshold * 1000:.0f} ms)")

    label_added = highlight_stalls(ax, times1, cumulative1, stall_indices1, add_label=True)
    highlight_stalls(ax, times2, cumulative2, stall_indices2, add_label=not label_added)

    # 格式化y轴为 K 格式
    def format_func(value, tick_number):
        if value >= 1000:
            return f'{int(value/1000)}K'
        return f'{int(value)}'
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    # 添加图例
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             edgecolor='black', framealpha=0.9)
    
    # 添加 inset（放大框）
    if inset_range is not None:
        start_time, end_time = inset_range
        
        # 创建inset axes - 位置 [left, bottom, width, height] in figure coordinates
        # 放在左上角，更小尺寸
        axins = ax.inset_axes([0.15, 0.58, 0.32, 0.28])
        
        # 过滤inset范围内的数据
        inset_indices1 = [i for i, t in enumerate(times1) if start_time <= t <= end_time]
        inset_times1 = [times1[i] for i in inset_indices1]
        inset_cumulative1 = [cumulative1[i] for i in inset_indices1]
        
        inset_indices2 = [i for i, t in enumerate(times2) if start_time <= t <= end_time]
        inset_times2 = [times2[i] for i in inset_indices2]
        inset_cumulative2 = [cumulative2[i] for i in inset_indices2]

        # 在inset中绘制
        axins.plot(inset_times1, inset_cumulative1, linewidth=2, 
                  color=color1, alpha=0.9, linestyle='-')
        axins.plot(inset_times2, inset_cumulative2, linewidth=2, 
                  color=color2, alpha=0.9, linestyle='-')

        # inset中标记停滞片段
        inset_stall1 = find_stall_indices(inset_times1, stall_threshold)
        inset_stall2 = find_stall_indices(inset_times2, stall_threshold)
        highlight_stalls(axins, inset_times1, inset_cumulative1, inset_stall1, add_label=False)
        highlight_stalls(axins, inset_times2, inset_cumulative2, inset_stall2, add_label=False)
        
        # 设置inset范围
        axins.set_xlim(start_time, end_time)
        if inset_cumulative1 and inset_cumulative2:
            inset_min = min(min(inset_cumulative1), min(inset_cumulative2))
            inset_max = max(max(inset_cumulative1), max(inset_cumulative2))
            margin = (inset_max - inset_min) * 0.1
            axins.set_ylim(inset_min - margin, inset_max + margin)
        
        # inset网格和样式
        axins.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axins.tick_params(labelsize=9)
        
        # 添加"Generation stall"标注（如果需要检测到stall）
        # 检测是否有明显的平台期
        if inset_cumulative2 and len(inset_cumulative2) > 2:
            # 简单检测：如果有连续的值变化很小
            diffs = [inset_cumulative2[i+1] - inset_cumulative2[i] 
                    for i in range(len(inset_cumulative2)-1)]
            avg_diff = np.mean(diffs) if diffs else 0
            
            for i, diff in enumerate(diffs):
                if diff < avg_diff * 0.3 and avg_diff > 0:  # 检测到明显减速
                    stall_time = inset_times2[i]
                    stall_tokens = inset_cumulative2[i]
                    
                    # 添加箭头标注
                    axins.annotate('Generation stall', 
                                  xy=(stall_time, stall_tokens),
                                  xytext=(stall_time + (end_time-start_time)*0.15, 
                                         stall_tokens - (inset_max-inset_min)*0.15),
                                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                                  fontsize=9, ha='left')
                    break
        
        # 在主图上标记inset区域
        from matplotlib.patches import Rectangle
        # 绘制矩形框 - 使用实际的y轴范围
        ylim = ax.get_ylim()
        rect = Rectangle((start_time, ylim[0]), end_time - start_time, ylim[1] - ylim[0],
                        linewidth=1.5, edgecolor='gray', facecolor='none',
                        linestyle='--', alpha=0.6)
        ax.add_patch(rect)
        
        # 连接线（从inset到主图的矩形区域）
        from matplotlib.patches import ConnectionPatch
        ylim = ax.get_ylim()
        # 左上角连接
        con1 = ConnectionPatch(xyA=(start_time, axins.get_ylim()[1]), coordsA=axins.transData,
                              xyB=(start_time, ylim[1]), coordsB=ax.transData,
                              color='gray', linestyle='--', alpha=0.5, linewidth=1)
        fig.add_artist(con1)
        
        # 右上角连接
        con2 = ConnectionPatch(xyA=(end_time, axins.get_ylim()[1]), coordsA=axins.transData,
                              xyB=(end_time, ylim[1]), coordsB=ax.transData,
                              color='gray', linestyle='--', alpha=0.5, linewidth=1)
        fig.add_artist(con2)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Paper-style plot saved to: {output_file}")
    
    # 显示图片
    if show_plot:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-style cumulative token comparison with inset zoom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with inset zoom
  python compare_token_plot_paper_style.py file1.json file2.json \\
    --inset-start 200 --inset-end 220
  
  # With custom labels
  python compare_token_plot_paper_style.py file1.json file2.json \\
    --label1 "No Encoding" --label2 "With Encoding" \\
    --inset-start 200 --inset-end 220
        """
    )
    
    parser.add_argument("file1", help="Path to first benchmark JSON file")
    parser.add_argument("file2", help="Path to second benchmark JSON file")
    parser.add_argument("--label1", default="No Encoding", help="Label for first file")
    parser.add_argument("--label2", default="With Encoding", help="Label for second file")
    parser.add_argument("--output", "-o", help="Output image file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--main-start", type=float, default=None, 
                       help="Main plot start time in seconds")
    parser.add_argument("--main-end", type=float, default=None, 
                       help="Main plot end time in seconds")
    parser.add_argument("--inset-start", type=float, default=None, 
                       help="Inset zoom start time in seconds")
    parser.add_argument("--inset-end", type=float, default=None, 
                       help="Inset zoom end time in seconds")
    parser.add_argument("--stall-threshold-ms", type=float, default=100.0,
                        help="Minimum gap (in milliseconds) between tokens to highlight as a stall")
    
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
        output_file = "token_comparison_paper_style.png"
    
    # 准备主图范围
    main_range = None
    if args.main_start is not None and args.main_end is not None:
        main_range = (args.main_start, args.main_end)
        print(f"\n📍 Main plot range: {args.main_start}s - {args.main_end}s")
    
    # 准备inset范围
    inset_range = None
    if args.inset_start is not None and args.inset_end is not None:
        inset_range = (args.inset_start, args.inset_end)
        print(f"📍 Inset zoom range: {args.inset_start}s - {args.inset_end}s")
    
    # 生成对比图
    try:
        plot_paper_style_comparison(
            analyzer1, analyzer2,
            label1=args.label1,
            label2=args.label2,
            output_file=output_file,
            show_plot=not args.no_show,
            main_range=main_range,
            inset_range=inset_range,
            stall_threshold=max(args.stall_threshold_ms, 0.0) / 1000.0
        )
        
        print("\n✅ Paper-style comparison plot generated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
