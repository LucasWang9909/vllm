#!/usr/bin/env python3
"""
累计Token生成图生成器

根据基准测试结果，绘制累计token生成随时间变化的图表
精确计算每个token的生成时间点
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import argparse
from pathlib import Path


class TokenEvent:
    """单个token生成事件"""
    def __init__(self, absolute_time: float, request_id: str, token_index: int, is_first_token: bool = False):
        self.absolute_time = absolute_time  # 绝对时间戳
        self.request_id = request_id
        self.token_index = token_index  # 在该请求中的token索引
        self.is_first_token = is_first_token
    
    def __lt__(self, other):
        return self.absolute_time < other.absolute_time


class CumulativeTokenAnalyzer:
    """累计Token分析器"""
    
    def __init__(self, benchmark_data: Dict[str, Any]):
        self.data = benchmark_data
        self.token_events: List[TokenEvent] = []
        self.start_time = None
        self.end_time = None
    
    def extract_token_events(self) -> List[TokenEvent]:
        """
        从基准测试数据中提取所有token生成事件
        返回按时间排序的token事件列表
        """
        print("🔍 Extracting token generation events...")
        
        all_events = []
        
        for request in self.data.get('requests', []):
            if not request.get('success', False):
                continue
                
            request_id = request.get('request_id', 'unknown')
            request_start_time = request.get('timestamp', 0)
            
            # 获取客户端测量的TTFT
            client_ttft_ms = request.get('client_metrics', {}).get('ttft_ms')
            if client_ttft_ms is None:
                print(f"⚠️  Skipping request {request_id}: no TTFT data")
                continue
            
            completion_tokens = request.get('tokens', {}).get('completion_tokens', 0)
            if completion_tokens <= 0:
                print(f"⚠️  Skipping request {request_id}: no completion tokens")
                continue
            
            # 计算第一个token的绝对时间
            first_token_time = request_start_time + (client_ttft_ms / 1000.0)
            all_events.append(TokenEvent(
                absolute_time=first_token_time,
                request_id=request_id,
                token_index=0,
                is_first_token=True
            ))
            
            # 获取后续token的间隔时间
            itl_list = request.get('detailed_data', {}).get('itl_list_seconds', [])
            
            current_time = first_token_time
            for i, itl_seconds in enumerate(itl_list):
                current_time += itl_seconds
                all_events.append(TokenEvent(
                    absolute_time=current_time,
                    request_id=request_id,
                    token_index=i + 1,
                    is_first_token=False
                ))
        
        # 按时间排序
        all_events.sort()
        
        print(f"✅ Extracted {len(all_events)} token events from {len(self.data.get('requests', []))} requests")
        
        if all_events:
            self.start_time = all_events[0].absolute_time
            self.end_time = all_events[-1].absolute_time
            duration = self.end_time - self.start_time
            print(f"📊 Time range: {duration:.2f} seconds")
        
        self.token_events = all_events
        return all_events
    
    def generate_cumulative_data(self) -> Tuple[List[float], List[int]]:
        """
        生成累计数据点
        返回 (时间列表, 累计token数列表)
        """
        if not self.token_events:
            self.extract_token_events()
        
        # 转换为相对时间（从开始时间算起）
        relative_times = []
        cumulative_tokens = []
        
        for i, event in enumerate(self.token_events):
            relative_time = event.absolute_time - self.start_time
            relative_times.append(relative_time)
            cumulative_tokens.append(i + 1)  # 累计token数
        
        return relative_times, cumulative_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.token_events:
            self.extract_token_events()
        
        total_tokens = len(self.token_events)
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # 计算吞吐量
        tokens_per_second = total_tokens / duration if duration > 0 else 0
        
        # 统计请求数
        unique_requests = set(event.request_id for event in self.token_events)
        
        # 计算时间窗口内的吞吐量变化
        window_size = max(1, duration / 20)  # 20个时间窗口
        throughput_windows = []
        
        for i in range(20):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            
            tokens_in_window = sum(1 for event in self.token_events 
                                 if window_start <= (event.absolute_time - self.start_time) < window_end)
            
            window_throughput = tokens_in_window / window_size if window_size > 0 else 0
            throughput_windows.append(window_throughput)
        
        return {
            'total_tokens': total_tokens,
            'total_requests': len(unique_requests),
            'duration_seconds': duration,
            'average_tokens_per_second': tokens_per_second,
            'peak_tokens_per_second': max(throughput_windows) if throughput_windows else 0,
            'min_tokens_per_second': min(throughput_windows) if throughput_windows else 0,
            'throughput_windows': throughput_windows
        }


def plot_cumulative_tokens(analyzer: CumulativeTokenAnalyzer, output_file: str = None, show_plot: bool = True):
    """绘制累计token生成图"""
    
    print("📈 Generating cumulative token plot...")
    
    # 获取数据
    times, cumulative = analyzer.generate_cumulative_data()
    stats = analyzer.get_statistics()
    
    if not times:
        print("❌ No data to plot")
        return
    
    # 创建图表
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 主图：累计token数
    ax1.plot(times, cumulative, linewidth=2, color='#1f77b4', alpha=0.8)
    ax1.fill_between(times, cumulative, alpha=0.3, color='#1f77b4')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Cumulative Tokens Generated', fontsize=12)
    ax1.set_title('Cumulative Token Generation Over Time\n'
                  f'Total: {stats["total_tokens"]:,} tokens, '
                  f'Duration: {stats["duration_seconds"]:.1f}s, '
                  f'Avg: {stats["average_tokens_per_second"]:.1f} tokens/s', 
                  fontsize=14, fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(times))
    ax1.set_ylim(0, max(cumulative))
    
    # 添加关键统计信息
    textstr = f'''Statistics:
• Total Requests: {stats["total_requests"]:,}
• Total Tokens: {stats["total_tokens"]:,}
• Duration: {stats["duration_seconds"]:.1f}s
• Avg Throughput: {stats["average_tokens_per_second"]:.1f} tokens/s
• Peak Throughput: {stats["peak_tokens_per_second"]:.1f} tokens/s'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 子图：吞吐量变化
    window_times = np.linspace(0, stats["duration_seconds"], len(stats["throughput_windows"]))
    ax2.plot(window_times, stats["throughput_windows"], 
             linewidth=2, color='#ff7f0e', marker='o', markersize=4)
    ax2.fill_between(window_times, stats["throughput_windows"], 
                     alpha=0.3, color='#ff7f0e')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax2.set_title('Token Generation Throughput Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加平均线
    avg_line = stats["average_tokens_per_second"]
    ax2.axhline(y=avg_line, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_line:.1f} tokens/s')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {output_file}")
    
    # 显示图片
    if show_plot:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate cumulative token generation plot")
    parser.add_argument("input_file", help="Path to benchmark JSON file")
    parser.add_argument("--output", "-o", help="Output image file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # 加载数据
    print(f"📂 Loading benchmark data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        return
    
    # 创建分析器
    analyzer = CumulativeTokenAnalyzer(data)
    
    # 生成图表
    output_file = args.output
    if not output_file:
        # 自动生成输出文件名
        base_name = input_path.stem
        output_file = f"{base_name}_cumulative_tokens.png"
    
    try:
        plot_cumulative_tokens(
            analyzer, 
            output_file=output_file, 
            show_plot=not args.no_show
        )
        
        # 打印详细统计
        stats = analyzer.get_statistics()
        print("\n📊 Detailed Statistics:")
        print(f"   Total Requests: {stats['total_requests']:,}")
        print(f"   Total Tokens: {stats['total_tokens']:,}")
        print(f"   Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"   Average Throughput: {stats['average_tokens_per_second']:.2f} tokens/s")
        print(f"   Peak Throughput: {stats['peak_tokens_per_second']:.2f} tokens/s")
        print(f"   Minimum Throughput: {stats['min_tokens_per_second']:.2f} tokens/s")
        
    except Exception as e:
        print(f"❌ Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
