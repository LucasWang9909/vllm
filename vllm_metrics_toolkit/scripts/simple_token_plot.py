#!/usr/bin/env python3
"""
简化版累计Token生成图
专门针对基准测试数据的快速可视化
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Tuple
import sys


def extract_token_timeline(data):
    """提取token生成时间线"""
    print("🔍 Processing token generation timeline...")
    
    # 收集所有token生成事件
    events = []  # [(absolute_time, cumulative_count)]
    
    for i, request in enumerate(data.get('requests', [])):
        if not request.get('success', False):
            continue
            
        # 请求开始时间戳
        start_time = request.get('timestamp', 0)
        
        # 第一个token时间 (TTFT)
        ttft_ms = request.get('client_metrics', {}).get('ttft_ms')
        if ttft_ms is None:
            continue
            
        completion_tokens = request.get('tokens', {}).get('completion_tokens', 0)
        if completion_tokens <= 0:
            continue
        
        # 第一个token的绝对时间
        first_token_time = start_time + (ttft_ms / 1000.0)
        events.append(first_token_time)
        
        # 后续token时间
        itl_list = request.get('detailed_data', {}).get('itl_list_seconds', [])
        current_time = first_token_time
        
        for itl in itl_list:
            current_time += itl
            events.append(current_time)
    
    # 排序并转换为累计数据
    events.sort()
    
    if not events:
        return [], []
    
    # 转换为相对时间（从第一个token开始）
    start_time = events[0]
    relative_times = [(t - start_time) for t in events]
    cumulative_counts = list(range(1, len(events) + 1))
    
    print(f"✅ Processed {len(events)} token events over {relative_times[-1]:.2f} seconds")
    
    return relative_times, cumulative_counts


def create_plot(times, cumulative, output_file="cumulative_tokens.png"):
    """创建累计token图"""
    print("📊 Creating cumulative token plot...")
    
    plt.figure(figsize=(12, 8))
    
    # 主图
    plt.plot(times, cumulative, linewidth=2, color='#2E86AB', alpha=0.9)
    plt.fill_between(times, cumulative, alpha=0.3, color='#2E86AB')
    
    # 标签和标题
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Cumulative Tokens Generated', fontsize=14)
    plt.title('Cumulative Token Generation Over Time', fontsize=16, fontweight='bold')
    
    # 网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 统计信息
    total_tokens = cumulative[-1] if cumulative else 0
    duration = times[-1] if times else 0
    avg_rate = total_tokens / duration if duration > 0 else 0
    
    # 添加统计文本
    stats_text = f'Total Tokens: {total_tokens:,}\nDuration: {duration:.1f}s\nAvg Rate: {avg_rate:.1f} tokens/s'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to: {output_file}")
    
    # 显示
    plt.show()
    
    return total_tokens, duration, avg_rate


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_token_plot.py <benchmark_json_file>")
        print("Example: python simple_token_plot.py my_test_20250922_165706.json.json")
        return
    
    input_file = sys.argv[1]
    
    # 加载数据
    print(f"📂 Loading data from: {input_file}")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # 提取时间线
    times, cumulative = extract_token_timeline(data)
    
    if not times:
        print("❌ No token data found")
        return
    
    # 创建图表
    # 生成输出文件名
    base_name = input_file.replace('.json.json', '').replace('.json', '')
    output_file = f"{base_name}_cumulative_tokens.png"
    total_tokens, duration, avg_rate = create_plot(times, cumulative, output_file)
    
    # 打印摘要
    print(f"\n📈 Summary:")
    print(f"   Total Tokens: {total_tokens:,}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Average Rate: {avg_rate:.1f} tokens/second")
    print(f"   Peak Rate: {(cumulative[-1] - cumulative[-min(100, len(cumulative))]) / min(100, len(times)) if len(times) > 10 else avg_rate:.1f} tokens/second (approx)")


if __name__ == "__main__":
    main()
