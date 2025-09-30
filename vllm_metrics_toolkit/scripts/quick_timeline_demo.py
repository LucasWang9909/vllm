#!/usr/bin/env python3
"""
快速时间线可视化演示
简单易用的接口来生成请求时间线图表
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from request_timeline_visualizer import RequestTimelineVisualizer
import json
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt


def quick_demo(json_file):
    """快速演示功能"""
    print(f"🚀 Quick Timeline Visualization Demo")
    print(f"📂 Input file: {json_file}")
    
    # 加载数据
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # 创建可视化器
    visualizer = RequestTimelineVisualizer(data)
    
    if not visualizer.timelines:
        print("❌ No timeline data found")
        return
    
    print(f"✅ Found {len(visualizer.timelines)} valid requests")
    
    # 1. 创建概览甘特图（前20个请求）
    print("📊 Creating overview Gantt chart...")
    fig1 = visualizer.create_gantt_chart(max_requests=20, figsize=(14, 10))
    if fig1:
        fig1.savefig('overview_timeline.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("💾 Overview saved: overview_timeline.png")
    
    # 2. 创建详细视图（前5个请求的ITL细节）
    print("📊 Creating detailed ITL view...")
    fig2 = visualizer.create_detailed_request_view(
        request_indices=[0, 1, 2, 3, 4], 
        figsize=(14, 8)
    )
    if fig2:
        fig2.savefig('detailed_itl_view.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("💾 Detailed view saved: detailed_itl_view.png")
    
    # 3. 生成统计报告
    stats = visualizer.generate_statistics_report()
    
    print(f"\n📈 Performance Summary:")
    print(f"   🔄 Total Requests: {stats.get('total_requests', 0)}")
    print(f"   ⚡ Max Concurrent: {stats.get('concurrency_stats', {}).get('max_concurrent', 0)}")
    print(f"   📊 Avg Concurrent: {stats.get('concurrency_stats', {}).get('avg_concurrent', 0):.1f}")
    print(f"   ⏱️  Avg Queue Time: {stats.get('queue_time_stats', {}).get('mean', 0):.1f}ms")
    print(f"   🧠 Avg Prefill Time: {stats.get('prefill_time_stats', {}).get('mean', 0):.1f}ms")
    print(f"   ⏲️  Avg Total Duration: {stats.get('duration_stats', {}).get('mean', 0):.1f}s")
    
    # 4. 显示并发分析洞察
    concurrency = stats.get('concurrency_stats', {})
    max_concurrent = concurrency.get('max_concurrent', 0)
    avg_concurrent = concurrency.get('avg_concurrent', 0)
    
    print(f"\n🎯 Key Insights:")
    if max_concurrent > 200:
        print(f"   🔥 High concurrency detected! Peak: {max_concurrent} requests")
        print(f"   💡 Your async RPS control is working well")
    
    queue_avg = stats.get('queue_time_stats', {}).get('mean', 0)
    if queue_avg > 100:
        print(f"   ⚠️  Significant queue time detected: {queue_avg:.1f}ms average")
        print(f"   💡 Consider optimizing request scheduling or increasing capacity")
    else:
        print(f"   ✅ Low queue times: {queue_avg:.1f}ms average - good performance!")
    
    efficiency = avg_concurrent / max_concurrent if max_concurrent > 0 else 0
    print(f"   📈 Concurrency efficiency: {efficiency:.1%} (avg/max concurrent)")
    
    print(f"\n🎨 Generated Visualizations:")
    print(f"   📊 overview_timeline.png - Shows request lifecycle stages")
    print(f"   🔍 detailed_itl_view.png - Shows individual token latencies")
    print(f"   💡 Use these to understand your system's performance patterns!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_timeline_demo.py <benchmark_json_file>")
        print("Example: python quick_timeline_demo.py my_test_20250922_165706.json.json")
        return
    
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    quick_demo(json_file)


if __name__ == "__main__":
    main()
