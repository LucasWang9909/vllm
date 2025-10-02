#!/usr/bin/env python3
"""
请求时间线可视化工具
创建甘特图样式的可视化，显示每个请求的详细处理阶段:
- 队列时间 (gen_ai.latency.time_in_queue)
- 预填充时间 (gen_ai.latency.time_in_model_prefill) 
- 每个token的ITL (Inter-token Latency)
- 当前时间点的并发请求数分析
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import argparse
from pathlib import Path


class RequestTimelineData:
    """单个请求的时间线数据"""
    def __init__(self, request_data: Dict[str, Any]):
        self.request_id = request_data.get('request_id', 'unknown')
        self.start_time = request_data.get('timestamp', 0)
        self.success = request_data.get('success', False)
        
        # 服务端指标
        server_metrics = request_data.get('server_metrics', {})
        self.queue_time_ms = server_metrics.get('queue_time_ms', 0)
        self.prefill_time_ms = server_metrics.get('prefill_time_ms', 0)
        self.decode_time_ms = server_metrics.get('decode_time_ms', 0)
        
        
        # 客户端指标
        client_metrics = request_data.get('client_metrics', {})
        self.client_ttft_ms = client_metrics.get('ttft_ms', 0)
        
        # Token信息
        tokens = request_data.get('tokens', {})
        self.prompt_tokens = tokens.get('prompt_tokens', 0)
        self.completion_tokens = tokens.get('completion_tokens', 0)
        
        # ITL详细数据
        detailed_data = request_data.get('detailed_data', {})
        self.itl_list_ms = detailed_data.get('itl_list_ms', [])
        
        # 计算关键时间点
        self._calculate_timeline()
    
    def _calculate_timeline(self):
        """计算请求的各个时间点"""
        # 所有时间都转换为相对于start_time的秒数
        self.queue_start = 0
        self.queue_end = self.queue_time_ms / 1000.0
        self.prefill_start = self.queue_end

        self.prefill_end = self.prefill_start + (self.prefill_time_ms / 1000.0)
        
        # Token生成时间点
        self.token_times = []
        current_time = self.prefill_end
        
        for i, itl_ms in enumerate(self.itl_list_ms):
            current_time += itl_ms / 1000.0
            self.token_times.append(current_time)
        
        # 总结束时间
        self.end_time = self.token_times[-1] if self.token_times else self.prefill_end
        self.total_duration = self.end_time - self.queue_start


class ConcurrentRequestsAnalyzer:
    """并发请求数分析器"""
    
    def __init__(self, timeline_data: List[RequestTimelineData]):
        self.timelines = timeline_data
        self.global_start_time = min(t.start_time for t in timeline_data) if timeline_data else 0
        
    def calculate_concurrent_requests(self, time_resolution: float = 0.1) -> Tuple[List[float], List[int]]:
        """
        计算每个时间点的并发请求数
        
        Args:
            time_resolution: 时间分辨率（秒）
        
        Returns:
            (时间点列表, 并发数列表)
        """
        if not self.timelines:
            return [], []
        
        # 计算全局时间范围
        max_end_time = max(t.start_time + t.total_duration for t in self.timelines)
        duration = max_end_time - self.global_start_time
        
        # 生成时间点
        time_points = np.arange(0, duration + time_resolution, time_resolution)
        concurrent_counts = []
        
        for t in time_points:
            absolute_time = self.global_start_time + t
            count = 0
            
            for timeline in self.timelines:
                request_start = timeline.start_time
                request_end = timeline.start_time + timeline.total_duration
                
                if request_start <= absolute_time <= request_end:
                    count += 1
            
            concurrent_counts.append(count)
        
        return time_points.tolist(), concurrent_counts


class RequestTimelineVisualizer:
    """请求时间线可视化器"""
    
    def __init__(self, benchmark_data: Dict[str, Any]):
        self.data = benchmark_data
        self.timelines = []
        self._load_timelines()
        self.itl_stats = self._calculate_itl_statistics()
    
    def _load_timelines(self):
        """加载所有请求的时间线数据"""
        print("📊 Loading request timelines...")
        
        for request in self.data.get('requests', []):
            if request.get('success', False):
                timeline = RequestTimelineData(request)
                if timeline.total_duration > 0:  # 确保有有效的时间数据
                    self.timelines.append(timeline)
        
        # 按开始时间排序
        self.timelines.sort(key=lambda x: x.start_time)
        
        print(f"✅ Loaded {len(self.timelines)} valid request timelines")
    
    def _calculate_itl_statistics(self) -> Dict[str, float]:
        """计算所有ITL的统计信息"""
        all_itls = []
        for timeline in self.timelines:
            # 将ITL从毫秒转换为秒
            itls_seconds = [itl_ms / 1000.0 for itl_ms in timeline.itl_list_ms]
            all_itls.extend(itls_seconds)
        
        if not all_itls:
            return {'mean': 0, 'std': 0}
        
        mean_itl = np.mean(all_itls)
        std_itl = np.std(all_itls)
        
        print(f"📊 ITL Statistics: Mean={mean_itl*1000:.1f}ms, Std={std_itl*1000:.1f}ms")
        
        return {
            'mean': mean_itl,
            'std': std_itl,
            'threshold_1': mean_itl,
            'threshold_2': mean_itl + std_itl,
            'threshold_3': mean_itl + 2 * std_itl
        }
    
    def _get_itl_color(self, itl_duration_seconds: float) -> str:
        """根据ITL长度返回对应的颜色"""
        if itl_duration_seconds <= self.itl_stats['threshold_1']:
            return '#2ECC71'  # 绿色 - 正常ITL
        elif itl_duration_seconds <= self.itl_stats['threshold_2']:
            return '#F39C12'  # 黄色 - 稍长ITL
        elif itl_duration_seconds <= self.itl_stats['threshold_3']:
            return '#E67E22'  # 橙色 - 长ITL
        else:
            return '#E74C3C'  # 红色 - 异常长ITL
    
    def create_gantt_chart(self, max_requests: int = 50, figsize: Tuple[int, int] = (25, 12)):
        """
        创建甘特图显示请求时间线
        
        Args:
            max_requests: 最大显示请求数（避免图表过于复杂）
            figsize: 图表大小
        """
        if not self.timelines:
            print("❌ No timeline data available")
            return None
        
        # 限制显示的请求数量
        display_timelines = self.timelines[:max_requests]
        
        # 计算相对时间基准
        global_start = min(t.start_time for t in display_timelines)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 颜色配置
        colors = {
            'queue': '#FF6B6B',      # 红色 - 队列等待
            'prefill': '#4ECDC4',    # 青色 - 预填充
            'token': '#45B7D1',      # 蓝色 - Token生成
            'itl': '#96CEB4'         # 绿色 - ITL间隔
        }
        
        # 绘制甘特图
        for i, timeline in enumerate(display_timelines):
            y_pos = i
            start_offset = timeline.start_time - global_start
            
            # 队列时间
            if timeline.queue_time_ms > 0:
                queue_rect = patches.Rectangle(
                    (start_offset + timeline.queue_start, y_pos - 0.4),
                    timeline.queue_end - timeline.queue_start, 0.8,
                    facecolor=colors['queue'], alpha=0.8, label='Queue' if i == 0 else ""
                )
                ax1.add_patch(queue_rect)
            
            # 预填充时间
            if timeline.prefill_time_ms > 0:
                prefill_rect = patches.Rectangle(
                    (start_offset + timeline.prefill_start, y_pos - 0.4),
                    timeline.prefill_end - timeline.prefill_start, 0.8,
                    facecolor=colors['prefill'], alpha=0.8, label='Prefill' if i == 0 else ""
                )
                ax1.add_patch(prefill_rect)
            
            # Token生成（每个ITL作为单独的段，根据长度用不同颜色）
            prev_time = timeline.prefill_end
            for token_time in timeline.token_times:
                itl_duration = token_time - prev_time
                itl_color = self._get_itl_color(itl_duration)
                
                token_rect = patches.Rectangle(
                    (start_offset + prev_time, y_pos - 0.3),
                    itl_duration, 0.6,
                    facecolor=itl_color,
                    edgecolor=(0, 0, 0, 0.5),
                    linewidth=0.2,
                    alpha=0.85
                )
                ax1.add_patch(token_rect)
                prev_time = token_time
            # 最后一个 token 段已通过边框体现分隔，无需额外线条
        
        # 设置甘特图样式
        ax1.set_xlim(0, max(t.start_time - global_start + t.total_duration for t in display_timelines))
        ax1.set_ylim(-0.5, len(display_timelines) - 0.5)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Request Index', fontsize=12)
        ax1.set_title(f'Request Timeline Visualization (First {len(display_timelines)} Requests)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加ITL颜色分级图例
        itl_legend_elements = [
            patches.Patch(color='#FF6B6B', label='Queue'),
            patches.Patch(color='#4ECDC4', label='Encode+Prefill'),
            patches.Patch(color='#2ECC71', label=f'Normal ITL (<{self.itl_stats["threshold_1"]*1000:.0f}ms)'),
            patches.Patch(color='#F39C12', label=f'Long ITL ({self.itl_stats["threshold_1"]*1000:.0f}-{self.itl_stats["threshold_2"]*1000:.0f}ms)'),
            patches.Patch(color='#E67E22', label=f'Longer ITL ({self.itl_stats["threshold_2"]*1000:.0f}-{self.itl_stats["threshold_3"]*1000:.0f}ms)'),
            patches.Patch(color='#E74C3C', label=f'Abnormal ITL (>{self.itl_stats["threshold_3"]*1000:.0f}ms)')
        ]
        ax1.legend(handles=itl_legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 反转Y轴（最新请求在上方）
        ax1.invert_yaxis()
        
        # 添加并发请求数分析
        analyzer = ConcurrentRequestsAnalyzer(self.timelines)
        time_points, concurrent_counts = analyzer.calculate_concurrent_requests()
        
        ax2.plot(time_points, concurrent_counts, linewidth=2, color='#E74C3C')
        ax2.fill_between(time_points, concurrent_counts, alpha=0.3, color='#E74C3C')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Concurrent Requests', fontsize=12)
        ax2.set_title('Number of Concurrent Requests Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        max_concurrent = max(concurrent_counts) if concurrent_counts else 0
        avg_concurrent = np.mean(concurrent_counts) if concurrent_counts else 0
        
        stats_text = f'Max Concurrent: {max_concurrent}\nAvg Concurrent: {avg_concurrent:.1f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_detailed_request_view(self, request_indices: List[int] = None, figsize: Tuple[int, int] = (15, 8)):
        """
        创建详细的单个或多个请求视图，显示每个ITL
        
        Args:
            request_indices: 要显示的请求索引列表，None表示前几个
            figsize: 图表大小
        """
        if not self.timelines:
            print("❌ No timeline data available")
            return None
        
        if request_indices is None:
            request_indices = list(range(min(10, len(self.timelines))))
        
        selected_timelines = [self.timelines[i] for i in request_indices if i < len(self.timelines)]
        
        if not selected_timelines:
            print("❌ No valid request indices")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        global_start = min(t.start_time for t in selected_timelines)
        
        colors = {
            'queue': '#FF6B6B',
            'prefill': '#4ECDC4', 
            'encoder': '#9B59B6',
            'token': '#45B7D1'
        }
        
        for i, timeline in enumerate(selected_timelines):
            y_pos = i * 2  # 增加间距以显示更多细节
            start_offset = timeline.start_time - global_start
            
            # 队列时间
            if timeline.queue_time_ms > 0:
                ax.barh(y_pos, timeline.queue_end - timeline.queue_start, 
                       left=start_offset + timeline.queue_start, height=0.6,
                       color=colors['queue'], alpha=0.8, label='Queue' if i == 0 else "")
            
            # 预填充时间
            if timeline.prefill_time_ms > 0:
                if timeline.encoder_start is not None and timeline.encoder_end is not None:
                    ax.barh(y_pos, timeline.encoder_end - timeline.encoder_start,
                           left=start_offset + timeline.encoder_start, height=0.6,
                           color=colors['encoder'], alpha=0.8, label='Encoder' if i == 0 else "")
                    if timeline.true_prefill_start is not None and timeline.true_prefill_end is not None:
                        ax.barh(y_pos, timeline.true_prefill_end - timeline.true_prefill_start,
                               left=start_offset + timeline.true_prefill_start, height=0.6,
                               color=colors['prefill'], alpha=0.8, label='Prefill' if i == 0 else "")
                else:
                    ax.barh(y_pos, timeline.prefill_end - timeline.prefill_start,
                           left=start_offset + timeline.prefill_start, height=0.6,
                           color=colors['prefill'], alpha=0.8, label='Prefill' if i == 0 else "")
            
            # 每个Token的ITL（详细显示，根据长度用不同颜色）
            prev_time = timeline.prefill_end
            for token_time in timeline.token_times:
                itl_duration = token_time - prev_time
                itl_color = self._get_itl_color(itl_duration)
                
                ax.barh(y_pos, itl_duration, left=start_offset + prev_time, height=0.4,
                       color=itl_color, edgecolor=(0, 0, 0, 0.5), linewidth=0.2, alpha=0.85)
                
                # 在每个token段上标注ITL时间（智能阈值）
                # 如果ITL超过平均值，就显示时间标注
                if itl_duration > self.itl_stats['threshold_1']:
                    ax.text(start_offset + prev_time + itl_duration/2, y_pos, 
                           f'{itl_duration*1000:.0f}ms', 
                           ha='center', va='center', fontsize=8, 
                           color='white', weight='bold')
                
                prev_time = token_time
            # 最后一个 token 段已通过边框体现分隔，无需额外线条
            
            # 添加请求ID标签
            ax.text(-0.5, y_pos, f'Req {request_indices[i]}\n{timeline.request_id[:8]}...', 
                   ha='right', va='center', fontsize=9)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Requests', fontsize=12)
        ax.set_title('Detailed Request Timeline with Individual Token Latencies', 
                    fontsize=14, fontweight='bold')
        
        # 添加ITL颜色分级图例
        itl_legend_elements = [
            patches.Patch(color='#FF6B6B', label='Queue'),
            patches.Patch(color='#4ECDC4', label='Encode+Prefill'),
            patches.Patch(color='#2ECC71', label=f'Normal ITL (<{self.itl_stats["threshold_1"]*1000:.0f}ms)'),
            patches.Patch(color='#F39C12', label=f'Long ITL ({self.itl_stats["threshold_1"]*1000:.0f}-{self.itl_stats["threshold_2"]*1000:.0f}ms)'),
            patches.Patch(color='#E67E22', label=f'Longer ITL ({self.itl_stats["threshold_2"]*1000:.0f}-{self.itl_stats["threshold_3"]*1000:.0f}ms)'),
            patches.Patch(color='#E74C3C', label=f'Abnormal ITL (>{self.itl_stats["threshold_3"]*1000:.0f}ms)')
        ]
        ax.legend(handles=itl_legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax.grid(True, alpha=0.3)
        
        # 设置Y轴标签
        ax.set_yticks([i * 2 for i in range(len(selected_timelines))])
        ax.set_yticklabels([f'Request {idx}' for idx in request_indices[:len(selected_timelines)]])
        
        plt.tight_layout()
        return fig
    
    def generate_statistics_report(self) -> Dict[str, Any]:
        """生成统计报告"""
        if not self.timelines:
            return {}
        
        # 基本统计
        total_requests = len(self.timelines)
        
        # 时间统计
        queue_times = [t.queue_time_ms for t in self.timelines if t.queue_time_ms > 0]
        prefill_times = [t.prefill_time_ms for t in self.timelines if t.prefill_time_ms > 0]
        total_durations = [t.total_duration for t in self.timelines]
        
        # 并发分析
        analyzer = ConcurrentRequestsAnalyzer(self.timelines)
        _, concurrent_counts = analyzer.calculate_concurrent_requests()
        
        # Token统计
        completion_tokens = [t.completion_tokens for t in self.timelines]
        itl_counts = [len(t.itl_list_ms) for t in self.timelines]
        
        return {
            'total_requests': total_requests,
            'queue_time_stats': {
                'mean': np.mean(queue_times) if queue_times else 0,
                'median': np.median(queue_times) if queue_times else 0,
                'max': max(queue_times) if queue_times else 0,
                'min': min(queue_times) if queue_times else 0
            },
            'prefill_time_stats': {
                'mean': np.mean(prefill_times) if prefill_times else 0,
                'median': np.median(prefill_times) if prefill_times else 0,
                'max': max(prefill_times) if prefill_times else 0,
                'min': min(prefill_times) if prefill_times else 0
            },
            'duration_stats': {
                'mean': np.mean(total_durations),
                'median': np.median(total_durations),
                'max': max(total_durations),
                'min': min(total_durations)
            },
            'concurrency_stats': {
                'max_concurrent': max(concurrent_counts) if concurrent_counts else 0,
                'avg_concurrent': np.mean(concurrent_counts) if concurrent_counts else 0,
                'min_concurrent': min(concurrent_counts) if concurrent_counts else 0
            },
            'token_stats': {
                'avg_completion_tokens': np.mean(completion_tokens),
                'avg_itl_count': np.mean(itl_counts)
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Visualize request timelines and concurrent processing")
    parser.add_argument("input_file", help="Path to benchmark JSON file")
    parser.add_argument("--max-requests", type=int, default=50, 
                       help="Maximum number of requests to show in gantt chart")
    parser.add_argument("--detailed-requests", nargs='+', type=int, 
                       help="Specific request indices for detailed view")
    parser.add_argument("--output-prefix", default="timeline", 
                       help="Output file prefix")
    parser.add_argument("--no-show", action="store_true", 
                       help="Don't display plots")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"📂 Loading data from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # 创建可视化器
    visualizer = RequestTimelineVisualizer(data)
    
    if not visualizer.timelines:
        print("❌ No valid timeline data found")
        return
    
    # 生成甘特图
    print("📊 Creating Gantt chart...")
    gantt_fig = visualizer.create_gantt_chart(max_requests=args.max_requests)
    if gantt_fig:
        gantt_output = f"{args.output_prefix}_gantt.png"
        gantt_fig.savefig(gantt_output, dpi=300, bbox_inches='tight')
        print(f"💾 Gantt chart saved to: {gantt_output}")
        
        if not args.no_show:
            plt.show()
    
    # 生成详细视图
    if args.detailed_requests:
        print("📊 Creating detailed request view...")
        detailed_fig = visualizer.create_detailed_request_view(args.detailed_requests)
        if detailed_fig:
            detailed_output = f"{args.output_prefix}_detailed.png"
            detailed_fig.savefig(detailed_output, dpi=300, bbox_inches='tight')
            print(f"💾 Detailed view saved to: {detailed_output}")
            
            if not args.no_show:
                plt.show()
    
    # 生成统计报告
    stats = visualizer.generate_statistics_report()
    print(f"\n📈 Statistics Report:")
    print(f"   Total Requests: {stats.get('total_requests', 0)}")
    print(f"   Max Concurrent: {stats.get('concurrency_stats', {}).get('max_concurrent', 0)}")
    print(f"   Avg Concurrent: {stats.get('concurrency_stats', {}).get('avg_concurrent', 0):.1f}")
    print(f"   Avg Queue Time: {stats.get('queue_time_stats', {}).get('mean', 0):.1f}ms")
    print(f"   Avg Prefill Time: {stats.get('prefill_time_stats', {}).get('mean', 0):.1f}ms")
    print(f"   Avg Total Duration: {stats.get('duration_stats', {}).get('mean', 0):.1f}s")


if __name__ == "__main__":
    main()
