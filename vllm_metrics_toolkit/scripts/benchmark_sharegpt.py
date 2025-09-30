#!/usr/bin/env python3
"""
ShareGPT基准测试脚本
使用ShareGPT数据集的前1000个对话进行vLLM性能测试

特点:
- 支持自定义RPS (每秒请求数)
- 使用vllm_metrics_client进行准确的性能指标测量
- 自动保存详细的测试结果
- 支持并发测试和速率限制

使用方法:
python benchmark_sharegpt.py --rps 10 --num_prompts 100
"""

import asyncio
import json
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from vllm_metrics_client import (
    VLLMMetricsClient, 
    RequestMetrics, 
    calculate_benchmark_summary,
    format_benchmark_summary,
    save_multiple_requests_metrics,
    generate_timestamped_filename
)


def load_sharegpt_prompts(dataset_path: str, num_prompts: int = 1000) -> List[str]:
    """
    从ShareGPT数据集中提取前N个human提示
    
    Args:
        dataset_path: 数据集文件路径
        num_prompts: 要提取的提示数量
    
    Returns:
        提示列表
    """
    print(f"Loading ShareGPT dataset from {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for item in data[:num_prompts]:
        # 查找conversations中的第一个human消息
        for conv in item.get('conversations', []):
            if conv.get('from') == 'human':
                prompt = conv.get('value', '').strip()
                if prompt:  # 确保提示不为空
                    prompts.append(prompt)
                break  # 只取每个对话的第一个human消息
    
    print(f"Extracted {len(prompts)} prompts from dataset")
    return prompts


class RateLimitedBenchmark:
    """支持速率限制的基准测试类"""
    
    def __init__(self, rps: float, vllm_url: str = "http://localhost:8000", 
                 jaeger_url: str = "http://localhost:16686"):
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 0
        self.client = VLLMMetricsClient(
            vllm_base_url=vllm_url,
            jaeger_base_url=jaeger_url
        )
        self.results: List[RequestMetrics] = []
    
    async def run_benchmark(self, prompts: List[str], 
                          temperature: float = 0.7, 
                          max_tokens: int = 150) -> List[RequestMetrics]:
        """
        运行基准测试
        
        Args:
            prompts: 要测试的提示列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            测试结果列表
        """
        print(f"\n🚀 Starting benchmark with {len(prompts)} prompts at {self.rps} RPS")
        print(f"   Temperature: {temperature}, Max tokens: {max_tokens}")
        print(f"   Estimated duration: {len(prompts) / self.rps:.1f} seconds\n")
        
        start_time = time.time()
        
        if self.rps <= 0:
            # 无速率限制，并发执行所有请求
            tasks = []
            for i, prompt in enumerate(prompts):
                task = self._send_single_request(prompt, i, temperature, max_tokens)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 过滤掉异常结果
            self.results = [r for r in results if isinstance(r, RequestMetrics)]
        else:
            # 真正的异步RPS控制 - 按间隔启动请求，但不等待完成
            tasks = []
            
            for i, prompt in enumerate(prompts):
                # 创建一个延迟启动的任务
                delay = i * self.interval
                task = asyncio.create_task(
                    self._send_request_with_delay(prompt, i, delay, temperature, max_tokens)
                )
                tasks.append(task)
                
                # 进度显示（基于启动的请求数）
                if (i + 1) % 50 == 0 or i + 1 == len(prompts):
                    print(f"Scheduled: {i+1}/{len(prompts)} requests")
            
            print(f"All {len(prompts)} requests scheduled. Waiting for completion...")
            
            # 等待所有请求完成，同时显示进度
            results = await self._gather_with_progress(tasks, len(prompts))
            # 过滤掉异常结果
            self.results = [r for r in results if isinstance(r, RequestMetrics)]
            
            # 显示最终完成状态
            successful = sum(1 for r in self.results if r.success)
            print(f"\n✅ All requests completed: {len(self.results)} total ({successful} successful)")
        
        end_time = time.time()
        
        # 收集服务端指标
        print("\n📊 Collecting server metrics...")
        await self._collect_server_metrics()
        
        return self.results, start_time, end_time
    
    async def _gather_with_progress(self, tasks: List[asyncio.Task], total: int) -> List[RequestMetrics]:
        """等待所有任务完成，同时显示实时进度"""
        completed = 0
        results = []
        
        # 使用asyncio.as_completed来获取实时完成状态
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if isinstance(result, RequestMetrics):
                    results.append(result)
                completed += 1
                
                # 每完成50个或达到里程碑时显示进度
                if completed % 50 == 0 or completed == total:
                    successful = sum(1 for r in results if r.success)
                    progress = completed / total * 100
                    print(f"Progress: {completed}/{total} ({progress:.1f}%) - "
                          f"Successful: {successful}")
                    
            except Exception as e:
                completed += 1
                print(f"Request failed: {e}")
        
        return results
    
    async def _send_request_with_delay(self, prompt: str, index: int, delay: float,
                                     temperature: float, max_tokens: int) -> RequestMetrics:
        """延迟发送单个请求（用于RPS控制）"""
        await asyncio.sleep(delay)
        return await self._send_single_request(prompt, index, temperature, max_tokens)
    
    async def _send_single_request(self, prompt: str, index: int, 
                                 temperature: float, max_tokens: int) -> RequestMetrics:
        """发送单个请求"""
        try:
            # 将prompt转换为OpenAI chat格式
            messages = [{"role": "user", "content": prompt}]
            
            metrics = await self.client.send_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True  # 使用流式响应以获得更准确的TTFT指标
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error in request {index}: {e}")
            # 返回一个失败的metrics对象
            failed_metrics = RequestMetrics(
                request_id=f"req_{index}",
                trace_id="failed",
                success=False,
                error_message=str(e)
            )
            return failed_metrics
    
    async def _collect_server_metrics(self):
        """批量收集服务端指标"""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("No successful requests to collect server metrics for")
            return
        
        print(f"Collecting server metrics for {len(successful_results)} successful requests...")
        
        # 并发收集服务端指标
        tasks = []
        for metrics in successful_results:
            task = asyncio.create_task(
                self._collect_single_server_metrics(metrics)
            )
            tasks.append(task)
        
        # 等待所有指标收集完成，设置较长的超时时间
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_single_server_metrics(self, metrics: RequestMetrics):
        """收集单个请求的服务端指标"""
        try:
            # 使用同步方法，但在executor中运行以避免阻塞
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.client.enrich_with_server_metrics, 
                metrics, 
                20  # 超时时间
            )
        except Exception as e:
            print(f"Failed to collect server metrics for {metrics.request_id}: {e}")


def print_results_summary(results: List[RequestMetrics], start_time: float, end_time: float):
    """打印结果摘要"""
    summary = calculate_benchmark_summary(results, start_time, end_time)
    print("\n" + "="*80)
    print("📈 BENCHMARK SUMMARY")
    print("="*80)
    print(format_benchmark_summary(summary))


def save_results(results: List[RequestMetrics], base_filename: str):
    """保存测试结果"""
    print(f"\n💾 Saving results...")
    
    # 保存为JSON格式（包含完整数据）
    json_filename = f"{base_filename}.json"
    save_multiple_requests_metrics(results, json_filename, format="json")
    print(f"   ✅ Detailed results saved to: {json_filename}")
    
    # 保存为CSV格式（便于分析）
    csv_filename = f"{base_filename}.csv"
    save_multiple_requests_metrics(results, csv_filename, format="csv")
    print(f"   ✅ CSV results saved to: {csv_filename}")


async def main():
    parser = argparse.ArgumentParser(description="ShareGPT基准测试")
    parser.add_argument("--rps", type=float, default=1.0, 
                       help="每秒请求数 (RPS). 设为0则无限制并发")
    parser.add_argument("--num_prompts", type=int, default=1000,
                       help="要测试的提示数量")
    parser.add_argument("--dataset_path", type=str, 
                       default="benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                       help="ShareGPT数据集路径")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000",
                       help="vLLM服务URL")
    parser.add_argument("--jaeger_url", type=str, default="http://localhost:16686",
                       help="Jaeger UI URL")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--max_tokens", type=int, default=150,
                       help="最大生成token数")
    parser.add_argument("--output_prefix", type=str, default="sharegpt_benchmark",
                       help="输出文件前缀")
    
    args = parser.parse_args()
    
    # 检查数据集文件
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    
    try:
        # 加载提示
        prompts = load_sharegpt_prompts(str(dataset_path), args.num_prompts)
        if not prompts:
            print("❌ No prompts loaded from dataset")
            return
        
        # 创建基准测试实例
        benchmark = RateLimitedBenchmark(
            rps=args.rps,
            vllm_url=args.vllm_url,
            jaeger_url=args.jaeger_url
        )
        
        # 运行测试
        results, start_time, end_time = await benchmark.run_benchmark(
            prompts=prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        if not results:
            print("❌ No results obtained from benchmark")
            return
        
        # 打印摘要
        print_results_summary(results, start_time, end_time)
        
        # 保存结果
        timestamp_filename = generate_timestamped_filename(args.output_prefix)
        save_results(results, timestamp_filename)
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   📊 Total requests: {len(results)}")
        print(f"   ✅ Successful: {sum(1 for r in results if r.success)}")
        print(f"   ❌ Failed: {sum(1 for r in results if not r.success)}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
