#!/usr/bin/env python3
"""
Qwen2.5-Omni simulated vision benchmark (text-only).

This script is identical in structure to qwen_vision_benchmark.py but it
never loads or checks images. It expects items with fields:
  - prompt: text to send (possibly appended with placeholders)
  - expected_response (optional)
  - metadata (optional)

It sends requests as text-only to vLLM and collects metrics.
"""

import asyncio
import json
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add parent dir to import vllm_metrics_client
sys.path.append(str(Path(__file__).parent.parent))

try:
    from vllm_metrics_client import (
        VLLMMetricsClient,
        RequestMetrics,
        calculate_benchmark_summary,
        format_benchmark_summary,
        save_multiple_requests_metrics,
        generate_timestamped_filename,
    )
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)


class QwenSimulateVisionBenchmark:
    """Qwen2.5-Omni simulated (text-only) benchmark."""

    def __init__(self, rps: float, vllm_url: str = "http://localhost:8000", 
                 jaeger_url: str = "http://localhost:16686"):
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 0
        self.client = VLLMMetricsClient(
            vllm_base_url=vllm_url,
            jaeger_base_url=jaeger_url
        )
        self.results: List[RequestMetrics] = []

    def create_messages(self, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": prompt
            }
        ]

    async def send_request(self, test_item: Dict[str, Any], index: int,
                           temperature: float, max_tokens: int) -> RequestMetrics:
        try:
            prompt = test_item.get('prompt', '')
            if not prompt:
                return RequestMetrics(
                    request_id=f"qwen_sim_req_{index}",
                    trace_id="failed",
                    success=False,
                    error_message="Empty prompt"
                )

            messages = self.create_messages(prompt)

            metrics = await self.client.send_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            if hasattr(metrics, 'metadata'):
                metrics.metadata = test_item.get('metadata', {})

            return metrics
        except Exception as e:
            return RequestMetrics(
                request_id=f"qwen_sim_req_{index}",
                trace_id="failed",
                success=False,
                error_message=str(e)
            )

    async def send_request_with_delay(self, test_item: Dict[str, Any], index: int, delay: float,
                                      temperature: float, max_tokens: int) -> RequestMetrics:
        await asyncio.sleep(delay)
        return await self.send_request(test_item, index, temperature, max_tokens)

    async def run(self, test_data: List[Dict[str, Any]], temperature: float = 0.7,
                  max_tokens: int = 150) -> List[RequestMetrics]:
        print(f"\n🚀 开始Qwen2.5-Omni文本模拟基准测试")
        print(f"   测试样本: {len(test_data)}")
        print(f"   RPS: {self.rps}")
        print(f"   温度: {temperature}")
        print(f"   最大tokens: {max_tokens}")

        start_time = time.time()

        if self.rps <= 0:
            tasks = [self.send_request(item, i, temperature, max_tokens)
                     for i, item in enumerate(test_data)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results = [r for r in results if isinstance(r, RequestMetrics)]
        else:
            tasks = []
            for i, item in enumerate(test_data):
                delay = i * self.interval
                task = asyncio.create_task(
                    self.send_request_with_delay(item, i, delay, temperature, max_tokens)
                )
                tasks.append(task)
                if (i + 1) % 20 == 0:
                    print(f"   已调度: {i+1}/{len(test_data)} 请求")
            print(f"   所有 {len(test_data)} 请求已调度，等待完成...")
            results = await self._gather_with_progress(tasks, len(test_data))
            self.results = [r for r in results if isinstance(r, RequestMetrics)]

        end_time = time.time()
        print("\n📊 收集服务端指标...")
        await self._collect_server_metrics()
        return self.results, start_time, end_time

    async def _gather_with_progress(self, tasks: List[asyncio.Task], total: int) -> List[RequestMetrics]:
        completed = 0
        results: List[RequestMetrics] = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if isinstance(result, RequestMetrics):
                    results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    successful = sum(1 for r in results if r.success)
                    progress = completed / total * 100
                    print(f"   进度: {completed}/{total} ({progress:.1f}%) - 成功: {successful}")
            except Exception as e:
                completed += 1
                print(f"   请求失败: {e}")
        return results

    async def _collect_server_metrics(self):
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("   没有成功的请求需要收集服务端指标")
            return
        print(f"   收集 {len(successful_results)} 个成功请求的服务端指标...")
        tasks = [asyncio.create_task(self._collect_single_server_metrics(m))
                 for m in successful_results]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_single_server_metrics(self, metrics: RequestMetrics):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.client.enrich_with_server_metrics,
                metrics,
                20,
            )
        except Exception as e:
            print(f"   收集服务端指标失败 {metrics.request_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni文本模拟基准测试")
    parser.add_argument("data_file", help="Text-only dataset JSON path")
    parser.add_argument("--rps", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--jaeger_url", type=str, default="http://localhost:16686")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--output_prefix", type=str, default="qwen_sim_text_benchmark")
    args = parser.parse_args()

    print("🔍 Qwen2.5-Omni文本模拟基准测试")
    print("=" * 50)
    print(f"数据文件: {args.data_file}")
    print(f"RPS: {args.rps}")

    # Load dataset
    with open(args.data_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if args.num_samples and args.num_samples < len(test_data):
        test_data = test_data[:args.num_samples]
        print(f"🎯 限制测试样本数量: {len(test_data)}")

    async def run_benchmark():
        bench = QwenSimulateVisionBenchmark(
            rps=args.rps,
            vllm_url=args.vllm_url,
            jaeger_url=args.jaeger_url,
        )
        results, start_time, end_time = await bench.run(
            test_data=test_data,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if not results:
            print("❌ 没有获得任何测试结果")
            return
        summary = calculate_benchmark_summary(results, start_time, end_time)
        print("\n" + "=" * 80)
        print("📈 Qwen2.5-Omni文本模拟基准测试摘要")
        print("=" * 80)
        print(format_benchmark_summary(summary))

        timestamp_filename = generate_timestamped_filename(args.output_prefix)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        json_file = results_dir / f"{timestamp_filename}.json"
        csv_file = results_dir / f"{timestamp_filename}.csv"
        save_multiple_requests_metrics(results, str(json_file), format="json")
        save_multiple_requests_metrics(results, str(csv_file), format="csv")
        print(f"\n💾 结果已保存:\n   JSON: {json_file}\n   CSV: {csv_file}")

    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()






