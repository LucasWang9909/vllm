#!/usr/bin/env python3
"""
多模态视觉基准测试
使用VisionArena数据集测试vLLM多模态模型的性能

支持功能:
- 图片+文本的多模态请求
- 异步RPS控制
- 完整的性能指标收集
- 可视化生成
"""

import asyncio
import json
import time
import argparse
import base64
from typing import List, Dict, Any
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from vllm_metrics_client import (
        VLLMMetricsClient, 
        RequestMetrics, 
        calculate_benchmark_summary,
        format_benchmark_summary,
        save_multiple_requests_metrics,
        generate_timestamped_filename
    )
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请确保已安装: PIL/Pillow")
    sys.exit(1)


class VisionBenchmark:
    """多模态视觉基准测试类"""
    
    def __init__(self, rps: float, vllm_url: str = "http://localhost:8000", 
                 jaeger_url: str = "http://localhost:16686"):
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 0
        self.client = VLLMMetricsClient(
            vllm_base_url=vllm_url,
            jaeger_base_url=jaeger_url
        )
        self.results: List[RequestMetrics] = []
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64字符串"""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            print(f"⚠️  图片编码失败 {image_path}: {e}")
            return ""
    
    def create_vision_messages(self, image_path: str, prompt: str) -> List[Dict[str, Any]]:
        """
        创建多模态消息格式
        
        Args:
            image_path: 图片路径
            prompt: 文本提示
            
        Returns:
            OpenAI格式的多模态消息
        """
        # 编码图片
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return []
        
        # 创建多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    }
                ]
            }
        ]
        
        return messages
    
    async def send_vision_request(self, test_item: Dict[str, Any], index: int,
                                temperature: float, max_tokens: int) -> RequestMetrics:
        """发送单个多模态请求"""
        try:
            image_path = test_item['image_path']
            prompt = test_item['prompt']
            
            # 检查图片文件是否存在
            if not Path(image_path).exists():
                print(f"⚠️  图片不存在: {image_path}")
                failed_metrics = RequestMetrics(
                    request_id=f"vision_req_{index}",
                    trace_id="failed",
                    success=False,
                    error_message=f"Image not found: {image_path}"
                )
                return failed_metrics
            
            # 创建多模态消息
            messages = self.create_vision_messages(image_path, prompt)
            if not messages:
                failed_metrics = RequestMetrics(
                    request_id=f"vision_req_{index}",
                    trace_id="failed",
                    success=False,
                    error_message="Failed to encode image"
                )
                return failed_metrics
            
            # 发送请求
            metrics = await self.client.send_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True  # 使用流式响应以获得准确的TTFT
            )
            
            # 添加额外的元数据
            if hasattr(metrics, 'metadata'):
                metrics.metadata = test_item.get('metadata', {})
            
            return metrics
            
        except Exception as e:
            print(f"❌ 请求失败 {index}: {e}")
            failed_metrics = RequestMetrics(
                request_id=f"vision_req_{index}",
                trace_id="failed",
                success=False,
                error_message=str(e)
            )
            return failed_metrics
    
    async def send_request_with_delay(self, test_item: Dict[str, Any], index: int, delay: float,
                                    temperature: float, max_tokens: int) -> RequestMetrics:
        """延迟发送请求（用于RPS控制）"""
        await asyncio.sleep(delay)
        return await self.send_vision_request(test_item, index, temperature, max_tokens)
    
    async def run_vision_benchmark(self, test_data: List[Dict[str, Any]], 
                                 temperature: float = 0.7, 
                                 max_tokens: int = 150) -> List[RequestMetrics]:
        """
        运行多模态基准测试
        
        Args:
            test_data: 测试数据列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            测试结果列表
        """
        print(f"\n🚀 开始多模态基准测试")
        print(f"   测试样本: {len(test_data)}")
        print(f"   RPS: {self.rps}")
        print(f"   温度: {temperature}")
        print(f"   最大tokens: {max_tokens}")
        
        start_time = time.time()
        
        if self.rps <= 0:
            # 无速率限制，并发执行
            tasks = []
            for i, test_item in enumerate(test_data):
                task = self.send_vision_request(test_item, i, temperature, max_tokens)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results = [r for r in results if isinstance(r, RequestMetrics)]
        else:
            # 异步RPS控制
            tasks = []
            
            for i, test_item in enumerate(test_data):
                delay = i * self.interval
                task = asyncio.create_task(
                    self.send_request_with_delay(test_item, i, delay, temperature, max_tokens)
                )
                tasks.append(task)
                
                if (i + 1) % 50 == 0:
                    print(f"   已调度: {i+1}/{len(test_data)} 请求")
            
            print(f"   所有 {len(test_data)} 请求已调度，等待完成...")
            
            # 等待所有请求完成，同时显示进度
            results = await self._gather_with_progress(tasks, len(test_data))
            self.results = [r for r in results if isinstance(r, RequestMetrics)]
        
        end_time = time.time()
        
        # 收集服务端指标
        print("\n📊 收集服务端指标...")
        await self._collect_server_metrics()
        
        return self.results, start_time, end_time
    
    async def _gather_with_progress(self, tasks: List[asyncio.Task], total: int) -> List[RequestMetrics]:
        """等待所有任务完成，同时显示进度"""
        completed = 0
        results = []
        
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if isinstance(result, RequestMetrics):
                    results.append(result)
                completed += 1
                
                if completed % 20 == 0 or completed == total:
                    successful = sum(1 for r in results if r.success)
                    progress = completed / total * 100
                    print(f"   进度: {completed}/{total} ({progress:.1f}%) - "
                          f"成功: {successful}")
                    
            except Exception as e:
                completed += 1
                print(f"   请求失败: {e}")
        
        return results
    
    async def _collect_server_metrics(self):
        """收集服务端指标"""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("   没有成功的请求需要收集服务端指标")
            return
        
        print(f"   收集 {len(successful_results)} 个成功请求的服务端指标...")
        
        tasks = []
        for metrics in successful_results:
            task = asyncio.create_task(
                self._collect_single_server_metrics(metrics)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_single_server_metrics(self, metrics: RequestMetrics):
        """收集单个请求的服务端指标"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.client.enrich_with_server_metrics, 
                metrics, 
                20
            )
        except Exception as e:
            print(f"   收集服务端指标失败 {metrics.request_id}: {e}")


def load_vision_test_data(data_file: str) -> List[Dict[str, Any]]:
    """加载视觉测试数据"""
    print(f"📂 加载测试数据: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 加载了 {len(data)} 个测试样本")
        
        # 显示前几个样本的信息
        for i, item in enumerate(data[:3]):
            categories = item.get('metadata', {}).get('categories', {})
            active_categories = [k for k, v in categories.items() if v]
            print(f"   样本 {i}: {item['prompt'][:50]}...")
            print(f"           类别: {active_categories}")
            print(f"           图片: {Path(item['image_path']).name}")
        
        if len(data) > 3:
            print(f"   ... 还有 {len(data) - 3} 个样本")
        
        return data
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="多模态视觉基准测试")
    parser.add_argument("data_file", help="VisionArena测试数据文件路径")
    parser.add_argument("--rps", type=float, default=1.0, 
                       help="每秒请求数 (RPS)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="要测试的样本数量 (默认: 全部)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000",
                       help="vLLM服务URL")
    parser.add_argument("--jaeger_url", type=str, default="http://localhost:16686",
                       help="Jaeger UI URL")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--max_tokens", type=int, default=150,
                       help="最大生成token数")
    parser.add_argument("--output_prefix", type=str, default="vision_benchmark",
                       help="输出文件前缀")
    
    args = parser.parse_args()
    
    print("🔍 多模态视觉基准测试")
    print("=" * 50)
    print(f"数据文件: {args.data_file}")
    print(f"RPS: {args.rps}")
    print(f"vLLM URL: {args.vllm_url}")
    
    # 加载测试数据
    test_data = load_vision_test_data(args.data_file)
    if not test_data:
        return
    
    # 限制样本数量
    if args.num_samples and args.num_samples < len(test_data):
        test_data = test_data[:args.num_samples]
        print(f"🎯 限制测试样本数量: {len(test_data)}")
    
    async def run_benchmark():
        # 创建基准测试实例
        benchmark = VisionBenchmark(
            rps=args.rps,
            vllm_url=args.vllm_url,
            jaeger_url=args.jaeger_url
        )
        
        # 运行测试
        results, start_time, end_time = await benchmark.run_vision_benchmark(
            test_data=test_data,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        if not results:
            print("❌ 没有获得任何测试结果")
            return
        
        # 生成摘要
        summary = calculate_benchmark_summary(results, start_time, end_time)
        print("\n" + "="*80)
        print("📈 多模态基准测试摘要")
        print("="*80)
        print(format_benchmark_summary(summary))
        
        # 保存结果
        timestamp_filename = generate_timestamped_filename(args.output_prefix)
        
        # 保存到results目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        json_file = results_dir / f"{timestamp_filename}.json"
        csv_file = results_dir / f"{timestamp_filename}.csv"
        
        save_multiple_requests_metrics(results, str(json_file), format="json")
        save_multiple_requests_metrics(results, str(csv_file), format="csv")
        
        print(f"\n💾 结果已保存:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
        
        print(f"\n✅ 多模态基准测试完成!")
        print(f"   📊 总请求: {len(results)}")
        print(f"   ✅ 成功: {sum(1 for r in results if r.success)}")
        print(f"   ❌ 失败: {sum(1 for r in results if not r.success)}")
    
    # 运行异步基准测试
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
