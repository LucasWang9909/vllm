#!/usr/bin/env python3
"""
快速运行ShareGPT基准测试的示例脚本
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_sharegpt import main, RateLimitedBenchmark, load_sharegpt_prompts
from vllm_metrics_client import generate_timestamped_filename


async def quick_test():
    """快速测试示例 - 使用前10个提示进行测试"""
    
    print("🔬 Quick Test - Testing with first 10 prompts")
    print("="*50)
    
    # 加载少量提示用于快速测试
    dataset_path = "benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    prompts = load_sharegpt_prompts(dataset_path, num_prompts=10)
    
    if not prompts:
        print("❌ No prompts loaded")
        return
    
    # 创建基准测试实例
    benchmark = RateLimitedBenchmark(rps=2.0)  # 2 RPS for quick test
    
    # 运行测试
    results, start_time, end_time = await benchmark.run_benchmark(
        prompts=prompts,
        temperature=0.7,
        max_tokens=50  # 较少的tokens用于快速测试
    )
    
    # 显示结果
    from vllm_metrics_client import calculate_benchmark_summary, format_benchmark_summary
    summary = calculate_benchmark_summary(results, start_time, end_time)
    print("\n" + format_benchmark_summary(summary))
    
    print(f"\n✅ Quick test completed!")
    print(f"   📊 Total requests: {len(results)}")
    print(f"   ✅ Successful: {sum(1 for r in results if r.success)}")


async def full_benchmark():
    """完整的1000个提示基准测试"""
    
    print("🚀 Full Benchmark - 1000 prompts at 1 RPS")
    print("="*50)
    print("⚠️  This will take approximately 16-17 minutes to complete")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Benchmark cancelled")
        return
    
    # 加载1000个提示
    dataset_path = "benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    prompts = load_sharegpt_prompts(dataset_path, num_prompts=1000)
    
    if not prompts:
        print("❌ No prompts loaded")
        return
    
    # 创建基准测试实例
    benchmark = RateLimitedBenchmark(rps=1.0)  # 1 RPS as requested
    
    # 运行测试
    results, start_time, end_time = await benchmark.run_benchmark(
        prompts=prompts,
        temperature=0.7,
        max_tokens=150
    )
    
    # 显示结果
    from vllm_metrics_client import (
        calculate_benchmark_summary, 
        format_benchmark_summary,
        save_multiple_requests_metrics,
        generate_timestamped_filename
    )
    
    summary = calculate_benchmark_summary(results, start_time, end_time)
    print("\n" + format_benchmark_summary(summary))
    
    # 保存结果
    filename = generate_timestamped_filename("sharegpt_1000_rps1")
    save_multiple_requests_metrics(results, f"{filename}.json", format="json")
    save_multiple_requests_metrics(results, f"{filename}.csv", format="csv")
    
    print(f"\n✅ Full benchmark completed!")
    print(f"   📊 Total requests: {len(results)}")
    print(f"   ✅ Successful: {sum(1 for r in results if r.success)}")
    print(f"   💾 Results saved to: {filename}.json and {filename}.csv")


async def main():
    print("ShareGPT Benchmark Test Options")
    print("="*40)
    print("1. Quick test (10 prompts, 2 RPS)")
    print("2. Full benchmark (1000 prompts, 1 RPS)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == "1":
        await quick_test()
    elif choice == "2":
        await full_benchmark()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
