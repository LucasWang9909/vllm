#!/usr/bin/env python3
"""
VisionArena数据集快速演示
展示如何下载、处理和使用VisionArena-Chat数据集进行多模态测试
"""

import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))

from vision_arena_processor import VisionArenaProcessor
from vision_benchmark import VisionBenchmark, load_vision_test_data


async def quick_demo():
    """快速演示VisionArena数据集的使用"""
    print("🚀 VisionArena-Chat数据集快速演示")
    print("=" * 60)
    print()
    
    # 第1步: 处理数据集
    print("📋 第1步: 下载和处理VisionArena数据集")
    print("-" * 40)
    
    # 创建处理器
    processor = VisionArenaProcessor("benchmark_datasets/vision_arena")
    
    # 处理少量样本用于演示
    print("🔽 正在下载和处理前50个样本...")
    processed_data = processor.process_dataset(num_samples=50, streaming=True)
    
    if not processed_data:
        print("❌ 数据处理失败，演示结束")
        return
    
    # 保存处理后的数据
    data_file = processor.save_processed_data(processed_data, "demo_vision_data.json")
    processor.print_statistics()
    
    print(f"\n✅ 第1步完成!")
    print(f"   📁 图片保存在: {processor.images_dir}")
    print(f"   📄 数据保存在: {data_file}")
    
    # 第2步: 检查数据
    print(f"\n📋 第2步: 检查处理后的数据")
    print("-" * 40)
    
    # 显示样本信息
    for i, item in enumerate(processed_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"   图片: {Path(item['image_path']).name}")
        print(f"   提示: {item['prompt'][:100]}...")
        
        categories = item['metadata']['categories']
        active_cats = [k for k, v in categories.items() if v]
        print(f"   类别: {active_cats}")
        print(f"   语言: {item['metadata']['language']}")
    
    # 第3步: 运行基准测试（如果vLLM服务可用）
    print(f"\n📋 第3步: 多模态基准测试")
    print("-" * 40)
    
    # 检查vLLM服务是否可用
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    vllm_available = True
                else:
                    vllm_available = False
    except:
        vllm_available = False
    
    if vllm_available:
        print("✅ 检测到vLLM服务，运行示例测试...")
        
        # 创建基准测试实例
        benchmark = VisionBenchmark(rps=2.0)  # 2 RPS用于演示
        
        # 只测试前3个样本
        demo_samples = processed_data[:3]
        
        try:
            results, start_time, end_time = await benchmark.run_vision_benchmark(
                test_data=demo_samples,
                temperature=0.7,
                max_tokens=100
            )
            
            print(f"\n🎯 测试结果:")
            print(f"   成功请求: {sum(1 for r in results if r.success)}/{len(results)}")
            
            for i, result in enumerate(results):
                if result.success:
                    print(f"\n   请求 {i+1} ✅:")
                    print(f"     TTFT: {result.client_ttft_ms:.1f}ms")
                    print(f"     E2E: {result.client_e2e_latency_ms:.1f}ms")
                else:
                    print(f"\n   请求 {i+1} ❌: {result.error_message}")
                    
        except Exception as e:
            print(f"⚠️  基准测试失败: {e}")
            print("   这可能是因为模型不支持多模态或配置问题")
    
    else:
        print("⚠️  未检测到vLLM服务 (http://localhost:8000)")
        print("   请确保已启动支持多模态的vLLM服务")
        print("   例如: vllm serve llava-hf/llava-1.5-7b-hf --port 8000")
    
    # 使用指南
    print(f"\n📋 完整使用指南")
    print("-" * 40)
    print("1️⃣  处理更多数据:")
    print("    python scripts/vision_arena_processor.py --num_samples 1000")
    print()
    print("2️⃣  运行完整基准测试:")
    print("    python scripts/vision_benchmark.py benchmark_datasets/vision_arena/vision_arena_test_data.json --rps 1")
    print()
    print("3️⃣  启动多模态vLLM服务:")
    print("    vllm serve llava-hf/llava-1.5-7b-hf \\")
    print("      --otlp-traces-endpoint http://localhost:4317 \\")
    print("      --collect-detailed-traces all \\")
    print("      --port 8000")
    print()
    print("4️⃣  分析结果:")
    print("    - 查看 results/ 目录中的JSON和CSV文件")
    print("    - 使用现有的可视化工具分析性能")
    
    print(f"\n🎉 演示完成!")


def main():
    print("选择操作:")
    print("1. 快速演示 (下载50个样本并演示)")
    print("2. 处理更多数据 (自定义数量)")
    print("3. 查看使用指南")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(quick_demo())
        elif choice == "2":
            num_samples = input("输入要处理的样本数量 (默认: 1000): ").strip()
            num_samples = int(num_samples) if num_samples else 1000
            
            print(f"🔽 处理 {num_samples} 个样本...")
            processor = VisionArenaProcessor("benchmark_datasets/vision_arena")
            processed_data = processor.process_dataset(num_samples=num_samples, streaming=True)
            
            if processed_data:
                processor.save_processed_data(processed_data, "vision_arena_test_data.json")
                processor.print_statistics()
                print(f"\n✅ 处理完成! 可以使用以下命令进行测试:")
                print(f"python scripts/vision_benchmark.py benchmark_datasets/vision_arena/vision_arena_test_data.json")
        elif choice == "3":
            print("""
🔍 VisionArena-Chat数据集完整使用指南
==========================================

📊 数据集信息:
- 来源: https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat
- 规模: 200K对话, 45个VLM模型, 138种语言
- 内容: 真实用户与多模态模型的对话数据
- 包含: 图片 + 文本对话

🚀 使用步骤:

1️⃣  下载和处理数据:
   python scripts/vision_arena_processor.py --num_samples 1000 --streaming

2️⃣  启动多模态vLLM服务:
   vllm serve llava-hf/llava-1.5-7b-hf \\
     --otlp-traces-endpoint http://localhost:4317 \\
     --collect-detailed-traces all \\
     --port 8000

3️⃣  启动Jaeger追踪服务:
   docker run --rm --name jaeger \\
     -p 16686:16686 -p 14250:14250 -p 4317:4317 -p 4318:4318 \\
     jaegertracing/all-in-one:latest

4️⃣  运行基准测试:
   python scripts/vision_benchmark.py \\
     benchmark_datasets/vision_arena/vision_arena_test_data.json \\
     --rps 1 --num_samples 500

5️⃣  分析结果:
   - JSON/CSV结果保存在 results/ 目录
   - 可使用现有的可视化工具分析性能
   - 支持累计token图、时间线图等

🎯 支持的模型类型:
- LLaVA系列: llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf
- Qwen-VL系列: Qwen/Qwen-VL-Chat
- 其他支持的多模态模型

⚠️  注意事项:
- 确保vLLM版本支持多模态功能
- 图片会自动转换为base64格式发送
- 测试数据包含真实用户对话，注意内容审核
- 建议先用少量样本测试模型兼容性
            """)
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
