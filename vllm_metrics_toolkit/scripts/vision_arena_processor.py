#!/usr/bin/env python3
"""
VisionArena-Chat数据集处理器
处理来自Hugging Face的VisionArena-Chat数据集，提取图片和对话数据
用于vLLM多模态模型的基准测试

数据集信息:
- 200K对话
- 45个VLM模型
- 138种语言
- ~43K独特图片
- 包含各种类别：Captioning, OCR, Entity Recognition等
"""

import os
import sys
import json
import argparse
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Tuple
import logging

# 添加父目录到路径以导入vllm_metrics_client
sys.path.append(str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv pip install datasets Pillow")
    sys.exit(1)


class VisionArenaProcessor:
    """VisionArena数据集处理器"""
    
    def __init__(self, output_dir: str = "benchmark_datasets/vision_arena"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.conversations_dir = self.output_dir / "conversations"
        self.images_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'saved_images': 0,
            'failed_images': 0,
            'categories': {},
            'languages': {},
            'models': {}
        }
    
    def load_dataset_sample(self, num_samples: int = 1000, streaming: bool = False) -> Any:
        """
        加载VisionArena数据集
        
        Args:
            num_samples: 要处理的样本数量（None表示全部）
            streaming: 是否使用流式加载（适合大数据集）
        """
        print(f"📂 加载VisionArena-Chat数据集...")
        print(f"   样本数量: {num_samples if num_samples else '全部'}")
        print(f"   流式加载: {streaming}")
        
        try:
            if streaming:
                dataset = load_dataset(
                    "lmarena-ai/VisionArena-Chat", 
                    streaming=True,
                    split="train"
                )
                if num_samples:
                    dataset = dataset.take(num_samples)
            else:
                dataset = load_dataset(
                    "lmarena-ai/VisionArena-Chat",
                    split=f"train[:{num_samples}]" if num_samples else "train"
                )
                
            return dataset
            
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return None
    
    def extract_image_from_sample(self, sample: Dict[str, Any], sample_id: str) -> Tuple[str, bool]:
        """
        从样本中提取并保存图片
        
        Args:
            sample: 数据集样本
            sample_id: 样本ID
            
        Returns:
            (图片路径, 是否成功)
        """
        try:
            images = sample.get('images', [])
            if not images:
                return "", False
            
            # 取第一张图片
            image_data = images[0]
            
            # 解码图片
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
                image = Image.open(BytesIO(image_bytes))
            else:
                # 如果是其他格式，尝试直接打开
                image = image_data
            
            # 保存图片
            image_filename = f"image_{sample_id}.jpg"
            image_path = self.images_dir / image_filename
            
            # 转换为RGB（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(image_path, 'JPEG', quality=85)
            self.stats['saved_images'] += 1
            
            return str(image_path), True
            
        except Exception as e:
            print(f"⚠️  图片提取失败 (样本 {sample_id}): {e}")
            self.stats['failed_images'] += 1
            return "", False
    
    def process_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理对话数据，转换为vLLM测试格式
        
        Args:
            sample: 原始样本数据
            
        Returns:
            处理后的对话数据
        """
        conversation = sample.get('conversation', [])
        
        # 提取第一轮对话（用户问题）
        user_message = ""
        assistant_message = ""
        
        # conversation是嵌套列表结构: [[{user}], [{assistant}]]
        for turn_group in conversation:
            if isinstance(turn_group, list):
                for turn in turn_group:
                    if isinstance(turn, dict):
                        role = turn.get('role', '')
                        content = turn.get('content', '')
                        
                        if role == 'user' and not user_message:
                            user_message = content
                        elif role == 'assistant' and not assistant_message:
                            assistant_message = content
                            break
            elif isinstance(turn_group, dict):
                # 如果直接是字典格式
                role = turn_group.get('role', '')
                content = turn_group.get('content', '')
                
                if role == 'user' and not user_message:
                    user_message = content
                elif role == 'assistant' and not assistant_message:
                    assistant_message = content
                    break
        
        # 统计类别
        categories = sample.get('categories', {})
        for category, is_present in categories.items():
            if is_present:
                self.stats['categories'][category] = self.stats['categories'].get(category, 0) + 1
        
        # 统计语言和模型
        language = sample.get('language', 'unknown')
        model = sample.get('model', 'unknown')
        self.stats['languages'][language] = self.stats['languages'].get(language, 0) + 1
        self.stats['models'][model] = self.stats['models'].get(model, 0) + 1
        
        return {
            'user_message': user_message,
            'assistant_message': assistant_message,
            'categories': categories,
            'language': language,
            'model': model,
            'num_turns': sample.get('num_turns', 1),
            'conversation_id': sample.get('conversation_id', ''),
            'timestamp': sample.get('tstamp', 0)
        }
    
    def create_vllm_test_format(self, image_path: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建适合vLLM测试的格式
        
        Args:
            image_path: 图片路径
            conversation: 对话数据
            
        Returns:
            vLLM测试格式的数据
        """
        return {
            'image_path': image_path,
            'prompt': conversation['user_message'],
            'expected_response': conversation['assistant_message'],
            'metadata': {
                'categories': conversation['categories'],
                'language': conversation['language'],
                'original_model': conversation['model'],
                'num_turns': conversation['num_turns'],
                'conversation_id': conversation['conversation_id']
            }
        }
    
    def process_dataset(self, num_samples: int = 1000, streaming: bool = True) -> List[Dict[str, Any]]:
        """
        处理整个数据集
        
        Args:
            num_samples: 处理的样本数量
            streaming: 是否使用流式处理
            
        Returns:
            处理后的测试数据列表
        """
        print(f"🚀 开始处理VisionArena数据集...")
        
        # 加载数据集
        dataset = self.load_dataset_sample(num_samples, streaming)
        if dataset is None:
            return []
        
        processed_data = []
        
        try:
            # 处理样本
            for i, sample in enumerate(dataset):
                self.stats['total_samples'] += 1
                
                # 生成样本ID
                sample_id = f"{i:06d}"
                
                # 提取图片
                image_path, image_success = self.extract_image_from_sample(sample, sample_id)
                
                if not image_success:
                    continue
                
                # 处理对话
                conversation = self.process_conversation(sample)
                
                # 检查是否有有效的用户消息
                if not conversation['user_message'].strip():
                    continue
                
                # 创建测试格式
                test_item = self.create_vllm_test_format(image_path, conversation)
                processed_data.append(test_item)
                
                self.stats['processed_samples'] += 1
                
                # 进度显示
                if (i + 1) % 100 == 0:
                    print(f"   处理进度: {i+1}/{num_samples if num_samples else '?'} "
                          f"(成功: {self.stats['processed_samples']}, "
                          f"图片: {self.stats['saved_images']})")
                
                # 如果不是流式且达到了指定数量，跳出
                if not streaming and num_samples and len(processed_data) >= num_samples:
                    break
                    
        except Exception as e:
            print(f"❌ 处理过程中出错: {e}")
        
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], filename: str = "vision_arena_test_data.json"):
        """保存处理后的数据"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 数据已保存: {output_file}")
        print(f"   处理的样本: {len(data)}")
        
        return str(output_file)
    
    def print_statistics(self):
        """打印统计信息"""
        print(f"\n📊 处理统计:")
        print(f"   总样本数: {self.stats['total_samples']}")
        print(f"   成功处理: {self.stats['processed_samples']}")
        print(f"   保存图片: {self.stats['saved_images']}")
        print(f"   失败图片: {self.stats['failed_images']}")
        
        print(f"\n🏷️  类别分布 (前10):")
        sorted_categories = sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            print(f"   {category}: {count}")
        
        print(f"\n🌍 语言分布 (前10):")
        sorted_languages = sorted(self.stats['languages'].items(), key=lambda x: x[1], reverse=True)
        for language, count in sorted_languages[:10]:
            print(f"   {language}: {count}")
        
        print(f"\n🤖 模型分布 (前10):")
        sorted_models = sorted(self.stats['models'].items(), key=lambda x: x[1], reverse=True)
        for model, count in sorted_models[:10]:
            print(f"   {model}: {count}")


def main():
    parser = argparse.ArgumentParser(description="处理VisionArena-Chat数据集")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="要处理的样本数量 (默认: 1000)")
    parser.add_argument("--output_dir", type=str, default="benchmark_datasets/vision_arena",
                       help="输出目录 (默认: benchmark_datasets/vision_arena)")
    parser.add_argument("--streaming", action="store_true",
                       help="使用流式加载 (适合大数据集)")
    parser.add_argument("--output_file", type=str, default="vision_arena_test_data.json",
                       help="输出文件名 (默认: vision_arena_test_data.json)")
    
    args = parser.parse_args()
    
    print("🔍 VisionArena-Chat数据集处理器")
    print("=" * 50)
    print(f"数据集来源: https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat")
    print(f"样本数量: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"流式处理: {args.streaming}")
    
    # 创建处理器
    processor = VisionArenaProcessor(args.output_dir)
    
    # 处理数据集
    processed_data = processor.process_dataset(
        num_samples=args.num_samples,
        streaming=args.streaming
    )
    
    if processed_data:
        # 保存数据
        output_file = processor.save_processed_data(processed_data, args.output_file)
        
        # 打印统计
        processor.print_statistics()
        
        print(f"\n✅ 处理完成!")
        print(f"   📁 图片目录: {processor.images_dir}")
        print(f"   📄 数据文件: {output_file}")
        print(f"   🎯 可用于vLLM多模态模型测试")
        
    else:
        print("❌ 未能处理任何数据")


if __name__ == "__main__":
    main()
