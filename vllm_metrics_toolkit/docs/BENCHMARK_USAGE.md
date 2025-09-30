# ShareGPT基准测试使用指南

这个工具包现在包含了专门的ShareGPT基准测试功能，可以使用前1000个对话提示进行性能测试。

## 📋 前提条件

### 1. 启动vLLM服务
```bash
# 启动支持OpenTelemetry的vLLM服务
vllm serve <your-model> \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces all \
  --port 8000
```

### 2. 启动Jaeger追踪服务
```bash
# 使用Docker启动Jaeger
docker run --rm --name jaeger \
  -p 16686:16686 -p 14250:14250 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

### 3. 激活Python环境
```bash
cd /home/ubuntu/vllm
source .venv/bin/activate
```

## 🚀 运行基准测试

### 方法1: 使用交互式脚本 (推荐)
```bash
cd /home/ubuntu/vllm/vllm_metrics_toolkit
python run_benchmark_example.py
```

这会给你选择：
- **快速测试**: 10个提示，2 RPS（约5秒完成）
- **完整基准测试**: 1000个提示，1 RPS（约17分钟完成）

### 方法2: 直接使用命令行
```bash
cd /home/ubuntu/vllm/vllm_metrics_toolkit

# 快速测试 - 10个提示
python benchmark_sharegpt.py --rps 2 --num_prompts 10 --max_tokens 50

# 完整基准测试 - 1000个提示，1 RPS
python benchmark_sharegpt.py --rps 1 --num_prompts 1000

# 自定义配置
python benchmark_sharegpt.py \
  --rps 1.5 \
  --num_prompts 500 \
  --temperature 0.8 \
  --max_tokens 200 \
  --output_prefix "my_test"
```

## 📊 输出结果

### 控制台输出
测试过程中会显示：
- 进度更新（每50个请求）
- 实时统计
- 最终基准测试摘要

### 保存的文件
测试完成后会自动保存：
- `{timestamp}_results.json` - 完整的详细数据
- `{timestamp}_results.csv` - 便于分析的CSV格式

### 关键指标
- **TTFT** (Time to First Token): 首次token生成时间
- **TPOT** (Time per Output Token): 每token平均生成时间
- **ITL** (Inter-token Latency): token间延迟
- **队列时间**: 服务端队列等待时间
- **端到端延迟**: 完整请求响应时间

## 🎛️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rps` | 1.0 | 每秒请求数。设为0则无限制并发 |
| `--num_prompts` | 1000 | 要测试的提示数量 |
| `--temperature` | 0.7 | 模型温度参数 |
| `--max_tokens` | 150 | 最大生成token数 |
| `--vllm_url` | http://localhost:8000 | vLLM服务URL |
| `--jaeger_url` | http://localhost:16686 | Jaeger UI URL |
| `--output_prefix` | sharegpt_benchmark | 输出文件前缀 |

## 🔍 数据集信息

- **数据集**: ShareGPT_V3_unfiltered_cleaned_split.json
- **总条目数**: 94,145个对话
- **测试范围**: 前1000个human提示
- **提示来源**: 每个对话中的第一个human消息

## 📈 结果分析

### 查看Jaeger追踪
1. 打开 http://localhost:16686
2. 搜索服务名: `vllm`
3. 查看详细的请求追踪信息

### CSV数据分析
可以使用Excel、Python pandas等工具分析CSV结果：
```python
import pandas as pd
df = pd.read_csv('your_results.csv')
print(df.describe())  # 统计摘要
```

## ⚠️ 注意事项

1. **测试时间**: 1000个提示@1RPS大约需要17分钟
2. **资源使用**: 确保vLLM服务有足够的GPU内存
3. **网络连接**: 确保客户端到vLLM服务的网络稳定
4. **中断恢复**: 可以使用Ctrl+C安全中断测试

## 🛠️ 故障排查

### 常见问题
1. **"No prompts loaded"**: 检查数据集文件路径
2. **连接失败**: 确认vLLM服务正在运行
3. **无服务端指标**: 检查OpenTelemetry和Jaeger配置
4. **内存不足**: 减少max_tokens或并发数

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# 然后运行测试脚本
```
