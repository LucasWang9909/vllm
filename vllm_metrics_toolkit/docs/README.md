# vLLM 性能指标监控工具包

一个用于准确测量 vLLM 单请求性能指标的 Python 工具包，支持并发场景下的精确指标获取。

## ✨ 特性

- **🎯 精确指标**: 基于 W3C Trace Context 标准，确保每个请求的指标准确对应
- **📊 全面监控**: 包含所有关键性能指标
  - **TTFT** (Time to First Token): 首次token生成时间
  - **TPOT** (Time per Output Token): 每token平均生成时间  
  - **ITL** (Inter-token Latency): token间平均延迟
  - **队列等待时间**: 请求在队列中的等待时间
  - **预填充时间**: prompt处理时间
  - **解码时间**: token生成时间
  - **端到端延迟**: 完整请求响应时间
- **🚀 并发支持**: 支持多个并发请求的独立指标测量
- **💾 数据保存**: 支持将请求指标保存为JSON/CSV格式，便于后续分析
- **📈 生产就绪**: 可直接用于生产环境的性能监控

## 🔧 环境要求

### vLLM 服务端配置

启动 vLLM 时需要启用 OpenTelemetry 追踪：

```bash
# 启动支持 OpenTelemetry 的 vLLM 服务
vllm serve Qwen/Qwen2.5-Omni-7B \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces all \
  --port 8000
```

### Jaeger 追踪服务

您可以选择使用 Docker 容器或原生安装的方式运行 Jaeger：

#### 选项1: Docker 方式 (推荐)
```bash
# 使用 Docker 启动一次性的Jaeger
docker run --rm --name jaeger \
  -p 16686:16686 -p 14250:14250 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

#### 选项2: 原生安装 (无容器)
如果您不想使用 Docker，可以下载并直接运行 Jaeger：

```bash
# 运行预配置的启动脚本
cd /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger
./start_jaeger.sh
```

或者手动安装：
```bash
# 下载 Jaeger
wget https://github.com/jaegertracing/jaeger/releases/download/v1.52.0/jaeger-1.52.0-linux-amd64.tar.gz
tar -xzf jaeger-1.52.0-linux-amd64.tar.gz
cd jaeger-1.52.0-linux-amd64

# 启动 Jaeger All-in-One
./jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=0.0.0.0:4317 \
  --collector.otlp.http.host-port=0.0.0.0:4318 \
  --query.http-server.host-port=0.0.0.0:16686 \
  --admin.http.host-port=0.0.0.0:14269 \
  --log-level=info
```

📖 **详细安装指南**: 参见 [`docs/NATIVE_JAEGER_INSTALL.md`](./NATIVE_JAEGER_INSTALL.md)

## 📦 安装

1. 克隆或下载此工具包
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 基本用法

```python
import asyncio
from vllm_metrics_client import VLLMMetricsClient, format_metrics

async def basic_example():
    # 创建客户端
    client = VLLMMetricsClient(
        vllm_base_url="http://localhost:8000",      # vLLM 服务地址
        jaeger_base_url="http://localhost:16686"    # Jaeger UI 地址
    )
    
    # 发送请求
    metrics = await client.send_request(
        messages=[{"role": "user", "content": "什么是人工智能？"}],
        temperature=0.7,
        max_tokens=100
    )
    
    # 获取服务端指标
    if metrics.success:
        client.enrich_with_server_metrics(metrics, timeout=15)
    
    # 显示结果
    print(format_metrics(metrics))

# 运行
asyncio.run(basic_example())
```

### 数据保存功能

```python
from vllm_metrics_client import (
    save_request_metrics, 
    save_multiple_requests_metrics,
    generate_timestamped_filename
)

# 保存单个请求数据
filename = generate_timestamped_filename("my_test")
save_request_metrics(metrics, filename, format="json")  # 保存为JSON
save_request_metrics(metrics, filename, format="csv")   # 保存为CSV

# 保存多个请求数据
save_multiple_requests_metrics(results_list, filename, format="json")
```

**保存的数据包含:**
- **发送时间**: 请求发送的时间戳
- **队列时间**: 服务端队列等待时间  
- **TTFT**: 首次token生成时间
- **TPOT**: 每token平均生成时间
- **ITL列表**: 完整的token间延迟数据
- **所有其他指标**: E2E延迟、token统计、请求参数等

### 并发请求测试

```python
async def concurrent_test():
    client = VLLMMetricsClient()
    
    # 创建多个并发请求
    tasks = []
    for i in range(3):
        task = client.send_request(
            messages=[{"role": "user", "content": f"请求{i+1}: 解释AI"}],
            temperature=0.5 + i*0.2,
            max_tokens=50 + i*20
        )
        tasks.append(task)
    
    # 并发执行
    results = await asyncio.gather(*tasks)
    
    # 获取服务端指标
    for metrics in results:
        if metrics.success:
            client.enrich_with_server_metrics(metrics)
    
    # 显示对比
    for i, metrics in enumerate(results):
        print(f"\n请求{i+1}指标:")
        print(format_metrics(metrics, show_details=False))

# 运行
asyncio.run(concurrent_test())
```

### 基准测试汇总统计

```python
from vllm_metrics_client import calculate_benchmark_summary, format_benchmark_summary
import time

async def benchmark_test():
    client = VLLMMetricsClient()
    
    # 发送多个测试请求
    start_time = time.time()
    results = []
    
    for i in range(10):
        metrics = await client.send_request(
            messages=[{"role": "user", "content": f"测试请求{i+1}"}],
            temperature=0.7,
            max_tokens=50
        )
        results.append(metrics)
    
    end_time = time.time()
    
    # 获取服务端指标
    for metrics in results:
        if metrics.success:
            client.enrich_with_server_metrics(metrics)
    
    # 生成汇总报告
    summary = calculate_benchmark_summary(results, start_time, end_time)
    print(format_benchmark_summary(summary))
```

## 📊 指标说明

### 客户端指标
- **client_e2e_latency**: 客户端测量的端到端延迟 (ms)
- **client_ttft**: 客户端测量的首次token时间 (ms)
- **client_tpot**: 客户端测量的每token平均时间 (ms)
- **client_itl**: 客户端测量的token间平均延迟 (ms)

### 服务端指标 (从 OpenTelemetry 获取)
- **server_queue_time**: 请求在队列中的等待时间 (ms)
- **server_prefill_time**: prompt预处理时间 (ms)
- **server_decode_time**: token生成时间 (ms)
- **server_inference_time**: 总推理时间 (ms)
- **server_e2e_time**: 服务端端到端时间 (ms)
- **server_ttft**: 服务端首次token时间 (ms)

### Token 统计
- **prompt_tokens**: 输入token数量
- **completion_tokens**: 输出token数量

## 🔧 高级配置

### 自定义客户端配置

```python
client = VLLMMetricsClient(
    vllm_base_url="http://your-vllm-server:8000",
    jaeger_base_url="http://your-jaeger:16686",
    otlp_endpoint="http://your-otlp-collector:4317"
)
```

### 请求参数选项

```python
metrics = await client.send_request(
    messages=[{"role": "user", "content": "你的问题"}],
    model="auto",                    # 模型名称，"auto"使用默认
    temperature=0.7,                 # 温度参数
    max_tokens=100,                  # 最大token数
    top_p=1.0,                      # Top-p采样
    stream=True,                     # 是否流式响应
    # 其他OpenAI兼容参数...
)
```

## 🔍 故障排查

### 常见问题

1. **无法获取服务端指标**
   - 确认 vLLM 启动时启用了 OpenTelemetry: `--otlp-traces-endpoint`
   - 确认 Jaeger 服务正在运行: `docker ps | grep jaeger`
   - 增加 timeout 参数: `client.enrich_with_server_metrics(metrics, timeout=30)`

2. **请求失败**
   - 检查 vLLM 服务状态和地址
   - 确认模型已正确加载
   - 查看错误信息: `print(metrics.error_message)`

3. **指标不准确**
   - 本工具使用 UUID 和 OpenTelemetry 保证指标准确性
   - 即使在高并发场景下也能正确区分每个请求
   - 无需担心指标混合问题

### 调试模式

```python
# 打开详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查 trace 数据
print(f"Trace ID: {metrics.trace_id}")
print(f"Jaeger URL: http://localhost:16686/trace/{metrics.trace_id}")
```

## 📈 性能监控最佳实践

1. **生产环境监控**
   - 定期收集关键指标(TTFT, 队列时间, 吞吐量)
   - 设置指标阈值告警
   - 监控并发场景下的性能表现

2. **性能优化**
   - 关注队列等待时间，优化并发处理
   - 监控TTFT，优化模型加载和预填充
   - 分析TPOT/ITL，优化token生成速度

3. **容量规划**
   - 使用并发测试评估系统容量
   - 基于E2E延迟规划用户体验
   - 根据队列时间调整资源配置

## 🤝 技术原理

本工具基于以下技术保证指标准确性：

- **W3C Trace Context**: 国际标准的分布式追踪协议
- **UUID 唯一性**: 每个请求都有全球唯一的追踪ID
- **OpenTelemetry**: 服务端指标的标准化收集
- **流式响应处理**: 客户端实时计算token级别指标

这确保了即使在高并发场景下，每个请求的指标都能准确对应，无混合或错误归属问题。

## 📄 许可证

MIT License

## 🆘 支持

如有问题或建议，请：
1. 查看本文档的故障排查部分
2. 检查 Jaeger UI 中的追踪数据: http://localhost:16686
3. 验证 vLLM 服务的 OpenTelemetry 配置

---

**注意**: 本工具需要 vLLM 服务启用 OpenTelemetry 支持才能获取完整的服务端指标。客户端指标(TTFT, TPOT, ITL)无需额外配置即可准确测量。