#!/usr/bin/env python3
"""
vLLM性能指标客户端
结合OpenTelemetry追踪和客户端测量，提供完整的单请求性能指标

特点:
- 支持并发请求的准确指标获取
- 包含所有关键指标: TTFT, TPOT, ITL, 队列时间等
- 基于W3C Trace Context标准，无指标混合问题
- 生产就绪，可直接用于性能监控

"""

import asyncio
import aiohttp
import json
import time
import uuid
import requests
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RequestMetrics:
    """单请求的完整性能指标"""
    # 基础信息
    request_id: str
    trace_id: str
    success: bool = False
    error_message: str = ""
    
    # 时间信息
    send_time: Optional[float] = None                # 请求发送时间戳
    
    @property
    def send_time_iso(self) -> Optional[str]:
        """获取ISO格式的发送时间"""
        if self.send_time:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.send_time))
        return None
    
    # 客户端测量指标 (毫秒)
    client_e2e_latency: float = 0.0              # 端到端延迟
    client_ttft: Optional[float] = None          # 首次token时间
    client_tpot: Optional[float] = None          # 每token平均时间
    client_itl: Optional[float] = None           # token间平均延迟
    
    # 服务端指标 (毫秒, 从OpenTelemetry获取)
    server_queue_time: Optional[float] = None    # 队列等待时间
    server_prefill_time: Optional[float] = None  # 预填充时间
    server_decode_time: Optional[float] = None   # 解码时间
    server_inference_time: Optional[float] = None # 推理总时间
    server_e2e_time: Optional[float] = None      # 服务端E2E时间
    server_ttft: Optional[float] = None          # 服务端TTFT
    
    
    # Token统计
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    
    # 请求参数
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    
    # 内容
    generated_text: str = ""
    
    # 客户端详细数据
    client_itl_list: List[float] = field(default_factory=list)  # 完整的token间延迟列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.send_time,
            "send_time_iso": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.send_time)) if self.send_time else None,
            "client_metrics": {
                "e2e_latency_ms": self.client_e2e_latency,
                "ttft_ms": self.client_ttft,
                "tpot_ms": self.client_tpot,
                "itl_ms": self.client_itl,
            },
            "server_metrics": {
                "queue_time_ms": self.server_queue_time,
                "prefill_time_ms": self.server_prefill_time,
                "decode_time_ms": self.server_decode_time,
                "inference_time_ms": self.server_inference_time,
                "e2e_time_ms": self.server_e2e_time,
                "ttft_ms": self.server_ttft,
                
            },
            "tokens": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
            "request_params": {
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
            "content": {
                "generated_text": self.generated_text,
                "text_length": len(self.generated_text),
            },
            "detailed_data": {
                "itl_list_seconds": self.client_itl_list,  # 原始秒为单位的ITL列表
                "itl_list_ms": [itl * 1000 for itl in self.client_itl_list] if self.client_itl_list else [],  # 毫秒为单位的ITL列表
            }
        }

class VLLMMetricsClient:
    """vLLM性能指标客户端"""
    
    def __init__(
        self, 
        vllm_base_url: str = "http://localhost:8000",
        jaeger_base_url: str = "http://localhost:16686",
        otlp_endpoint: str = "http://localhost:4317"
    ):
        """
        初始化客户端
        
        Args:
            vllm_base_url: vLLM服务的基础URL
            jaeger_base_url: Jaeger UI的基础URL
            otlp_endpoint: OpenTelemetry Collector的OTLP端点
        """
        self.vllm_base_url = vllm_base_url
        self.jaeger_base_url = jaeger_base_url
        self.otlp_endpoint = otlp_endpoint
        self.setup_tracing()
        
    def setup_tracing(self):
        """设置OpenTelemetry追踪"""
        try:
            resource = Resource.create({
                "service.name": "vllm-metrics-client",
                "service.version": "1.0.0"
            })
            
            provider = TracerProvider(resource=resource)
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(span_processor)
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
        except Exception:
            # 如果OpenTelemetry设置失败，使用空的tracer
            self.tracer = None
    
    def _generate_trace_context(self) -> Dict[str, str]:
        """生成唯一的trace context"""
        trace_id = str(uuid.uuid4()).replace('-', '')[:32]
        span_id = str(uuid.uuid4()).replace('-', '')[:16]
        traceparent = f"00-{trace_id}-{span_id}-01"
        
        return {
            "traceparent": traceparent,
            "trace_id": trace_id,
            "span_id": span_id
        }
    
    async def send_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        stream: bool = True,
        **kwargs
    ) -> RequestMetrics:
        """
        发送请求并获取完整的性能指标
        
        Args:
            messages: 对话消息列表
            model: 模型名称，"auto"表示使用服务端默认模型
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: Top-p采样参数
            stream: 是否使用流式响应
            **kwargs: 其他请求参数
            
        Returns:
            RequestMetrics: 包含所有指标的结果对象
        """
        request_id = str(uuid.uuid4())
        trace_context = self._generate_trace_context()
        
        metrics = RequestMetrics(
            request_id=request_id,
            trace_id=trace_context["trace_id"],
            send_time=time.time()
        )
        
        try:
            start_time = time.perf_counter()
            
            # 构建请求数据
            request_data = {
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **kwargs
            }
            
            # 只在非auto模式下设置model
            if model != "auto":
                request_data["model"] = model
            
            # 设置headers
            headers = {
                "Content-Type": "application/json",
                "traceparent": trace_context["traceparent"],
                "X-Request-Id": request_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vllm_base_url}/v1/chat/completions",
                    json=request_data,
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        metrics.error_message = f"HTTP {response.status}: {error_text}"
                        return metrics
                    
                    # 处理响应
                    if stream:
                        await self._process_streaming_response(response, metrics, start_time)
                    else:
                        await self._process_non_streaming_response(response, metrics, start_time)
                    
                    metrics.success = True
                    
        except Exception as e:
            metrics.error_message = str(e)
        
        return metrics
    
    async def _process_streaming_response(self, response, metrics: RequestMetrics, start_time: float):
        """处理流式响应"""
        generated_text = ""
        first_token_time = None
        inter_token_latencies = []
        most_recent_timestamp = start_time
        
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    if data["choices"][0]["delta"].get("content"):
                        current_time = time.perf_counter()
                        
                        if first_token_time is None:
                            first_token_time = current_time
                            metrics.client_ttft = (current_time - start_time) * 1000
                        else:
                            itl = current_time - most_recent_timestamp
                            inter_token_latencies.append(itl)
                        
                        generated_text += data["choices"][0]["delta"]["content"]
                        most_recent_timestamp = current_time
                        
                except json.JSONDecodeError:
                    continue
        
        end_time = time.perf_counter()
        
        # 计算指标
        metrics.client_e2e_latency = (end_time - start_time) * 1000
        metrics.generated_text = generated_text
        metrics.client_itl_list = inter_token_latencies
        
        if inter_token_latencies:
            metrics.client_tpot = (sum(inter_token_latencies) / len(inter_token_latencies)) * 1000
            metrics.client_itl = metrics.client_tpot
    
    async def _process_non_streaming_response(self, response, metrics: RequestMetrics, start_time: float):
        """处理非流式响应"""
        response_data = await response.json()
        end_time = time.perf_counter()
        
        metrics.client_e2e_latency = (end_time - start_time) * 1000
        
        if "choices" in response_data and response_data["choices"]:
            metrics.generated_text = response_data["choices"][0]["message"]["content"]
    
    def get_server_metrics(self, trace_id: str, timeout: int = 10) -> bool:
        """
        从OpenTelemetry获取服务端指标
        
        Args:
            trace_id: 追踪ID
            timeout: 超时时间(秒)
            
        Returns:
            bool: 是否成功获取指标
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.jaeger_base_url}/api/traces/{trace_id}")
                if response.status_code == 200:
                    trace_data = response.json()
                    return self._extract_server_metrics_from_trace(trace_data)
            except Exception:
                pass
            
            time.sleep(0.5)
        
        return False
    
    def _extract_server_metrics_from_trace(self, trace_data: Dict) -> Dict[str, Any]:
        """从trace数据中提取服务端指标"""
        try:
            for trace in trace_data.get("data", []):
                for span in trace.get("spans", []):
                    if span.get("operationName") == "llm_request":
                        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
                        return {
                            "queue_time": self._safe_float_ms(tags.get("gen_ai.latency.time_in_queue")),
                            "prefill_time": self._safe_float_ms(tags.get("gen_ai.latency.time_in_model_prefill")),
                            "decode_time": self._safe_float_ms(tags.get("gen_ai.latency.time_in_model_decode")),
                            "inference_time": self._safe_float_ms(tags.get("gen_ai.latency.time_in_model_inference")),
                            "e2e_time": self._safe_float_ms(tags.get("gen_ai.latency.e2e")),
                            "ttft": self._safe_float_ms(tags.get("gen_ai.latency.time_to_first_token")),
                            
                            "prompt_tokens": self._safe_int(tags.get("gen_ai.usage.prompt_tokens")),
                            "completion_tokens": self._safe_int(tags.get("gen_ai.usage.completion_tokens")),
                            "model": tags.get("gen_ai.response.model"),
                            "temperature": self._safe_float(tags.get("gen_ai.request.temperature")),
                            "top_p": self._safe_float(tags.get("gen_ai.request.top_p")),
                            "max_tokens": self._safe_int(tags.get("gen_ai.request.max_tokens")),
                        }
        except Exception:
            pass
        
        return {}
    
    def enrich_with_server_metrics(self, metrics: RequestMetrics, timeout: int = 10) -> bool:
        """
        用服务端指标丰富RequestMetrics对象
        
        Args:
            metrics: 要丰富的指标对象
            timeout: 超时时间(秒)
            
        Returns:
            bool: 是否成功获取并设置服务端指标
        """
        server_data = {}
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.jaeger_base_url}/api/traces/{metrics.trace_id}")
                if response.status_code == 200:
                    trace_data = response.json()
                    server_data = self._extract_server_metrics_from_trace(trace_data)
                    if server_data:
                        break
            except Exception:
                pass
            
            time.sleep(0.5)
        
        if not server_data:
            return False
        
        # 设置服务端指标
        metrics.server_queue_time = server_data.get("queue_time")
        metrics.server_prefill_time = server_data.get("prefill_time")
        metrics.server_decode_time = server_data.get("decode_time")
        metrics.server_inference_time = server_data.get("inference_time")
        metrics.server_e2e_time = server_data.get("e2e_time")
        metrics.server_ttft = server_data.get("ttft")
        
        metrics.prompt_tokens = server_data.get("prompt_tokens")
        metrics.completion_tokens = server_data.get("completion_tokens")
        metrics.model_name = server_data.get("model")
        metrics.temperature = server_data.get("temperature")
        metrics.top_p = server_data.get("top_p")
        metrics.max_tokens = server_data.get("max_tokens")
        
        return True
    
    def _safe_float_ms(self, value) -> Optional[float]:
        """安全转换为浮点数并转换为毫秒"""
        if value is not None:
            try:
                return float(value) * 1000
            except:
                pass
        return None
    
    def _safe_float(self, value) -> Optional[float]:
        """安全转换为浮点数"""
        if value is not None:
            try:
                return float(value)
            except:
                pass
        return None
    
    def _safe_int(self, value) -> Optional[int]:
        """安全转换为整数"""
        if value is not None:
            try:
                return int(value)
            except:
                pass
        return None
    
    def get_jaeger_url(self, trace_id: str) -> str:
        """获取Jaeger trace查看URL"""
        return f"{self.jaeger_base_url}/trace/{trace_id}"

def format_metrics(metrics: RequestMetrics, show_details: bool = True) -> str:
    """
    格式化显示指标
    
    Args:
        metrics: 指标对象
        show_details: 是否显示详细信息
        
    Returns:
        str: 格式化的指标字符串
    """
    if not metrics.success:
        return f"❌ 请求失败: {metrics.error_message}"
    
    lines = [
        "="*80,
        "📊 vLLM请求性能指标",
        "="*80,
        f"🆔 请求ID: {metrics.request_id}",
        f"🔗 Trace ID: {metrics.trace_id}",
        f"✅ 状态: 成功",
    ]
    
    if metrics.model_name:
        lines.append(f"🤖 模型: {metrics.model_name}")
    
    lines.extend([
        "",
        "⏱️  延迟指标 (毫秒):",
        f"  客户端E2E延迟:     {metrics.client_e2e_latency:.2f}ms"
    ])
    
    if metrics.server_e2e_time:
        lines.append(f"  服务端E2E延迟:     {metrics.server_e2e_time:.2f}ms")
    
    lines.append("")
    lines.append("🚀 TTFT (首次Token时间):")
    if metrics.client_ttft:
        lines.append(f"  客户端TTFT:        {metrics.client_ttft:.2f}ms")
    if metrics.server_ttft:
        lines.append(f"  服务端TTFT:        {metrics.server_ttft:.2f}ms")
    
    lines.append("")
    lines.append("⚡ Token生成指标:")
    if metrics.client_tpot:
        lines.append(f"  客户端TPOT:        {metrics.client_tpot:.2f}ms")
    if metrics.client_itl:
        lines.append(f"  ITL (令牌间延迟):   {metrics.client_itl:.2f}ms")
    
    if any([metrics.server_queue_time, metrics.server_prefill_time, 
            metrics.server_decode_time, metrics.server_inference_time]):
        lines.append("")
        lines.append("🖥️  服务端详细指标:")
        if metrics.server_queue_time is not None:
            lines.append(f"  队列等待时间:       {metrics.server_queue_time:.2f}ms")
        if metrics.server_prefill_time:
            lines.append(f"  预填充时间:         {metrics.server_prefill_time:.2f}ms")
        if metrics.server_decode_time:
            lines.append(f"  解码时间:           {metrics.server_decode_time:.2f}ms")
        if metrics.server_inference_time:
            lines.append(f"  推理总时间:         {metrics.server_inference_time:.2f}ms")
        
    
    if metrics.prompt_tokens or metrics.completion_tokens:
        lines.append("")
        lines.append("🔢 Token统计:")
        if metrics.prompt_tokens:
            lines.append(f"  输入tokens:        {metrics.prompt_tokens}")
        if metrics.completion_tokens:
            lines.append(f"  输出tokens:        {metrics.completion_tokens}")
        if metrics.client_itl_list:
            lines.append(f"  ITL数据点:         {len(metrics.client_itl_list)} 个")
    
    if any([metrics.temperature, metrics.top_p, metrics.max_tokens]):
        lines.append("")
        lines.append("⚙️  请求参数:")
        if metrics.temperature is not None:
            lines.append(f"  温度:              {metrics.temperature}")
        if metrics.top_p is not None:
            lines.append(f"  Top-p:             {metrics.top_p}")
        if metrics.max_tokens:
            lines.append(f"  最大tokens:        {metrics.max_tokens}")
    
    if show_details and metrics.generated_text:
        lines.append("")
        content = metrics.generated_text[:200] + "..." if len(metrics.generated_text) > 200 else metrics.generated_text
        lines.append(f"📄 生成内容: {content}")
    
    lines.append("")
    lines.append(f"🔗 Jaeger链接: http://localhost:16686/trace/{metrics.trace_id}")
    lines.append("="*80)
    
    return "\n".join(lines)


@dataclass
class BenchmarkSummary:
    """基准测试汇总统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    benchmark_duration: float = 0.0
    
    # Token统计
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # 吞吐量指标
    request_throughput: float = 0.0
    output_token_throughput: float = 0.0
    total_token_throughput: float = 0.0
    
    # TTFT统计 (毫秒)
    ttft_values: List[float] = field(default_factory=list)
    mean_ttft: float = 0.0
    median_ttft: float = 0.0
    p99_ttft: float = 0.0
    
    # TPOT统计 (毫秒)
    tpot_values: List[float] = field(default_factory=list)
    mean_tpot: float = 0.0
    median_tpot: float = 0.0
    p99_tpot: float = 0.0
    
    # ITL统计 (毫秒)
    itl_values: List[float] = field(default_factory=list)
    mean_itl: float = 0.0
    median_itl: float = 0.0
    p99_itl: float = 0.0
    
    # 队列时间统计 (毫秒)
    queue_time_values: List[float] = field(default_factory=list)
    mean_queue_time: float = 0.0
    median_queue_time: float = 0.0
    p99_queue_time: float = 0.0


def calculate_benchmark_summary(results: List[RequestMetrics], start_time: float, end_time: float) -> BenchmarkSummary:
    """计算基准测试汇总统计"""
    summary = BenchmarkSummary()
    
    # 基本统计
    summary.total_requests = len(results)
    summary.successful_requests = sum(1 for r in results if r.success)
    summary.failed_requests = summary.total_requests - summary.successful_requests
    summary.benchmark_duration = end_time - start_time
    
    # 只统计成功的请求
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        return summary
    
    # Token统计
    summary.total_input_tokens = sum(r.prompt_tokens or 0 for r in successful_results)
    summary.total_output_tokens = sum(r.completion_tokens or 0 for r in successful_results)
    
    # 吞吐量计算
    if summary.benchmark_duration > 0:
        summary.request_throughput = summary.successful_requests / summary.benchmark_duration
        summary.output_token_throughput = summary.total_output_tokens / summary.benchmark_duration
        summary.total_token_throughput = (summary.total_input_tokens + summary.total_output_tokens) / summary.benchmark_duration
    
    # 收集各种延迟数据
    ttft_values = []
    tpot_values = []
    itl_values = []
    queue_time_values = []
    
    for result in successful_results:
        # TTFT (使用客户端或服务端数据)
        if result.client_ttft is not None:
            ttft_values.append(result.client_ttft)
        elif result.server_ttft is not None:
            ttft_values.append(result.server_ttft)
        
        # TPOT
        if result.client_tpot is not None:
            tpot_values.append(result.client_tpot)
        
        # ITL (使用ITL列表中的所有值)
        if result.client_itl_list:
            itl_values.extend([itl * 1000 for itl in result.client_itl_list])  # 转换为毫秒
        
        # 队列时间
        if result.server_queue_time is not None:
            queue_time_values.append(result.server_queue_time)
    
    # 计算统计数据
    def calculate_stats(values):
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values)
        return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 99))
    
    # TTFT统计
    summary.ttft_values = ttft_values
    summary.mean_ttft, summary.median_ttft, summary.p99_ttft = calculate_stats(ttft_values)
    
    # TPOT统计
    summary.tpot_values = tpot_values
    summary.mean_tpot, summary.median_tpot, summary.p99_tpot = calculate_stats(tpot_values)
    
    # ITL统计
    summary.itl_values = itl_values
    summary.mean_itl, summary.median_itl, summary.p99_itl = calculate_stats(itl_values)
    
    # 队列时间统计
    summary.queue_time_values = queue_time_values
    summary.mean_queue_time, summary.median_queue_time, summary.p99_queue_time = calculate_stats(queue_time_values)
    
    return summary


def format_benchmark_summary(summary: BenchmarkSummary) -> str:
    """格式化基准测试汇总报告"""
    lines = [
        "="*50,
        " "*10 + "Serving Benchmark Result" + " "*10,
        "="*50,
        f"Successful requests:                     {summary.successful_requests:<10}",
        f"Failed requests:                         {summary.failed_requests:<10}",
        f"Benchmark duration (s):                  {summary.benchmark_duration:<10.2f}",
        f"Total input tokens:                      {summary.total_input_tokens:<10}",
        f"Total generated tokens:                  {summary.total_output_tokens:<10}",
        f"Request throughput (req/s):              {summary.request_throughput:<10.2f}",
        f"Output token throughput (tok/s):         {summary.output_token_throughput:<10.2f}",
        f"Total Token throughput (tok/s):          {summary.total_token_throughput:<10.2f}",
    ]
    
    # TTFT统计
    if summary.ttft_values:
        lines.extend([
            "-"*15 + "Time to First Token" + "-"*16,
            f"Mean TTFT (ms):                          {summary.mean_ttft:<10.2f}",
            f"Median TTFT (ms):                        {summary.median_ttft:<10.2f}",
            f"P99 TTFT (ms):                           {summary.p99_ttft:<10.2f}",
        ])
    
    # TPOT统计
    if summary.tpot_values:
        lines.extend([
            "-"*5 + "Time per Output Token (excl. 1st token)" + "-"*6,
            f"Mean TPOT (ms):                          {summary.mean_tpot:<10.2f}",
            f"Median TPOT (ms):                        {summary.median_tpot:<10.2f}",
            f"P99 TPOT (ms):                           {summary.p99_tpot:<10.2f}",
        ])
    
    # ITL统计
    if summary.itl_values:
        lines.extend([
            "-"*15 + "Inter-token Latency" + "-"*16,
            f"Mean ITL (ms):                           {summary.mean_itl:<10.2f}",
            f"Median ITL (ms):                         {summary.median_itl:<10.2f}",
            f"P99 ITL (ms):                            {summary.p99_itl:<10.2f}",
        ])
    
    # 队列时间统计
    if summary.queue_time_values:
        lines.extend([
            "-"*18 + "Queue Time" + "-"*19,
            f"Mean Queue Time (ms):                    {summary.mean_queue_time:<10.2f}",
            f"Median Queue Time (ms):                  {summary.median_queue_time:<10.2f}",
            f"P99 Queue Time (ms):                     {summary.p99_queue_time:<10.2f}",
        ])
    
    lines.append("="*50)
    
    return "\n".join(lines)


# =============================================================================
# 数据保存功能
# =============================================================================

def save_request_metrics(metrics: RequestMetrics, filename: str, format: str = "json") -> bool:
    """
    保存单个请求的指标数据到文件
    
    Args:
        metrics: RequestMetrics对象
        filename: 文件名 (不含扩展名)
        format: 文件格式 ("json" 或 "csv")
    
    Returns:
        bool: 保存是否成功
    """
    import json
    import os
    from datetime import datetime
    
    try:
        if format.lower() == "json":
            # JSON格式保存
            filepath = f"{filename}.json"
            data = metrics.to_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 请求数据已保存到: {filepath}")
            return True
            
        elif format.lower() == "csv":
            # CSV格式保存
            import csv
            filepath = f"{filename}.csv"
            
            # 展平数据结构用于CSV
            data = metrics.to_dict()
            flattened_data = {}
            
            # 基础信息
            flattened_data.update({
                'request_id': data['request_id'],
                'trace_id': data['trace_id'],
                'success': data['success'],
                'error_message': data['error_message'],
                'timestamp': data['timestamp'],
                'send_time_iso': data['send_time_iso'],
            })
            
            # 客户端指标
            for key, value in data['client_metrics'].items():
                flattened_data[f'client_{key}'] = value
            
            # 服务端指标
            for key, value in data['server_metrics'].items():
                flattened_data[f'server_{key}'] = value
            
            # Token信息
            for key, value in data['tokens'].items():
                flattened_data[key] = value
            
            # 请求参数
            for key, value in data['request_params'].items():
                flattened_data[f'param_{key}'] = value
            
            # 内容信息
            flattened_data['generated_text'] = data['content']['generated_text']
            flattened_data['text_length'] = data['content']['text_length']
            
            # ITL列表 (转换为字符串)
            itl_list_ms = data['detailed_data']['itl_list_ms']
            flattened_data['itl_list_ms'] = str(itl_list_ms) if itl_list_ms else "[]"
            flattened_data['itl_count'] = len(itl_list_ms) if itl_list_ms else 0
            
            # 写入CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_data.keys())
                writer.writeheader()
                writer.writerow(flattened_data)
            
            print(f"✅ 请求数据已保存到: {filepath}")
            return True
            
        else:
            print(f"❌ 不支持的格式: {format}")
            return False
            
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
        return False


def save_multiple_requests_metrics(metrics_list: List[RequestMetrics], filename: str, format: str = "json") -> bool:
    """
    保存多个请求的指标数据到文件
    
    Args:
        metrics_list: RequestMetrics对象列表
        filename: 文件名 (不含扩展名)
        format: 文件格式 ("json" 或 "csv")
    
    Returns:
        bool: 保存是否成功
    """
    import json
    import csv
    from datetime import datetime
    
    if not metrics_list:
        print("❌ 没有数据需要保存")
        return False
    
    try:
        if format.lower() == "json":
            # JSON格式保存
            filepath = f"{filename}.json"
            data = {
                "metadata": {
                    "total_requests": len(metrics_list),
                    "successful_requests": sum(1 for m in metrics_list if m.success),
                    "failed_requests": sum(1 for m in metrics_list if not m.success),
                    "export_time": datetime.now().isoformat(),
                    "format_version": "1.0"
                },
                "requests": [metrics.to_dict() for metrics in metrics_list]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(metrics_list)} 个请求数据已保存到: {filepath}")
            return True
            
        elif format.lower() == "csv":
            # CSV格式保存
            filepath = f"{filename}.csv"
            
            if not metrics_list:
                print("❌ 没有数据需要保存")
                return False
            
            # 获取第一个请求的数据结构作为参考
            first_data = metrics_list[0].to_dict()
            
            # 定义CSV列
            fieldnames = [
                'request_id', 'trace_id', 'success', 'error_message', 
                'timestamp', 'send_time_iso',
                'client_e2e_latency_ms', 'client_ttft_ms', 'client_tpot_ms', 'client_itl_ms',
                'server_queue_time_ms', 'server_prefill_time_ms', 'server_decode_time_ms',
                'server_inference_time_ms', 'server_e2e_time_ms', 'server_ttft_ms',
                'prompt_tokens', 'completion_tokens',
                'param_model', 'param_temperature', 'param_top_p', 'param_max_tokens',
                'generated_text', 'text_length',
                'itl_list_ms', 'itl_count'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for metrics in metrics_list:
                    data = metrics.to_dict()
                    
                    # 展平数据
                    row = {
                        'request_id': data['request_id'],
                        'trace_id': data['trace_id'],
                        'success': data['success'],
                        'error_message': data['error_message'],
                        'timestamp': data['timestamp'],
                        'send_time_iso': data['send_time_iso'],
                        
                        'client_e2e_latency_ms': data['client_metrics']['e2e_latency_ms'],
                        'client_ttft_ms': data['client_metrics']['ttft_ms'],
                        'client_tpot_ms': data['client_metrics']['tpot_ms'],
                        'client_itl_ms': data['client_metrics']['itl_ms'],
                        
                        'server_queue_time_ms': data['server_metrics']['queue_time_ms'],
                        'server_prefill_time_ms': data['server_metrics']['prefill_time_ms'],
                        'server_decode_time_ms': data['server_metrics']['decode_time_ms'],
                        'server_inference_time_ms': data['server_metrics']['inference_time_ms'],
                        'server_e2e_time_ms': data['server_metrics']['e2e_time_ms'],
                        'server_ttft_ms': data['server_metrics']['ttft_ms'],
                        
                        
                        'prompt_tokens': data['tokens']['prompt_tokens'],
                        'completion_tokens': data['tokens']['completion_tokens'],
                        
                        'param_model': data['request_params']['model'],
                        'param_temperature': data['request_params']['temperature'],
                        'param_top_p': data['request_params']['top_p'],
                        'param_max_tokens': data['request_params']['max_tokens'],
                        
                        'generated_text': data['content']['generated_text'],
                        'text_length': data['content']['text_length'],
                        
                        'itl_list_ms': str(data['detailed_data']['itl_list_ms']),
                        'itl_count': len(data['detailed_data']['itl_list_ms']) if data['detailed_data']['itl_list_ms'] else 0
                    }
                    
                    writer.writerow(row)
            
            print(f"✅ {len(metrics_list)} 个请求数据已保存到: {filepath}")
            return True
            
        else:
            print(f"❌ 不支持的格式: {format}")
            return False
            
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
        return False


def generate_timestamped_filename(prefix: str = "vllm_metrics") -> str:
    """
    生成带时间戳的文件名
    
    Args:
        prefix: 文件名前缀
        
    Returns:
        str: 带时间戳的文件名 (不含扩展名)
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
