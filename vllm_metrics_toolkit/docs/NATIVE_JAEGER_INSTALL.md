# 原生 Jaeger 安装指南

本指南介绍如何在不使用 Docker 容器的情况下安装和运行 Jaeger，用于 vLLM 性能指标追踪。

## 📋 系统要求

- Linux (推荐 Ubuntu 18.04+)
- 至少 512MB 可用内存
- 至少 1GB 可用磁盘空间

## 🚀 快速安装 (推荐)

### 1. 下载最新版本的 Jaeger

```bash
# 创建安装目录
mkdir -p ~/jaeger
cd ~/jaeger

# 获取最新版本号 (或直接使用固定版本)
JAEGER_VERSION="1.52.0"

# 下载 Jaeger 二进制文件
wget https://github.com/jaegertracing/jaeger/releases/download/v${JAEGER_VERSION}/jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz

# 解压文件
tar -xzf jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz

# 进入解压目录
cd jaeger-${JAEGER_VERSION}-linux-amd64
```

### 2. 启动 Jaeger (All-in-One 模式)

```bash
# 基本启动 (内存存储)
./jaeger-all-in-one --collector.otlp.enabled=true

# 或者指定日志级别
./jaeger-all-in-one --collector.otlp.enabled=true --log-level=info
```

### 3. 验证安装

打开浏览器访问：`http://localhost:16686`

您应该能看到 Jaeger UI 界面。

## 🔧 详细配置

### 启动参数说明

```bash
./jaeger-all-in-one \
  --collector.otlp.enabled=true \          # 启用 OTLP 接收器 (重要!)
  --collector.otlp.grpc.host-port=0.0.0.0:4317 \  # OTLP gRPC 端口
  --collector.otlp.http.host-port=0.0.0.0:4318 \  # OTLP HTTP 端口
  --query.host-port=0.0.0.0:16686 \        # UI 端口
  --admin.http.host-port=0.0.0.0:14269 \   # 管理端口
  --log-level=info                          # 日志级别
```

### 端口说明

| 端口 | 用途 | 说明 |
|------|------|------|
| 16686 | Jaeger UI | Web 界面 |
| 4317 | OTLP gRPC | vLLM 发送追踪数据的端口 |
| 4318 | OTLP HTTP | 备用 HTTP 端口 |
| 14269 | Admin | 健康检查和指标 |

## 💾 数据持久化配置

### 方案1: Badger 本地存储 (推荐用于开发)

```bash
# 创建数据目录
mkdir -p ~/jaeger/data

# 启动带本地存储的 Jaeger
./jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --span-storage-type=badger \
  --badger.directory-key=~/jaeger/data \
  --badger.directory-value=~/jaeger/data
```

### 方案2: 文件存储 (简单但有限制)

```bash
./jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --span-storage-type=memory \
  --memory.max-traces=50000
```

## 🔄 创建系统服务 (可选)

### 1. 创建服务文件

```bash
sudo tee /etc/systemd/system/jaeger.service > /dev/null <<EOF
[Unit]
Description=Jaeger All-in-One
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/jaeger/jaeger-1.52.0-linux-amd64
ExecStart=$HOME/jaeger/jaeger-1.52.0-linux-amd64/jaeger-all-in-one --collector.otlp.enabled=true --span-storage-type=badger --badger.directory-key=$HOME/jaeger/data --badger.directory-value=$HOME/jaeger/data --log-level=info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### 2. 启用和启动服务

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启用服务 (开机自启)
sudo systemctl enable jaeger

# 启动服务
sudo systemctl start jaeger

# 检查状态
sudo systemctl status jaeger
```

## 🧪 测试安装

### 1. 检查服务状态

```bash
# 检查端口是否监听
netstat -tlnp | grep -E "(16686|4317|4318)"

# 或使用 ss
ss -tlnp | grep -E "(16686|4317|4318)"
```

### 2. 健康检查

```bash
# 检查 Jaeger 健康状态
curl http://localhost:14269/health

# 检查 OTLP 接收器
curl http://localhost:14269/metrics | grep otlp
```

### 3. 使用 vLLM 工具包测试

```bash
cd /home/ubuntu/vllm/vllm_metrics_toolkit
source ../.venv/bin/activate

# 快速测试 (确保 vLLM 服务在运行)
python scripts/benchmark_sharegpt.py --rps 1 --num_prompts 5 --max_tokens 20
```

## 🔧 与 vLLM 集成

### 1. 启动 vLLM 服务

```bash
# 启动支持 OpenTelemetry 的 vLLM
vllm serve <your-model> \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces all \
  --port 8000
```

### 2. 配置工具包

不需要修改任何代码，工具包默认配置就指向本地 Jaeger：

```python
# 默认配置已经正确
client = VLLMMetricsClient(
    otlp_endpoint="http://localhost:4317"  # 这是默认值
)
```

## 🐛 故障排查

### 常见问题

1. **端口被占用**
   ```bash
   # 查看占用端口的进程
   sudo lsof -i :4317
   sudo lsof -i :16686
   ```

2. **权限问题**
   ```bash
   # 确保用户有执行权限
   chmod +x ~/jaeger/jaeger-*/jaeger-all-in-one
   ```

3. **防火墙问题**
   ```bash
   # Ubuntu UFW
   sudo ufw allow 16686
   sudo ufw allow 4317
   
   # CentOS/RHEL firewalld
   sudo firewall-cmd --permanent --add-port=16686/tcp
   sudo firewall-cmd --permanent --add-port=4317/tcp
   sudo firewall-cmd --reload
   ```

### 日志查看

```bash
# 如果使用 systemd 服务
sudo journalctl -u jaeger -f

# 或直接运行时的输出
./jaeger-all-in-one --collector.otlp.enabled=true --log-level=debug
```

## 🚀 性能优化

### 生产环境建议

```bash
./jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --span-storage-type=badger \
  --badger.directory-key=/var/lib/jaeger/data \
  --badger.directory-value=/var/lib/jaeger/data \
  --collector.queue-size=5000 \
  --collector.num-workers=100 \
  --query.max-clock-skew-adjustment=0s \
  --log-level=warn
```

## 📊 监控和维护

### 查看存储使用情况

```bash
# 查看数据目录大小
du -sh ~/jaeger/data

# 清理旧数据 (如果使用 Badger)
# Badger 会自动压缩，但可以手动清理
```

### 备份数据

```bash
# 备份 Badger 数据库
tar -czf jaeger-backup-$(date +%Y%m%d).tar.gz ~/jaeger/data
```

---

**🎉 完成！** 

现在您已经有了一个完全原生的 Jaeger 安装，可以与 vLLM 指标工具包完美配合使用，无需任何容器依赖。
