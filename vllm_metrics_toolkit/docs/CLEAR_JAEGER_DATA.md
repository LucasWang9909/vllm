# Jaeger 数据清理指南

当您需要重新开始测试或清空历史追踪数据时，可以使用以下方法清理 Jaeger。

## 🚀 快速清理 (推荐)

### 使用自动清理脚本
```bash
cd /home/ubuntu/vllm/vllm_metrics_toolkit
./clear_jaeger_data.sh
```

脚本提供三种清理选项：
1. **清理数据文件** - 删除所有历史追踪数据
2. **清理缓存 + 重启** - 清理数据并自动重启服务
3. **完全重置** - 删除所有数据和配置文件

## 🔧 手动清理方法

### 方法1: Docker 容器清理

如果使用 Docker 方式运行 Jaeger：

```bash
# 停止并删除容器
docker stop jaeger
docker rm jaeger

# 重新启动全新容器
docker run --rm --name jaeger \
  -p 16686:16686 -p 14250:14250 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

**优点**: 最简单快速，数据完全重置
**缺点**: 需要重新下载镜像（如果被删除）

### 方法2: 原生安装数据清理

如果使用原生安装：

#### 2a. 清理 Badger 数据库 (推荐)
```bash
# 停止 Jaeger 服务
pkill -f "jaeger-all-in-one"

# 删除数据目录
rm -rf /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data

# 重新创建数据目录
mkdir -p /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data

# 重启 Jaeger
cd /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger
./start_jaeger.sh
```

#### 2b. 使用内存模式 (临时)
```bash
# 停止当前 Jaeger
pkill -f "jaeger-all-in-one"

# 启动内存模式 (重启后数据自动清空)
cd /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/jaeger-1.52.0-linux-amd64
./jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=0.0.0.0:4317 \
  --collector.otlp.http.host-port=0.0.0.0:4318 \
  --query.http-server.host-port=0.0.0.0:16686 \
  --span-storage-type=memory \
  --memory.max-traces=10000
```

### 方法3: API 清理 (高级)

Jaeger 没有直接的 API 来删除数据，但可以通过以下方式：

```bash
# 重启服务实现"软清理"
curl -X POST http://localhost:14269/admin/shutdown
sleep 2
cd /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger
./start_jaeger.sh
```

## 🎯 针对不同场景的建议

### 场景1: 开发测试阶段
**推荐**: 使用 Docker 方式 + 容器重启
```bash
docker restart jaeger
# 或
docker stop jaeger && docker run --rm --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest
```

### 场景2: 性能基准测试
**推荐**: 使用自动清理脚本
```bash
./clear_jaeger_data.sh
# 选择选项 2 (清理缓存 + 重启)
```

### 场景3: 生产环境监控
**推荐**: 定期备份 + 选择性清理
```bash
# 备份当前数据
tar -czf jaeger-backup-$(date +%Y%m%d).tar.gz /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data

# 清理旧数据
./clear_jaeger_data.sh
```

### 场景4: 紧急数据清理
**推荐**: 强制清理所有进程和数据
```bash
# 强制停止所有 Jaeger 进程
pkill -9 -f jaeger

# 删除所有相关数据
rm -rf /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data
rm -f /tmp/jaeger.pid

# 重新启动
cd /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger
./start_jaeger.sh
```

## 🔍 验证清理结果

### 1. 检查 Jaeger UI
打开 http://localhost:16686，确认：
- 服务列表为空或只显示新的服务
- 没有历史追踪数据
- 时间范围选择器显示最近时间

### 2. 检查数据目录
```bash
# 查看数据目录大小
du -sh /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data

# 列出数据文件
ls -la /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data
```

### 3. 测试新数据收集
```bash
cd /home/ubuntu/vllm/vllm_metrics_toolkit
source ../.venv/bin/activate

# 运行简单测试
python scripts/benchmark_sharegpt.py --rps 1 --num_prompts 2 --max_tokens 20

# 检查是否有新的追踪数据出现在 Jaeger UI
```

## ⚠️ 注意事项

### 数据丢失警告
- **所有清理操作都会永久删除历史数据**
- 清理前请确保已备份重要的测试结果
- 建议在清理前导出重要的追踪数据

### 系统影响
- 清理过程中会短暂中断追踪收集
- 正在运行的 vLLM 测试可能丢失部分追踪数据
- 建议在测试间隙进行清理操作

### 备份建议
```bash
# 自动备份脚本
#!/bin/bash
BACKUP_DIR="/home/ubuntu/jaeger_backups"
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/jaeger-$(date +%Y%m%d_%H%M%S).tar.gz" \
  /home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data
echo "备份完成: $BACKUP_DIR/jaeger-$(date +%Y%m%d_%H%M%S).tar.gz"
```

## 🔄 自动清理 (可选)

### 定时清理脚本
创建定时任务，定期清理旧数据：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨2点清理）
0 2 * * * /home/ubuntu/vllm/vllm_metrics_toolkit/clear_jaeger_data.sh >/dev/null 2>&1
```

### 基于磁盘空间的清理
```bash
#!/bin/bash
DATA_DIR="/home/ubuntu/vllm/vllm_metrics_toolkit/jaeger/data"
MAX_SIZE_MB=1000  # 1GB

CURRENT_SIZE=$(du -sm "$DATA_DIR" | cut -f1)
if [ "$CURRENT_SIZE" -gt "$MAX_SIZE_MB" ]; then
    echo "数据目录超过 ${MAX_SIZE_MB}MB，执行清理..."
    /home/ubuntu/vllm/vllm_metrics_toolkit/clear_jaeger_data.sh
fi
```

---

## 📝 快速参考

| 清理方式 | 命令 | 耗时 | 数据恢复性 |
|---------|------|------|-----------|
| Docker 重启 | `docker restart jaeger` | 5-10秒 | 不可恢复 |
| 自动脚本 | `./clear_jaeger_data.sh` | 10-30秒 | 不可恢复 |
| 手动删除 | `rm -rf jaeger/data` | 1-5秒 | 不可恢复 |
| 内存模式 | 重启到内存模式 | 10秒 | 不可恢复 |

选择最适合您场景的清理方式，开始全新的测试！🎉
