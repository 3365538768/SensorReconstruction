# 🗂️ 训练日志管理系统

本目录包含了 4DGaussians 和笼节点模型的所有训练记录，按类型分类存储。

## 📁 目录结构

```
logs/
├── 4DGaussians/           # 4DGaussians模型训练日志
│   └── [实验名称]/         # 按实验名称分类
│       ├── training_[时间戳].log      # 详细训练日志
│       ├── config_[时间戳].json       # 训练配置记录
│       └── metrics_[时间戳].json      # 性能指标数据
├── cage_model/            # 笼节点模型训练日志
│   └── [实验名称]/         # 按实验名称分类
│       ├── training_[时间戳].log      # 详细训练日志
│       ├── config_[时间戳].json       # 训练配置记录
│       └── metrics_[时间戳].json      # 性能指标数据
├── tensorboard/           # TensorBoard日志备份
│   ├── 4DGaussians/       # 4DGaussians的TensorBoard日志
│   └── cage_model/        # 笼节点模型的TensorBoard日志
└── sge_jobs/              # SGE作业日志备份
    ├── 4DGaussians/       # 4DGaussians SGE作业日志
    │   └── [实验名称]/
    │       ├── sge_output_[时间戳].log    # SGE标准输出
    │       ├── sge_error_[时间戳].log     # SGE错误输出
    │       └── job_summary_[时间戳].txt   # 作业摘要信息
    └── cage_model/        # 笼节点模型SGE作业日志
        └── [实验名称]/
            ├── sge_output_[时间戳].log    # SGE标准输出
            ├── sge_error_[时间戳].log     # SGE错误输出
            └── job_summary_[时间戳].txt   # 作业摘要信息
```

## 🚀 日志系统特性

### ✅ 自动记录

- **训练过程**: 每次 iteration/epoch 的详细统计信息
- **配置信息**: 所有训练参数和超参数
- **性能指标**: Loss、PSNR、训练时间等关键指标
- **系统信息**: GPU 使用、节点信息、环境配置

### ✅ 分类管理

- **按模型类型分类**: 4DGaussians vs 笼节点模型
- **按实验名称分组**: 每个实验独立的日志文件夹
- **按时间戳命名**: 避免文件覆盖，支持多次训练

### ✅ 多格式支持

- **结构化日志**: JSON 格式的配置和指标数据
- **可读日志**: 人类友好的训练过程日志
- **TensorBoard**: 可视化训练过程和指标
- **SGE 日志**: 集群作业的完整执行记录

## 🔍 使用方法

### 查看训练日志

```bash
# 查看最新的4DGaussians训练日志
ls -la logs/4DGaussians/

# 查看特定实验的训练过程
tail -f logs/4DGaussians/[实验名称]/training_*.log

# 查看笼节点模型训练统计
cat logs/cage_model/[实验名称]/metrics_*.json
```

### 分析性能指标

```bash
# 查看训练配置
cat logs/4DGaussians/[实验名称]/config_*.json

# 查看性能统计
python -m json.tool logs/4DGaussians/[实验名称]/metrics_*.json
```

### TensorBoard 可视化

```bash
# 启动TensorBoard查看训练过程
tensorboard --logdir=logs/tensorboard/4DGaussians/[实验名称]

# 比较多个实验
tensorboard --logdir=logs/tensorboard/
```

### SGE 作业信息

```bash
# 查看作业摘要
cat logs/sge_jobs/4DGaussians/[实验名称]/job_summary_*.txt

# 查看作业输出日志
less logs/sge_jobs/4DGaussians/[实验名称]/sge_output_*.log
```

## 📊 日志文件详解

### training\_[时间戳].log

包含详细的训练过程记录：

- 每个 iteration 的 loss、PSNR 统计
- 训练阶段切换信息
- 模型保存通知
- 错误和警告信息

### config\_[时间戳].json

记录完整的训练配置：

```json
{
  "timestamp": "2025-07-30T02:26:39",
  "log_type": "4DGaussians",
  "experiment_name": "dnerf/bending",
  "config": {
    "model_params": {...},
    "optimization_params": {...},
    "pipeline_params": {...}
  }
}
```

### metrics\_[时间戳].json

存储性能指标和统计数据：

```json
{
  "start_time": "2025-07-30T02:26:39",
  "training_stats": {
    "start_info": {...},
    "iterations": [...],
    "epochs": [...],
    "completion_info": {...}
  },
  "performance_metrics": {...}
}
```

### job*summary*[时间戳].txt

SGE 作业的摘要信息：

- 作业 ID 和基本信息
- 训练参数配置
- 执行节点和 GPU 信息
- 输出文件统计
- 成功/失败状态

## ⚙️ 配置选项

### 修改日志级别

在训练脚本中调整日志记录的详细程度：

```python
# 在utils/logging_utils.py中修改
logging.getLogger().setLevel(logging.DEBUG)  # 更详细的日志
logging.getLogger().setLevel(logging.WARNING)  # 只记录重要信息
```

### 自定义日志位置

可以通过环境变量或参数修改日志存储位置：

```bash
export LOG_DIR="/custom/log/path"
python train.py --log_dir /custom/log/path
```

## 🧹 日志维护

### 清理策略

建议定期清理旧的日志文件：

```bash
# 删除30天前的日志文件
find logs/ -name "*.log" -mtime +30 -delete
find logs/ -name "*.json" -mtime +30 -delete

# 保留最近的10个实验日志
ls -t logs/4DGaussians/ | tail -n +11 | xargs rm -rf
```

### 备份策略

重要实验的日志建议定期备份：

```bash
# 压缩并备份重要实验日志
tar -czf important_experiment_logs.tar.gz logs/4DGaussians/important_experiment/
tar -czf cage_model_logs.tar.gz logs/cage_model/
```

## 🔧 故障排除

### 常见问题

1. **日志文件权限错误**

   ```bash
   chmod -R 755 logs/
   ```

2. **磁盘空间不足**

   ```bash
   # 检查日志文件夹大小
   du -sh logs/
   # 清理旧日志
   find logs/ -name "*.log" -mtime +7 -delete
   ```

3. **TensorBoard 无法启动**

   ```bash
   # 检查TensorBoard日志目录
   ls -la logs/tensorboard/
   # 重新生成TensorBoard日志
   tensorboard --logdir=logs/tensorboard/ --reload_interval=1
   ```

4. **SGE 日志备份失败**
   - 检查 SGE 作业是否正确设置了 JOB_ID
   - 确认 SGE 输出文件存在
   - 验证日志目录写权限

## 📈 最佳实践

1. **实验命名规范**

   - 使用描述性的实验名称
   - 包含日期和版本信息
   - 避免特殊字符和空格

2. **日志监控**

   - 定期检查日志文件大小
   - 监控训练过程是否正常记录
   - 及时备份重要实验日志

3. **性能分析**
   - 使用 metrics.json 文件进行性能对比
   - 结合 TensorBoard 进行可视化分析
   - 保存训练配置便于重现实验

---

**📧 联系方式**: 如有日志系统相关问题，请联系 zchen27@nd.edu
