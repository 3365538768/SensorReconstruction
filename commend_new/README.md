# 4DGaussians SGE 自动化脚本使用指南

## 📋 概述

本文档包含三个 SGE (Sun Grid Engine) 自动化脚本，用于在 CRC 集群上运行 4DGaussians 完整流水线：

1. **`data_preprocessing.sge.sh`** - 数据预处理（RIFE 插帧）
2. **`train_4dgs.sge.sh`** - 4DGaussians 训练和渲染
3. **`inference_4dgs.sge.sh`** - 模型推理和性能评估

## 🚀 快速开始

### 前提条件

1. **项目结构**：确保项目位于 `/users/$USER/SensorReconstruction/`
2. **Conda 环境**：已创建 `Gaussians4D` 环境
3. **数据准备**：Blender 输出的 `originframe` 文件夹已准备完毕

### 基本使用流程

```bash
# 1. 数据预处理
qsub commend_new/data_preprocessing.sge.sh

# 2. 训练和渲染（等数据预处理完成后）
export ACTION_NAME="your_action_name"
qsub commend_new/train_4dgs.sge.sh

# 3. 推理测试（等训练完成后）
qsub commend_new/inference_4dgs.sge.sh
```

---

## 📋 详细说明

### 1. 数据预处理脚本

**文件名**: `data_preprocessing.sge.sh`

**功能**:

- 自动检测 `originframe` 文件夹中的视角数量
- 动态生成 VIEWS 和 TIME_MAP 配置
- 执行 RIFE 插帧处理
- 进行数据集分割（train/val/test）
- 迁移数据到 `data/dnerf/SPLITS/`

**资源配置**: 8 CPU 核心 + 1 GPU 卡

**使用方法**:

```bash
qsub commend_new/data_preprocessing.sge.sh
```

**输出结果**:

- `data/dnerf/SPLITS/` - 标准化数据集
- `ECCV2022-RIFE/SPLITS` - 符号链接

### 2. 训练脚本

**文件名**: `train_4dgs.sge.sh`

**功能**:

- 4DGaussians 模型训练（20000 iterations）
- 生成渲染结果（train/test/video）
- 导出逐帧 3DGS 模型

**资源配置**: 16 CPU 核心 + 2 GPU 卡

**动作名称配置**:

```bash
# 方法1: 环境变量（推荐）
export ACTION_NAME="walking_01"
qsub commend_new/train_4dgs.sge.sh

# 方法2: 配置文件
mkdir -p config
echo "jumping_02" > config/action_name.txt
qsub commend_new/train_4dgs.sge.sh

# 方法3: 自动生成（包含时间戳）
qsub commend_new/train_4dgs.sge.sh
```

**输出结果**:

- `output/dnerf/{ACTION_NAME}/` - 完整训练结果
- `output/dnerf/{ACTION_NAME}/point_cloud/iteration_20000/` - 训练模型
- `output/dnerf/{ACTION_NAME}/gaussian_pertimestamp/` - 逐帧模型

### 3. 推理脚本

**文件名**: `inference_4dgs.sge.sh`

**功能**:

- 模型性能评估
- 渲染速度测试
- 生成详细推理报告

**资源配置**: 8 CPU 核心 + 1 GPU 卡

**使用方法**:

```bash
# 自动检测最新模型
qsub commend_new/inference_4dgs.sge.sh

# 指定特定模型
export ACTION_NAME="walking_01"
qsub commend_new/inference_4dgs.sge.sh
```

**输出结果**:

- `output/dnerf/{ACTION_NAME}/inference_{timestamp}/` - 推理结果
- `inference_report.md` - 详细推理报告

---

## 🔧 高级配置

### 环境变量

- **`ACTION_NAME`**: 动作名称，用于区分不同实验
- **`CUDA_HOME`**: CUDA 安装路径（自动设置）
- **`OMP_NUM_THREADS`**: OpenMP 线程数（自动设置）

### 命名规范

推荐使用以下动作命名格式：

- `动作类型_编号`: 如 `walking_01`, `jumping_02`
- `场景_动作_版本`: 如 `indoor_dancing_v1`
- `日期_实验`: 如 `20250120_test`

### 资源调整

如需调整计算资源，修改脚本头部的 SGE 参数：

```bash
#$ -pe smp 16        # CPU 核心数
#$ -l gpu_card=2     # GPU 卡数
```

---

## 📊 作业监控

### 查看作业状态

```bash
# 查看作业队列
qstat -u $USER

# 查看特定作业详情
qstat -j <job_id>

# 查看作业历史
qacct -j <job_id>
```

### 取消作业

```bash
qdel <job_id>
```

### 查看输出日志

```bash
# SGE 自动生成的日志文件
ls -la *.o* *.e*

# 查看实时日志
tail -f <script_name>.o<job_id>
```

---

## 🐛 故障排除

### 常见问题

1. **环境错误**

   ```bash
   ❌ 请先激活 Gaussians4D 环境
   ```

   **解决**: 确保 conda 环境正确创建和激活

2. **GPU 不可用**

   ```bash
   ❌ GPU 不可用，请检查 CUDA 环境
   ```

   **解决**: 检查 CUDA 模块加载和环境变量

3. **数据未准备**

   ```bash
   ❌ 未找到 originframe 文件夹
   ```

   **解决**: 确认 Blender 输出数据已正确放置

4. **权限问题**
   ```bash
   ❌ 项目目录不存在
   ```
   **解决**: 检查项目路径和权限设置

### 调试模式

如需调试，可以在交互式 GPU 节点上运行：

```bash
# 申请交互式GPU资源
qrsh -q gpu -l gpu_card=1 -pe smp 8

# 手动执行脚本内容进行调试
cd /users/$USER/SensorReconstruction
conda activate Gaussians4D
# ... 执行具体命令
```

---

## 📈 性能优化建议

### 1. 资源配置优化

- **数据预处理**: 1 GPU 卡足够，主要是 I/O 密集
- **模型训练**: 建议 2 GPU 卡，加速训练过程
- **推理测试**: 1 GPU 卡足够，主要是评估性能

### 2. 并行作业

可以同时运行多个不同动作的训练作业：

```bash
export ACTION_NAME="walking_01" && qsub commend_new/train_4dgs.sge.sh
export ACTION_NAME="jumping_02" && qsub commend_new/train_4dgs.sge.sh
export ACTION_NAME="dancing_03" && qsub commend_new/train_4dgs.sge.sh
```

### 3. 存储管理

定期清理临时文件和旧的实验结果：

```bash
# 清理备份文件
find . -name "*_backup_*" -type d -mtime +7 -exec rm -rf {} \;

# 清理临时推理结果
find output/dnerf/*/inference_* -mtime +7 -exec rm -rf {} \;
```

---

## 📞 技术支持

### 文档位置

- **auto.md**: 交互式流程指南
- **development_record.md**: 项目开发历史
- **objective.md**: 项目目标和技术路线

### 联系方式

如遇到问题，请：

1. 检查 SGE 日志文件 (`*.o*` 和 `*.e*`)
2. 查看项目 `development_record.md` 中的历史解决方案
3. 参考 CRC 集群官方文档

---

## 📚 附录

### SGE 参数说明

| 参数              | 说明                             |
| ----------------- | -------------------------------- |
| `-M $USER@nd.edu` | 邮件通知地址                     |
| `-m abe`          | 邮件通知时机（开始、结束、错误） |
| `-pe smp N`       | 申请 N 个 CPU 核心               |
| `-q gpu`          | 提交到 GPU 队列                  |
| `-l gpu_card=N`   | 申请 N 张 GPU 卡                 |
| `-N job_name`     | 作业名称                         |

### 目录结构

```
/users/$USER/SensorReconstruction/
├── ECCV2022-RIFE/
│   ├── originframe/          # Blender 输出数据
│   ├── morepipeline.py       # RIFE 插帧脚本
│   └── get_together.py       # 数据分割脚本
├── data/dnerf/SPLITS/        # 标准化数据集
├── output/dnerf/             # 训练输出
├── config/                   # 配置文件
└── commend_new/              # SGE 脚本
    ├── data_preprocessing.sge.sh
    ├── train_4dgs.sge.sh
    ├── inference_4dgs.sge.sh
    └── README.md
```

---

_更新时间: 2025-07-20 | 维护者: SensorReconstruction 团队_
