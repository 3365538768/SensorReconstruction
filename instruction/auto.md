# 4DGaussians 自动化流程指南 - auto.md

## 🚀 一键执行脚本

```bash
#!/bin/bash
# complete_4dgs_pipeline.sh - 完整自动化流程

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 完整自动化流程 ==="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 1. 环境检查
echo "检查运行环境..."
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Gaussians4D" ]; then
    echo "❌ 请先激活 Gaussians4D 环境: conda activate Gaussians4D"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ GPU 不可用，请检查 CUDA 环境"
    exit 1
fi

echo "✅ 环境检查通过"

# 2. 获取用户输入
echo "请输入动作名称和编号（例如：walking_01, jumping_02）："
read -p "动作名称+编号: " action_name
if [ -z "$action_name" ]; then
    echo "❌ 动作名称不能为空"
    exit 1
fi

# 3. 数据预处理流程
echo "执行数据预处理..."
cd ECCV2022-RIFE

# 检查并处理 originframe
if [ ! -d "originframe" ]; then
    echo "❌ 未找到 originframe 文件夹，请确认 Blender 输出已准备"
    exit 1
fi

# 获取文件夹数量并更新配置
cd originframe
folders=($(ls -1 | sort))
folder_count=${#folders[@]}
cd ..

# 生成 VIEWS 和 TIME_MAP
views_array=""
time_map=""
for i in "${!folders[@]}"; do
    views_array+='"'${folders[$i]}'"'
    if [ $i -lt $((folder_count-1)) ]; then views_array+=","; fi

    if [ $folder_count -eq 1 ]; then
        time_value="1.0"
    else
        time_value=$(echo "scale=1; $i / ($folder_count - 1)" | bc -l)
    fi
    time_map+='"'${folders[$i]}'": '$time_value
    if [ $i -lt $((folder_count-1)) ]; then time_map+=","; fi
done

# 更新 morepipeline.py 配置
sed -i "s/VIEWS\s*=.*/VIEWS = [$views_array]/" morepipeline.py
sed -i "s/TIME_MAP\s*=.*/TIME_MAP = {$time_map}/" morepipeline.py

echo "✅ 配置已更新: $folder_count 个视角"

# 4. 执行 RIFE 插帧
echo "执行 RIFE 插帧..."
python morepipeline.py

# 5. 数据集分割
echo "执行数据集分割..."
python get_together.py

# 6. 数据迁移
echo "迁移数据到项目目录..."
cd ..
mkdir -p data/dnerf
if [ -d "data/dnerf/SPLITS" ]; then
    mv data/dnerf/SPLITS data/dnerf/SPLITS_backup_$(date '+%Y%m%d_%H%M%S')
fi
mv ECCV2022-RIFE/SPLITS data/dnerf/
cd ECCV2022-RIFE && ln -sf ../data/dnerf/SPLITS SPLITS && cd ..

echo "✅ 数据预处理完成"

# 7. 4DGaussians 训练
echo "开始 4DGaussians 训练..."
python train.py \
    -s data/dnerf/SPLITS \
    --port 6017 \
    --expname "dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

echo "✅ 训练完成"

# 8. 渲染
echo "生成渲染结果..."
python render.py \
    --model_path "output/dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

echo "✅ 渲染完成"

# 9. 导出逐帧模型
echo "导出逐帧 3DGS 模型..."
python export_perframe_3DGS.py \
    --iteration 20000 \
    --configs arguments/dnerf/jumpingjacks.py \
    --model_path "output/dnerf/$action_name"

echo "✅ 模型导出完成"

# 10. 最终统计
echo "=== 流程完成统计 ==="
if [ -d "data/dnerf/SPLITS" ]; then
    train_count=$(find data/dnerf/SPLITS/train -name "*.png" 2>/dev/null | wc -l)
    val_count=$(find data/dnerf/SPLITS/val -name "*.png" 2>/dev/null | wc -l)
    test_count=$(find data/dnerf/SPLITS/test -name "*.png" 2>/dev/null | wc -l)
    echo "数据集: train($train_count) + val($val_count) + test($test_count) = $((train_count + val_count + test_count)) 张图像"
fi

if [ -d "output/dnerf/$action_name/gaussian_pertimestamp" ]; then
    ply_count=$(find "output/dnerf/$action_name/gaussian_pertimestamp" -name "*.ply" | wc -l)
    echo "导出模型: $ply_count 个 PLY 文件"
fi

echo "结果位置: output/dnerf/$action_name/"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "✅ 4DGaussians 完整流程执行完毕！"
```

---

## 📋 分步执行（可选）

如果需要分步执行或调试，可以按以下步骤：

### 步骤 1: 环境准备

```bash
# 检查 GPU 资源
free_gpus.sh @crc_gpu

# 申请 GPU 资源
qrsh -q gpu -l gpu_card=1 -pe smp 8

# 激活环境
conda activate Gaussians4D
```

### 步骤 2: 数据预处理

```bash
cd ECCV2022-RIFE

# 自动配置 VIEWS 和 TIME_MAP
cd originframe && folders=($(ls -1 | sort)) && cd ..
folder_count=${#folders[@]}

# 更新配置文件（根据实际文件夹数量）
# VIEWS = ["A", "B", "C", "D"]  # 示例：4 个视角
# TIME_MAP = {"A": 0.0, "B": 0.3, "C": 0.6, "D": 1.0}

# 执行插帧和分割
python morepipeline.py
python get_together.py

# 迁移数据
cd .. && mkdir -p data/dnerf
mv ECCV2022-RIFE/SPLITS data/dnerf/
cd ECCV2022-RIFE && ln -sf ../data/dnerf/SPLITS SPLITS && cd ..
```

### 步骤 3: 训练和渲染

```bash
# 获取动作名称
read -p "动作名称+编号: " action_name

# 训练
python train.py \
    -s data/dnerf/SPLITS \
    --port 6017 \
    --expname "dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

# 渲染
python render.py \
    --model_path "output/dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

# 导出
python export_perframe_3DGS.py \
    --iteration 20000 \
    --configs arguments/dnerf/jumpingjacks.py \
    --model_path "output/dnerf/$action_name"
```

---

## 📝 重要说明

### 环境要求

- **GPU**: 需要 GPU 节点（推荐 NVIDIA A10/L40S/A100）
- **内存**: 训练需要 12GB+ VRAM
- **存储**: 至少 10GB 可用空间
- **环境**: Gaussians4D conda 环境

### 预期时间

- **数据预处理**: 10-30 分钟（取决于数据量）
- **训练**: 1-3 小时（20000 iterations）
- **渲染**: 10-30 分钟
- **导出**: 5-15 分钟
- **总计**: 约 2-4 小时

### 输出结果

- **数据集**: `data/dnerf/SPLITS/` (train/val/test)
- **训练模型**: `output/dnerf/{action_name}/point_cloud/iteration_20000/`
- **渲染图像**: `output/dnerf/{action_name}/{train,test,video}/ours_20000/renders/`
- **逐帧模型**: `output/dnerf/{action_name}/gaussian_pertimestamp/`

### 故障排除

- **GPU 内存不足**: 减少批处理大小或使用更少视角
- **CUDA 错误**: 检查 CUDA 环境和 PyTorch 版本
- **文件夹命名**: 确保 originframe 中文件夹按 A、B、C、D 顺序命名
- **端口冲突**: 修改 `--port 6017` 为其他可用端口

---

## 🔧 使用方法

### 方法 1: 一键执行（推荐）

```bash
# 下载并执行脚本
wget https://raw.githubusercontent.com/your-repo/complete_4dgs_pipeline.sh
chmod +x complete_4dgs_pipeline.sh
./complete_4dgs_pipeline.sh
```

### 方法 2: 复制粘贴

将上述完整脚本复制到终端中执行

### 方法 3: 分步执行

按照分步执行部分的命令逐步运行

---

_最后更新: 2025-07-20 00:24:52 | 维护者: zchen27@nd.edu_
_优化版本：简化流程，减少中断，提升执行效率_
