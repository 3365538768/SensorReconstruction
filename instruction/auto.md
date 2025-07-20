# 自动化开发指令文档 - auto.md

## 🚀 完整工作流程指南

### 概述

本文档描述了从 GPU 资源获取到 RIFE 插帧数据处理的完整自动化流程，用于 4DGaussians 项目的数据预处理阶段。

---

## 📋 步骤 1: GPU 资源检查与获取

### 1.1 检查 GPU 资源状态

```bash
# 检查服务器 GPU 可用性
free_gpus.sh @crc_gpu
```

**说明**：查看当前可用的 GPU 节点和空闲 GPU 卡数量

### 1.2 申请 GPU 资源

```bash
# 根据实际资源情况调整 gpu_card 数量（获取最大可用数量）
qrsh -q gpu -l gpu_card=1 -pe smp 8

# 如果有更多 GPU 可用，可以申请更多：
# qrsh -q gpu -l gpu_card=2 -pe smp 8  # 2 张卡
# qrsh -q gpu -l gpu_card=4 -pe smp 8  # 4 张卡
```

**说明**：

- `gpu_card=X`: GPU 卡数量，根据 `free_gpus.sh` 结果设置最大值
- `pe smp 8`: 并行环境，8 个 CPU 核心
- 成功后会分配到 GPU 节点（如 qa-a10-033.crc.nd.edu）

---

## 📋 步骤 2: 环境配置

### 2.1 激活 Conda 环境

```bash
# 激活 Gaussians4D 环境
conda activate Gaussians4D

# 验证环境激活成功
echo "当前环境: $(conda info --envs | grep '*')"
python --version
```

### 2.2 验证 GPU 可用性

```bash
# 检查 CUDA 和 GPU 状态
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

---

## 📋 步骤 3: Blender 数据验证与处理

### 3.1 检查 Blender 输出文件夹

```bash
# 进入 ECCV2022-RIFE 目录
cd ECCV2022-RIFE

# 检查是否存在 Blender 输出文件夹
ls -la | grep "^d"
echo "检查是否有新的 Blender 输出文件夹需要处理..."
```

### 3.2 重命名 Blender 文件夹

```bash
# 假设 Blender 输出文件夹名为 blender_output（实际名称可能不同）
# 将其重命名为 originframe
if [ -d "blender_output" ]; then
    mv blender_output originframe
    echo "已将 Blender 输出文件夹重命名为 originframe"
elif [ -d "originframe" ]; then
    echo "originframe 文件夹已存在"
else
    echo "❌ 错误: 未找到 Blender 输出文件夹"
    exit 1
fi
```

### 3.3 验证文件夹命名规范

```bash
# 检查 originframe 中的子文件夹命名
cd originframe
ls -1 | sort

# 验证文件夹是否按大写字母顺序命名 (A, B, C, D, ...)
echo "检查文件夹命名是否符合规范 (A, B, C, D...)："
expected_folders=("A" "B" "C" "D" "E" "F" "G" "H")
actual_folders=($(ls -1 | sort))

for i in "${!actual_folders[@]}"; do
    if [ "${actual_folders[$i]}" != "${expected_folders[$i]}" ]; then
        echo "❌ 文件夹命名不规范: 期望 ${expected_folders[$i]}, 实际 ${actual_folders[$i]}"
        echo "请手动重命名文件夹为大写字母顺序"
        exit 1
    fi
done

echo "✅ 文件夹命名符合规范"
folder_count=${#actual_folders[@]}
echo "检测到 $folder_count 个视角文件夹: ${actual_folders[*]}"

cd ..
```

---

## 📋 步骤 4: 配置 morepipeline.py

### 4.1 检查当前 VIEWS 和 TIME_MAP 配置

```bash
# 显示当前 morepipeline.py 中的配置
echo "当前 morepipeline.py 配置："
grep -n "VIEWS\s*=" morepipeline.py
grep -n "TIME_MAP\s*=" morepipeline.py
```

### 4.2 自动更新配置（根据实际文件夹数量）

```bash
# 获取 originframe 中的文件夹列表
cd originframe
folders=($(ls -1 | sort))
folder_count=${#folders[@]}
cd ..

echo "根据 $folder_count 个文件夹更新 morepipeline.py 配置..."

# 生成 VIEWS 列表
views_array=""
for folder in "${folders[@]}"; do
    views_array+='"'$folder'",'
done
views_array=${views_array%,}  # 移除最后的逗号

# 生成 TIME_MAP
time_map=""
for i in "${!folders[@]}"; do
    if [ $folder_count -eq 1 ]; then
        time_value="1.0"
    else
        time_value=$(echo "scale=1; $i / ($folder_count - 1)" | bc -l)
    fi
    time_map+='"'${folders[$i]}'": '$time_value','
done
time_map=${time_map%,}  # 移除最后的逗号

echo "新的配置："
echo "VIEWS = [$views_array]"
echo "TIME_MAP = {$time_map}"
```

### 4.3 更新 morepipeline.py 文件

```python
# 使用 sed 命令更新配置文件
sed -i "s/VIEWS\s*=.*/VIEWS = [$views_array]/" morepipeline.py
sed -i "s/TIME_MAP\s*=.*/TIME_MAP = {$time_map}/" morepipeline.py

echo "✅ morepipeline.py 配置已更新"

# 验证更新结果
echo "更新后的配置："
grep -n "VIEWS\s*=" morepipeline.py
grep -n "TIME_MAP\s*=" morepipeline.py
```

---

## 📋 步骤 5: 运行 RIFE 插帧

### 5.1 执行 morepipeline.py

```bash
echo "开始执行 RIFE 插帧..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python morepipeline.py

if [ $? -eq 0 ]; then
    echo "✅ morepipeline.py 执行成功"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ morepipeline.py 执行失败"
    exit 1
fi
```

### 5.2 验证插帧结果

```bash
# 检查生成的 FINAL 目录
if [ -d "FINAL" ]; then
    echo "✅ FINAL 目录已生成"
    echo "FINAL 目录内容："
    ls -la FINAL/

    # 统计生成的文件数量
    file_count=$(find FINAL -name "*.png" | wc -l)
    echo "生成的图像文件数量: $file_count"
else
    echo "❌ 错误: 未找到 FINAL 目录"
    exit 1
fi
```

---

## 📋 步骤 6: 数据集分割

### 6.1 执行 get_together.py

```bash
echo "开始执行数据集分割..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python get_together.py

if [ $? -eq 0 ]; then
    echo "✅ get_together.py 执行成功"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ get_together.py 执行失败"
    exit 1
fi
```

### 6.2 验证分割结果

```bash
# 检查生成的 SPLITS 目录
if [ -d "SPLITS" ]; then
    echo "✅ SPLITS 目录已生成"
    echo "SPLITS 目录结构："
    ls -la SPLITS/

    # 统计各数据集的图像数量
    train_count=$(find SPLITS/train -name "*.png" 2>/dev/null | wc -l)
    val_count=$(find SPLITS/val -name "*.png" 2>/dev/null | wc -l)
    test_count=$(find SPLITS/test -name "*.png" 2>/dev/null | wc -l)

    echo "数据集分割统计："
    echo "  训练集 (train): $train_count 张图像"
    echo "  验证集 (val): $val_count 张图像"
    echo "  测试集 (test): $test_count 张图像"
    echo "  总计: $((train_count + val_count + test_count)) 张图像"

    # 检查 JSON 文件
    echo "JSON 文件:"
    ls -la SPLITS/*.json
else
    echo "❌ 错误: 未找到 SPLITS 目录"
    exit 1
fi
```

---

## 📋 步骤 7: 数据迁移

### 7.1 准备目标目录

```bash
# 回到主项目目录
cd ..

# 检查目标目录 /data/dnerf
if [ ! -d "data" ]; then
    mkdir -p data
    echo "已创建 data 目录"
fi

if [ ! -d "data/dnerf" ]; then
    mkdir -p data/dnerf
    echo "已创建 data/dnerf 目录"
fi
```

### 7.2 移动 SPLITS 文件夹

```bash
# 移动 SPLITS 文件夹到目标位置
if [ -d "ECCV2022-RIFE/SPLITS" ]; then
    echo "正在移动 SPLITS 文件夹到 data/dnerf/..."

    # 如果目标位置已有 SPLITS，先备份
    if [ -d "data/dnerf/SPLITS" ]; then
        backup_name="SPLITS_backup_$(date '+%Y%m%d_%H%M%S')"
        mv data/dnerf/SPLITS data/dnerf/$backup_name
        echo "已将原有 SPLITS 备份为 $backup_name"
    fi

    # 移动新的 SPLITS
    mv ECCV2022-RIFE/SPLITS data/dnerf/
    echo "✅ SPLITS 文件夹已成功移动到 data/dnerf/"

    # 创建符号链接供 VSCode 访问
    cd ECCV2022-RIFE
    ln -sf ../data/dnerf/SPLITS SPLITS
    echo "✅ 已创建符号链接 ECCV2022-RIFE/SPLITS -> ../data/dnerf/SPLITS"
    cd ..
else
    echo "❌ 错误: 未找到 ECCV2022-RIFE/SPLITS 目录"
    exit 1
fi
```

### 7.3 验证最终结果

```bash
echo "=== 最终验证 ==="
echo "目标目录结构："
ls -la data/dnerf/SPLITS/

echo "符号链接验证："
ls -la ECCV2022-RIFE/SPLITS

echo "数据完整性检查："
train_count=$(find data/dnerf/SPLITS/train -name "*.png" 2>/dev/null | wc -l)
val_count=$(find data/dnerf/SPLITS/val -name "*.png" 2>/dev/null | wc -l)
test_count=$(find data/dnerf/SPLITS/test -name "*.png" 2>/dev/null | wc -l)

echo "最终数据统计："
echo "  训练集: $train_count 张图像"
echo "  验证集: $val_count 张图像"
echo "  测试集: $test_count 张图像"
echo "  总计: $((train_count + val_count + test_count)) 张图像"

if [ $((train_count + val_count + test_count)) -gt 0 ]; then
    echo "✅ 数据处理流程全部完成！"
    echo "可以开始 4DGaussians 训练了"
else
    echo "❌ 错误: 数据处理未成功完成"
    exit 1
fi
```

---

## 🔧 完整自动化脚本

### 创建一键执行脚本

```bash
#!/bin/bash
# auto_pipeline.sh - 完整自动化流程脚本

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 数据处理自动化流程 ==="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 步骤 1: 检查环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Gaussians4D" ]; then
    echo "❌ 请先激活 Gaussians4D 环境"
    echo "运行: conda activate Gaussians4D"
    exit 1
fi

# 步骤 2: 检查 GPU
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ GPU 不可用，请检查 CUDA 环境"
    exit 1
fi

echo "✅ 环境检查完成"

# 步骤 3-7: 执行主流程
cd ECCV2022-RIFE

# ... (此处包含上述所有步骤的代码)

echo "=== 流程完成 ==="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
```

---

## 📝 注意事项

### 环境要求

- 必须在 GPU 节点上运行
- 需要激活 Gaussians4D conda 环境
- 确保有足够的存储空间（至少 10GB）

### 错误处理

- 每步都有验证机制，出错时立即停止
- 自动备份重要数据
- 详细的错误提示和解决建议

### 性能优化

- 支持多 GPU 加速（根据资源情况调整）
- 自动检测可用资源
- 并行处理能力

### 可定制性

- VIEWS 和 TIME_MAP 根据实际数据自动配置
- 支持不同数量的视角
- 灵活的文件夹结构适配

---

## 🎯 预期结果

完成本流程后，将获得：

1. **标准化数据集**: train/val/test 三个数据集，符合 NeRF 训练格式
2. **高质量插帧**: 基于 RIFE 的时序插值结果
3. **完整元数据**: transforms\_\*.json 文件包含完整的相机参数
4. **组织良好的文件结构**: 便于后续 4DGaussians 训练使用

---

## 📋 步骤 8: 4DGaussians 训练

### 8.1 获取用户输入的动作名称

```bash
# 提示用户输入动作名称和编号
echo "请输入动作名称和编号（例如：walking_01, jumping_02, dancing_03 等）："
read -p "动作名称+编号: " action_name

# 验证输入不为空
if [ -z "$action_name" ]; then
    echo "❌ 错误: 动作名称不能为空"
    exit 1
fi

echo "✅ 设置动作名称为: $action_name"
```

### 8.2 执行 4DGaussians 训练

```bash
echo "开始 4DGaussians 训练..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 执行训练命令
python train.py \
    -s data/dnerf/SPLITS \
    --port 6017 \
    --expname "dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

if [ $? -eq 0 ]; then
    echo "✅ 4DGaussians 训练完成"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "训练结果保存在: output/dnerf/$action_name"
else
    echo "❌ 训练失败"
    exit 1
fi
```

### 8.3 验证训练结果

```bash
# 检查训练输出目录
if [ -d "output/dnerf/$action_name" ]; then
    echo "✅ 训练输出目录已生成"
    echo "训练结果目录结构："
    ls -la "output/dnerf/$action_name/"

    # 检查关键文件
    if [ -f "output/dnerf/$action_name/point_cloud/iteration_20000/point_cloud.ply" ]; then
        echo "✅ 高斯点云模型文件存在"
    else
        echo "⚠️  警告: 高斯点云模型文件未找到"
    fi

    # 检查配置文件
    if [ -f "output/dnerf/$action_name/cfg_args" ]; then
        echo "✅ 训练配置文件存在"
    fi
else
    echo "❌ 错误: 训练输出目录未生成"
    exit 1
fi
```

---

## 📋 步骤 9: 渲染结果生成

### 9.1 执行渲染

```bash
echo "开始渲染训练结果..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 执行渲染命令
python render.py \
    --model_path "output/dnerf/$action_name" \
    --configs arguments/dnerf/jumpingjacks.py

if [ $? -eq 0 ]; then
    echo "✅ 渲染完成"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ 渲染失败"
    exit 1
fi
```

### 9.2 验证渲染结果

```bash
# 检查渲染输出
echo "检查渲染结果..."

# 训练集渲染
if [ -d "output/dnerf/$action_name/train/ours_20000/renders" ]; then
    train_renders=$(find "output/dnerf/$action_name/train/ours_20000/renders" -name "*.png" | wc -l)
    echo "✅ 训练集渲染图像: $train_renders 张"
else
    echo "⚠️  警告: 训练集渲染结果未找到"
fi

# 测试集渲染
if [ -d "output/dnerf/$action_name/test/ours_20000/renders" ]; then
    test_renders=$(find "output/dnerf/$action_name/test/ours_20000/renders" -name "*.png" | wc -l)
    echo "✅ 测试集渲染图像: $test_renders 张"
else
    echo "⚠️  警告: 测试集渲染结果未找到"
fi

# 视频渲染
if [ -d "output/dnerf/$action_name/video/ours_20000/renders" ]; then
    video_renders=$(find "output/dnerf/$action_name/video/ours_20000/renders" -name "*.png" | wc -l)
    echo "✅ 视频渲染帧数: $video_renders 张"
else
    echo "⚠️  警告: 视频渲染结果未找到"
fi

echo "渲染结果详情："
ls -la "output/dnerf/$action_name/"
```

---

## 📋 步骤 10: 导出逐帧 3DGS 模型

### 10.1 执行模型导出

```bash
echo "开始导出逐帧 3DGS 模型..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 执行导出命令
python export_perframe_3DGS.py \
    --iteration 20000 \
    --configs arguments/dnerf/jumpingjacks.py \
    --model_path "output/dnerf/$action_name"

if [ $? -eq 0 ]; then
    echo "✅ 逐帧 3DGS 模型导出完成"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ 模型导出失败"
    exit 1
fi
```

### 10.2 验证导出结果

```bash
# 检查 gaussian_pertimestamp 文件夹
if [ -d "output/dnerf/$action_name/gaussian_pertimestamp" ]; then
    echo "✅ gaussian_pertimestamp 文件夹已生成"

    # 统计导出的模型文件
    ply_count=$(find "output/dnerf/$action_name/gaussian_pertimestamp" -name "*.ply" | wc -l)
    echo "导出的 .ply 模型文件数量: $ply_count"

    # 显示文件夹内容
    echo "gaussian_pertimestamp 文件夹内容："
    ls -la "output/dnerf/$action_name/gaussian_pertimestamp/"

    # 检查文件大小
    echo "模型文件大小统计："
    du -sh "output/dnerf/$action_name/gaussian_pertimestamp/"
else
    echo "❌ 错误: gaussian_pertimestamp 文件夹未生成"
    exit 1
fi
```

---

## 📋 步骤 11: 最终验证与总结

### 11.1 完整性检查

```bash
echo "=== 完整流程验证 ==="
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 数据预处理验证
if [ -d "data/dnerf/SPLITS" ]; then
    echo "✅ 数据预处理: SPLITS 数据集存在"
else
    echo "❌ 数据预处理: SPLITS 数据集缺失"
fi

# 训练结果验证
if [ -d "output/dnerf/$action_name" ]; then
    echo "✅ 训练结果: 模型输出目录存在"
else
    echo "❌ 训练结果: 模型输出目录缺失"
fi

# 渲染结果验证
render_dirs=("train/ours_20000/renders" "test/ours_20000/renders" "video/ours_20000/renders")
for dir in "${render_dirs[@]}"; do
    if [ -d "output/dnerf/$action_name/$dir" ]; then
        echo "✅ 渲染结果: $dir 存在"
    else
        echo "⚠️  渲染结果: $dir 缺失"
    fi
done

# 导出结果验证
if [ -d "output/dnerf/$action_name/gaussian_pertimestamp" ]; then
    echo "✅ 模型导出: gaussian_pertimestamp 文件夹存在"
else
    echo "❌ 模型导出: gaussian_pertimestamp 文件夹缺失"
fi
```

### 11.2 性能统计

```bash
echo "=== 性能统计 ==="

# 数据集规模
if [ -d "data/dnerf/SPLITS" ]; then
    train_images=$(find data/dnerf/SPLITS/train -name "*.png" 2>/dev/null | wc -l)
    val_images=$(find data/dnerf/SPLITS/val -name "*.png" 2>/dev/null | wc -l)
    test_images=$(find data/dnerf/SPLITS/test -name "*.png" 2>/dev/null | wc -l)
    echo "数据集规模: 训练($train_images) + 验证($val_images) + 测试($test_images) = $((train_images + val_images + test_images)) 张图像"
fi

# 模型大小
if [ -f "output/dnerf/$action_name/point_cloud/iteration_20000/point_cloud.ply" ]; then
    model_size=$(du -h "output/dnerf/$action_name/point_cloud/iteration_20000/point_cloud.ply" | cut -f1)
    echo "主模型大小: $model_size"
fi

# 导出模型统计
if [ -d "output/dnerf/$action_name/gaussian_pertimestamp" ]; then
    export_size=$(du -sh "output/dnerf/$action_name/gaussian_pertimestamp/" | cut -f1)
    export_count=$(find "output/dnerf/$action_name/gaussian_pertimestamp" -name "*.ply" | wc -l)
    echo "导出模型: $export_count 个文件，总大小 $export_size"
fi

# 存储空间使用
total_size=$(du -sh "output/dnerf/$action_name/" | cut -f1)
echo "项目总存储: $total_size"
```

### 11.3 最终结果位置

```bash
echo "=== 结果文件位置 ==="
echo "📁 数据集位置: data/dnerf/SPLITS/"
echo "📁 训练结果位置: output/dnerf/$action_name/"
echo "📁 渲染结果位置: output/dnerf/$action_name/{train,test,video}/ours_20000/renders/"
echo "📁 逐帧模型位置: output/dnerf/$action_name/gaussian_pertimestamp/"
echo ""
echo "🎯 主要输出文件："
echo "  - 高斯点云模型: output/dnerf/$action_name/point_cloud/iteration_20000/point_cloud.ply"
echo "  - 逐帧 3DGS 模型: output/dnerf/$action_name/gaussian_pertimestamp/*.ply"
echo "  - 渲染图像: output/dnerf/$action_name/*/ours_20000/renders/*.png"
echo ""
echo "✅ 4DGaussians 完整流程执行完毕！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
```

---

## 🔧 更新的完整自动化脚本

### 创建包含训练流程的一键执行脚本

```bash
#!/bin/bash
# complete_pipeline.sh - 从数据预处理到模型导出的完整流程

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 完整自动化流程 ==="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 环境检查
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Gaussians4D" ]; then
    echo "❌ 请先激活 Gaussians4D 环境"
    echo "运行: conda activate Gaussians4D"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ GPU 不可用，请检查 CUDA 环境"
    exit 1
fi

# 获取用户输入
echo "请输入动作名称和编号（例如：walking_01, jumping_02, dancing_03 等）："
read -p "动作名称+编号: " action_name

if [ -z "$action_name" ]; then
    echo "❌ 错误: 动作名称不能为空"
    exit 1
fi

echo "✅ 设置动作名称为: $action_name"

# 执行数据预处理流程（步骤 1-7）
# ... (包含之前的所有数据处理步骤)

# 执行训练和渲染流程（步骤 8-11）
echo "开始训练阶段..."

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

echo "=== 完整流程执行完毕 ==="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果位置: output/dnerf/$action_name/"
```

---

## 📝 新增注意事项

### 训练阶段要求

- **GPU 内存**: 训练需要至少 12GB VRAM（推荐 24GB+）
- **训练时间**: 完整训练约需 1-3 小时（取决于数据规模和 GPU 性能）
- **存储空间**: 训练输出约需 5-10GB 存储空间
- **端口使用**: 默认使用端口 6017，确保端口未被占用

### 渲染阶段要求

- **内存需求**: 渲染过程需要额外的 GPU 内存
- **渲染时间**: 渲染时间与图像数量成正比（约 1-2 分钟/百张图像）
- **输出格式**: 生成 PNG 格式的高质量渲染图像

### 导出阶段要求

- **模型大小**: 逐帧模型文件较大，确保有足够存储空间
- **导出时间**: 导出时间取决于时间步数（约 10-30 秒/帧）
- **文件格式**: 生成标准 PLY 格式的 3D 高斯模型文件

### 用户交互优化

- **动作命名规范**: 建议使用 "动作类型\_编号" 格式（如 walking_01）
- **避免特殊字符**: 动作名称中避免使用空格和特殊符号
- **版本管理**: 不同实验使用不同编号便于管理

---

_最后更新: 2025-07-20 00:13:18 | 维护者: zchen27@nd.edu_
_基于 ECCV2022-RIFE 和 4DGaussians 项目优化_
