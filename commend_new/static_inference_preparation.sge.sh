#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 8                # 分配 8 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N static_inference_prep # 作业名称

set -e  # 遇到错误立即退出

echo "=== 静态场景推理数据准备与训练作业 ==="
echo "作业开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo "工作节点: $(hostname)"
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

#### ——— 1. 加载模块 ———
module load cmake/3.22.1
module load cuda/11.8
module load intel/24.2

#### ——— 2. 设置环境变量 ———
export OMP_NUM_THREADS=$NSLOTS
export CUDA_HOME=/opt/crc/c/cuda/11.8

#### ——— 3. 激活 Conda 环境 ———
source ~/.bashrc
conda activate Gaussians4D

# 验证环境
echo "Python 环境验证:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

#### ——— 4. 设置工作目录 ———
PROJECT_ROOT="/users/$USER/SensorReconstruction"
RIFE_DIR="$PROJECT_ROOT/ECCV2022-RIFE"

echo "项目根目录: $PROJECT_ROOT"
echo "RIFE 工作目录: $RIFE_DIR"

# 检查项目目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 错误: 项目目录不存在 $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

#### ——— 5. 创建静态场景数据 ———
echo "开始静态场景数据准备..."

# 进入 RIFE 目录
cd "$RIFE_DIR"

# 检查是否存在原始blender输出数据
if [ ! -d "originframe" ]; then
    echo "❌ 错误: 未找到 originframe 文件夹"
    echo "请确认 Blender 输出数据已准备完毕"
    exit 1
fi

echo "✅ 发现 originframe 文件夹"

# 进入originframe检查文件夹A
cd originframe
if [ ! -d "A" ]; then
    echo "❌ 错误: 未找到文件夹A"
    echo "请确认 Blender 输出包含文件夹A"
    exit 1
fi

echo "✅ 发现文件夹A"

# 创建静态场景：复制A为B
echo "创建静态场景: 复制文件夹A为文件夹B..."
if [ -d "B" ]; then
    echo "警告: 文件夹B已存在，将先删除..."
    rm -rf B
fi

cp -r A B
echo "✅ 文件夹A已复制为文件夹B"

# 统计文件数量
A_COUNT=$(find A -name "*.png" | wc -l)
B_COUNT=$(find B -name "*.png" | wc -l)
echo "文件夹A包含 $A_COUNT 张图片"
echo "文件夹B包含 $B_COUNT 张图片"

if [ "$A_COUNT" -ne "$B_COUNT" ]; then
    echo "❌ 错误: 复制后文件数量不匹配"
    exit 1
fi

cd ..  # 回到RIFE目录

# 更新 morepipeline.py 的配置（创建静态场景配置）
echo "配置静态场景参数..."

# 备份原始配置（如果存在）
if [ -f "morepipeline.py.bak" ]; then
    echo "恢复原始 morepipeline.py 配置..."
    cp morepipeline.py.bak morepipeline.py
else
    echo "备份原始 morepipeline.py 配置..."
    cp morepipeline.py morepipeline.py.bak
fi

# 修改配置为静态场景（A=0.0, B=1.0）
sed -i 's/VIEWS.*=.*/VIEWS = ["A", "B"]/' morepipeline.py
sed -i 's/TIME_MAP.*=.*/TIME_MAP = {"A": 0.0, "B": 1.0}/' morepipeline.py

echo "✅ 静态场景配置完成"
echo "VIEWS = [\"A\", \"B\"]"
echo "TIME_MAP = {\"A\": 0.0, \"B\": 1.0}"

#### ——— 6. 运行静态场景处理（跳过插帧） ———
echo "运行静态场景处理（跳过插帧）..."

python morepipeline.py --skip_interp

if [ ! -d "FINAL" ]; then
    echo "❌ 错误: FINAL 目录未生成"
    exit 1
fi

# 验证FINAL目录内容
FINAL_DIRS=$(find FINAL -maxdepth 1 -type d | wc -l)
FINAL_DIRS=$((FINAL_DIRS - 1))  # 减去FINAL目录本身
echo "FINAL 目录包含 $FINAL_DIRS 个时间点目录"

if [ "$FINAL_DIRS" -ne 2 ]; then
    echo "❌ 错误: 期望2个时间点目录（A和B），实际发现 $FINAL_DIRS 个"
    exit 1
fi

echo "✅ 静态场景处理完成"

#### ——— 7. 数据分割和迁移 ———
echo "开始数据分割和迁移..."

# 先确保符号链接不干扰本地处理
cd "$RIFE_DIR"
if [ -L "SPLITS" ]; then
    rm SPLITS  # 删除符号链接
fi

# 运行 get_together.py 进行数据分割
python get_together.py

if [ ! -d "SPLITS" ]; then
    echo "❌ 错误: SPLITS 目录未生成"
    exit 1
fi

# 验证分割结果
TRAIN_COUNT=$(find SPLITS/train -name "*.png" 2>/dev/null | wc -l)
VAL_COUNT=$(find SPLITS/val -name "*.png" 2>/dev/null | wc -l)
TEST_COUNT=$(find SPLITS/test -name "*.png" 2>/dev/null | wc -l)

echo "数据分割结果:"
echo "  训练集: $TRAIN_COUNT 张图片"
echo "  验证集: $VAL_COUNT 张图片"
echo "  测试集: $TEST_COUNT 张图片"

# 迁移数据到项目标准位置
cd "$PROJECT_ROOT"

# 清理现有数据（如果存在）
if [ -e "data/dnerf/SPLITS" ]; then
    echo "检测到现有数据，直接覆盖..."
    rm -rf data/dnerf/SPLITS
fi

mkdir -p data/dnerf
mv "$RIFE_DIR/SPLITS" data/dnerf/

# 创建符号链接以保持兼容性
cd "$RIFE_DIR"
ln -sf ../data/dnerf/SPLITS SPLITS
cd "$PROJECT_ROOT"

echo "✅ 数据迁移完成"

#### ——— 8. 4DGaussians 训练（静态推理模式） ———
echo "开始 4DGaussians 静态推理训练..."

# 自动生成动作名称：static_时间戳
CURRENT_TIME=$(date '+%Y%m%d_%H%M%S')
ACTION_NAME="static_$CURRENT_TIME"

echo "自动生成动作名称: $ACTION_NAME"

# 保存动作名称到配置文件
mkdir -p config
echo "$ACTION_NAME" > config/action_name.txt

# 验证动作名称
if [[ ! "$ACTION_NAME" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "❌ 错误: 动作名称格式不正确"
    exit 1
fi

# 设置输出目录
OUTPUT_DIR="output/dnerf/$ACTION_NAME"
echo "训练输出目录: $OUTPUT_DIR"

# 设置训练端口（避免冲突）
TRAIN_PORT=${TRAIN_PORT:-6017}
echo "训练端口: $TRAIN_PORT"

# 运行 4DGaussians 训练
echo "开始 4DGaussians 训练..."
python train.py \
    -s data/dnerf/SPLITS \
    --port $TRAIN_PORT \
    --expname "dnerf/$ACTION_NAME" \
    --configs arguments/dnerf/jumpingjacks.py \
    --iterations 10000

# 验证训练结果
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 错误: 训练输出目录未生成"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR/point_cloud" ]; then
    echo "❌ 错误: 点云模型目录未生成"
    exit 1
fi

echo "✅ 4DGaussians 训练完成"

#### ——— 9. 导出逐帧模型 ———
echo "开始导出逐帧模型..."

python export_perframe_3DGS.py \
    --iteration 10000 \
    --configs arguments/dnerf/jumpingjacks.py \
    --model_path "$OUTPUT_DIR"

# 验证导出结果
GAUSSIAN_DIR="$OUTPUT_DIR/gaussian_pertimestamp"
if [ ! -d "$GAUSSIAN_DIR" ]; then
    echo "❌ 错误: 逐帧模型目录未生成"
    exit 1
fi

PLY_COUNT=$(find "$GAUSSIAN_DIR" -name "*.ply" | wc -l)
echo "导出了 $PLY_COUNT 个逐帧模型文件"

if [ "$PLY_COUNT" -eq 0 ]; then
    echo "❌ 错误: 没有导出任何PLY文件"
    exit 1
fi

echo "✅ 逐帧模型导出完成"

#### ——— 10. 保留第一个PLY文件 ———
echo "保留第一个PLY文件，删除其他文件..."

# 获取第一个PLY文件
FIRST_PLY=$(find "$GAUSSIAN_DIR" -name "*.ply" | sort | head -1)
FIRST_PLY_NAME=$(basename "$FIRST_PLY")

if [ -z "$FIRST_PLY" ]; then
    echo "❌ 错误: 未找到PLY文件"
    exit 1
fi

echo "第一个PLY文件: $FIRST_PLY_NAME"

# 创建临时目录保存第一个文件
TEMP_DIR="/tmp/static_ply_$$"
mkdir -p "$TEMP_DIR"
cp "$FIRST_PLY" "$TEMP_DIR/"

# 清空原目录并只保留第一个文件
rm -rf "$GAUSSIAN_DIR"/*
cp "$TEMP_DIR/$FIRST_PLY_NAME" "$GAUSSIAN_DIR/"

# 清理临时目录
rm -rf "$TEMP_DIR"

# 验证结果
FINAL_PLY_COUNT=$(find "$GAUSSIAN_DIR" -name "*.ply" | wc -l)
echo "最终保留 $FINAL_PLY_COUNT 个PLY文件"

if [ "$FINAL_PLY_COUNT" -ne 1 ]; then
    echo "❌ 错误: 应该只保留1个PLY文件，实际保留了 $FINAL_PLY_COUNT 个"
    exit 1
fi

echo "✅ PLY文件筛选完成，只保留: $FIRST_PLY_NAME"

#### ——— 11. 生成静态推理报告 ———
REPORT_FILE="$OUTPUT_DIR/static_inference_report.md"
echo "生成静态推理报告: $REPORT_FILE"

cat > "$REPORT_FILE" << EOF
# 静态场景推理训练完成报告

## 基本信息
- 动作名称: $ACTION_NAME
- 训练时间: $(date '+%Y-%m-%d %H:%M:%S')
- 工作节点: $(hostname)
- 训练模式: 静态场景推理

## 数据配置
- 静态场景: A(t=0.0) → B(t=1.0)
- 跳过插帧: 是（使用 --skip_interp）
- 训练迭代数: 10000（快速收敛）

## 输出文件
- 训练模型: $OUTPUT_DIR/point_cloud/iteration_10000/
- 静态PLY: $GAUSSIAN_DIR/$FIRST_PLY_NAME
- 配置文件: config/action_name.txt

## 数据统计
- 原始图片A: $A_COUNT 张
- 复制图片B: $B_COUNT 张
- 训练集: $TRAIN_COUNT 张
- 验证集: $VAL_COUNT 张
- 测试集: $TEST_COUNT 张
- 导出PLY: 1 个（仅保留第一个）

## 后续使用

### 1. 查看静态模型
\`\`\`bash
# 使用MeshLab或CloudCompare查看
meshlab $GAUSSIAN_DIR/$FIRST_PLY_NAME
\`\`\`

### 2. 用于推理任意物体
此静态模型可作为参考场景，用于推理任意物体的变形。

### 3. 笼节点模型训练
可基于此静态场景进行笼节点模型训练。

## 技术说明
- 静态场景通过复制文件夹A为B创建
- 时间设置为0.0和1.0，创建最简单的时序变化
- 跳过RIFE插帧，直接使用原始帧
- 快速训练模式，降低迭代数至10000
- 只保留第一个PLY文件，作为静态参考模型

EOF

echo "========================================="
echo "静态场景推理数据准备与训练完成"
echo "动作名称: $ACTION_NAME"
echo "静态PLY: $GAUSSIAN_DIR/$FIRST_PLY_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "报告文件: $REPORT_FILE"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "✅ 静态场景推理训练成功完成！" 
