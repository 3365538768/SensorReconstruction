#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 8                # 分配 8 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N data_preprocessing    # 作业名称

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 数据预处理作业 ==="
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
# 使用用户主目录，适配所有用户
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

#### ——— 5. 数据预处理流程 ———
echo "开始数据预处理流程..."

# 进入 RIFE 目录
cd "$RIFE_DIR"

# 检查 originframe 目录
if [ ! -d "originframe" ]; then
    echo "❌ 错误: 未找到 originframe 文件夹"
    echo "请确认 Blender 输出数据已准备完毕"
    exit 1
fi

echo "✅ 发现 originframe 文件夹"

# 获取文件夹列表并动态配置
cd originframe
folders=($(ls -1 | sort))
folder_count=${#folders[@]}
cd ..

echo "检测到 $folder_count 个视角文件夹: ${folders[*]}"

# 生成 VIEWS 和 TIME_MAP 配置
views_array=""
time_map=""
for i in "${!folders[@]}"; do
    views_array+='"'${folders[$i]}'"'
    if [ $i -lt $((folder_count-1)) ]; then 
        views_array+=","
    fi
    
    if [ $folder_count -eq 1 ]; then
        time_value="1.0"
    else
        time_value=$(echo "scale=1; $i / ($folder_count - 1)" | bc -l)
    fi
    time_map+='"'${folders[$i]}'": '$time_value
    if [ $i -lt $((folder_count-1)) ]; then 
        time_map+=","
    fi
done

echo "自动生成配置:"
echo "VIEWS = [$views_array]"
echo "TIME_MAP = {$time_map}"

# 更新 morepipeline.py 配置
cp morepipeline.py morepipeline.py.backup
sed -i "s/VIEWS\s*=.*/VIEWS = [$views_array]/" morepipeline.py
sed -i "s/TIME_MAP\s*=.*/TIME_MAP = {$time_map}/" morepipeline.py

echo "✅ morepipeline.py 配置已更新"

# 执行 RIFE 插帧
echo "开始 RIFE 插帧处理..."
echo "插帧开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python morepipeline.py

if [ $? -eq 0 ]; then
    echo "✅ RIFE 插帧完成"
else
    echo "❌ RIFE 插帧失败"
    exit 1
fi

# 执行数据集分割
echo "开始数据集分割..."
echo "分割开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python get_together.py

if [ $? -eq 0 ]; then
    echo "✅ 数据集分割完成"
else
    echo "❌ 数据集分割失败"
    exit 1
fi

# 数据迁移
echo "开始数据迁移..."
cd "$PROJECT_ROOT"
mkdir -p data/dnerf

# 备份现有数据
if [ -d "data/dnerf/SPLITS" ]; then
    backup_name="SPLITS_backup_$(date '+%Y%m%d_%H%M%S')"
    mv data/dnerf/SPLITS data/dnerf/$backup_name
    echo "已备份原有数据为: $backup_name"
fi

# 移动新数据
mv "$RIFE_DIR/SPLITS" data/dnerf/
echo "✅ 数据已迁移到 data/dnerf/SPLITS"

# 创建符号链接
cd "$RIFE_DIR"
ln -sf ../data/dnerf/SPLITS SPLITS
echo "✅ 已创建符号链接"

#### ——— 6. 最终验证和统计 ———
cd "$PROJECT_ROOT"

echo "=== 数据预处理完成统计 ==="
if [ -d "data/dnerf/SPLITS" ]; then
    train_count=$(find data/dnerf/SPLITS/train -name "*.png" 2>/dev/null | wc -l)
    val_count=$(find data/dnerf/SPLITS/val -name "*.png" 2>/dev/null | wc -l)
    test_count=$(find data/dnerf/SPLITS/test -name "*.png" 2>/dev/null | wc -l)
    total_count=$((train_count + val_count + test_count))
    
    echo "数据集统计:"
    echo "  训练集: $train_count 张图像"
    echo "  验证集: $val_count 张图像"
    echo "  测试集: $test_count 张图像"
    echo "  总计: $total_count 张图像"
    
    # 检查 JSON 文件
    json_count=$(find data/dnerf/SPLITS -name "transforms_*.json" | wc -l)
    echo "  JSON 配置文件: $json_count 个"
    
    if [ $total_count -gt 0 ] && [ $json_count -eq 3 ]; then
        echo "✅ 数据预处理流程全部完成！"
        echo "📁 数据位置: $PROJECT_ROOT/data/dnerf/SPLITS/"
        echo "🔗 符号链接: $RIFE_DIR/SPLITS -> ../data/dnerf/SPLITS"
    else
        echo "❌ 数据预处理验证失败"
        exit 1
    fi
else
    echo "❌ 错误: SPLITS 目录未生成"
    exit 1
fi

#### ——— 7. 作业完成信息 ———
echo "=== 作业完成信息 ==="
echo "作业结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo "工作目录: $PROJECT_ROOT"
echo "数据已准备完毕，可以开始训练阶段"
echo ""
echo "下一步执行命令:"
echo "qsub commend_new/train_4dgs.sge.sh" 