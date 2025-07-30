#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 8                # 分配 8 个 CPU 核心（降低资源需求）
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡（降低资源需求）
#$ -N train_4dgs            # 作业名称

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 训练作业 ==="
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

echo "项目根目录: $PROJECT_ROOT"

# 检查项目目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 错误: 项目目录不存在 $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# 检查数据是否准备完毕
if [ ! -d "data/dnerf/SPLITS" ]; then
    echo "❌ 错误: 数据集未准备完毕，请先运行数据预处理作业"
    echo "执行命令: qsub commend_new/data_preprocessing.sge.sh"
    exit 1
fi

echo "✅ 数据集检查通过"

#### ——— 5. 获取动作名称配置 ———
# 从环境变量或文件读取动作名称，如果没有则使用默认值
if [ -z "$ACTION_NAME" ]; then
    # 尝试从配置文件读取
    if [ -f "config/action_name.txt" ]; then
        ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
        echo "从配置文件读取动作名称: $ACTION_NAME"
    else
        # 使用默认动作名称，包含时间戳避免冲突
        ACTION_NAME="experiment_$(date '+%Y%m%d_%H%M%S')"
        echo "使用默认动作名称: $ACTION_NAME"
    fi
else
    echo "从环境变量读取动作名称: $ACTION_NAME"
fi

# 验证动作名称
if [[ ! "$ACTION_NAME" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "❌ 错误: 动作名称只能包含字母、数字和下划线"
    echo "当前动作名称: $ACTION_NAME"
    exit 1
fi

echo "✅ 动作名称设置为: $ACTION_NAME"

#### ——— 6. 4DGaussians 训练 ———
echo "开始 4DGaussians 训练..."
echo "训练开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "模型输出路径: output/dnerf/$ACTION_NAME"

# 执行训练
python train.py \
    -s data/dnerf/SPLITS \
    --port 6017 \
    --expname "dnerf/$ACTION_NAME" \
    --configs arguments/dnerf/jumpingjacks.py

if [ $? -eq 0 ]; then
    echo "✅ 4DGaussians 训练完成"
    echo "训练结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 备份SGE日志到logs文件夹
    if [ ! -z "$JOB_ID" ]; then
        echo "备份SGE日志文件到logs文件夹..."
        LOG_BACKUP_DIR="logs/sge_jobs/4DGaussians/$ACTION_NAME"
        mkdir -p "$LOG_BACKUP_DIR"
        
        TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
        
        # 复制SGE输出和错误日志
        if [ -f "train_4dgs.o$JOB_ID" ]; then
            cp "train_4dgs.o$JOB_ID" "$LOG_BACKUP_DIR/sge_output_${TIMESTAMP}.log"
            echo "✅ SGE输出日志已备份: $LOG_BACKUP_DIR/sge_output_${TIMESTAMP}.log"
        fi
        
        if [ -f "train_4dgs.e$JOB_ID" ]; then
            cp "train_4dgs.e$JOB_ID" "$LOG_BACKUP_DIR/sge_error_${TIMESTAMP}.log"
            echo "✅ SGE错误日志已备份: $LOG_BACKUP_DIR/sge_error_${TIMESTAMP}.log"
        fi
        
        # 创建作业信息摘要
        echo "Creating job summary..."
        cat > "$LOG_BACKUP_DIR/job_summary_${TIMESTAMP}.txt" << EOF
SGE作业信息摘要
================
作业ID: $JOB_ID
作业名称: 4DGaussians训练
实验名称: $ACTION_NAME
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
结束时间: $(date '+%Y-%m-%d %H:%M:%S')
节点信息: $(hostname)
GPU信息: $(nvidia-smi --query-gpu=name --format=csv,noheader)
输出目录: output/dnerf/$ACTION_NAME
日志目录: logs/4DGaussians/$ACTION_NAME
状态: 训练成功完成
EOF
        echo "✅ 作业摘要已创建: $LOG_BACKUP_DIR/job_summary_${TIMESTAMP}.txt"
    fi
else
    echo "❌ 训练失败"
    # 即使失败也备份日志用于调试
    if [ ! -z "$JOB_ID" ]; then
        LOG_BACKUP_DIR="logs/sge_jobs/4DGaussians/$ACTION_NAME"
        mkdir -p "$LOG_BACKUP_DIR"
        TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
        
        if [ -f "train_4dgs.o$JOB_ID" ]; then
            cp "train_4dgs.o$JOB_ID" "$LOG_BACKUP_DIR/sge_output_failed_${TIMESTAMP}.log"
        fi
        if [ -f "train_4dgs.e$JOB_ID" ]; then
            cp "train_4dgs.e$JOB_ID" "$LOG_BACKUP_DIR/sge_error_failed_${TIMESTAMP}.log"
        fi
    fi
    exit 1
fi

# 验证训练结果
if [ -f "output/dnerf/$ACTION_NAME/point_cloud/iteration_20000/point_cloud.ply" ]; then
    model_size=$(du -h "output/dnerf/$ACTION_NAME/point_cloud/iteration_20000/point_cloud.ply" | cut -f1)
    echo "✅ 训练模型生成成功，大小: $model_size"
else
    echo "❌ 错误: 训练模型文件未生成"
    exit 1
fi

#### ——— 7. 渲染结果生成 ———
echo "开始渲染结果生成..."
echo "渲染开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python render.py \
    --model_path "output/dnerf/$ACTION_NAME" \
    --configs arguments/dnerf/jumpingjacks.py

if [ $? -eq 0 ]; then
    echo "✅ 渲染完成"
    echo "渲染结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ 渲染失败"
    exit 1
fi

# 验证渲染结果
render_check=0
for render_type in train test video; do
    render_dir="output/dnerf/$ACTION_NAME/$render_type/ours_20000/renders"
    if [ -d "$render_dir" ]; then
        render_count=$(find "$render_dir" -name "*.png" | wc -l)
        echo "✅ $render_type 渲染: $render_count 张图像"
        render_check=$((render_check + 1))
    else
        echo "⚠️  警告: $render_type 渲染结果未找到"
    fi
done

if [ $render_check -eq 0 ]; then
    echo "❌ 错误: 所有渲染结果都未生成"
    exit 1
fi

#### ——— 8. 导出逐帧 3DGS 模型 ———
echo "开始导出逐帧 3DGS 模型..."
echo "导出开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python export_perframe_3DGS.py \
    --iteration 20000 \
    --configs arguments/dnerf/jumpingjacks.py \
    --model_path "output/dnerf/$ACTION_NAME"

if [ $? -eq 0 ]; then
    echo "✅ 逐帧模型导出完成"
    echo "导出结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "❌ 模型导出失败"
    exit 1
fi

# 验证导出结果
if [ -d "output/dnerf/$ACTION_NAME/gaussian_pertimestamp" ]; then
    ply_count=$(find "output/dnerf/$ACTION_NAME/gaussian_pertimestamp" -name "*.ply" | wc -l)
    export_size=$(du -sh "output/dnerf/$ACTION_NAME/gaussian_pertimestamp/" | cut -f1)
    echo "✅ 逐帧模型导出成功: $ply_count 个文件，总大小 $export_size"
else
    echo "❌ 错误: gaussian_pertimestamp 文件夹未生成"
    exit 1
fi

#### ——— 9. 最终统计和验证 ———
echo "=== 4DGaussians 训练完成统计 ==="

# 数据集统计
if [ -d "data/dnerf/SPLITS" ]; then
    train_count=$(find data/dnerf/SPLITS/train -name "*.png" 2>/dev/null | wc -l)
    val_count=$(find data/dnerf/SPLITS/val -name "*.png" 2>/dev/null | wc -l)
    test_count=$(find data/dnerf/SPLITS/test -name "*.png" 2>/dev/null | wc -l)
    echo "输入数据集: train($train_count) + val($val_count) + test($test_count) = $((train_count + val_count + test_count)) 张图像"
fi

# 模型统计
main_model="output/dnerf/$ACTION_NAME/point_cloud/iteration_20000/point_cloud.ply"
if [ -f "$main_model" ]; then
    model_size=$(du -h "$main_model" | cut -f1)
    echo "主模型: $model_size"
fi

# 渲染统计
total_renders=0
for render_type in train test video; do
    render_dir="output/dnerf/$ACTION_NAME/$render_type/ours_20000/renders"
    if [ -d "$render_dir" ]; then
        count=$(find "$render_dir" -name "*.png" | wc -l)
        total_renders=$((total_renders + count))
    fi
done
echo "渲染图像: $total_renders 张"

# 导出模型统计
if [ -d "output/dnerf/$ACTION_NAME/gaussian_pertimestamp" ]; then
    ply_count=$(find "output/dnerf/$ACTION_NAME/gaussian_pertimestamp" -name "*.ply" | wc -l)
    export_size=$(du -sh "output/dnerf/$ACTION_NAME/gaussian_pertimestamp/" | cut -f1)
    echo "导出模型: $ply_count 个 PLY 文件，$export_size"
fi

# 总存储使用
if [ -d "output/dnerf/$ACTION_NAME" ]; then
    total_size=$(du -sh "output/dnerf/$ACTION_NAME/" | cut -f1)
    echo "总存储使用: $total_size"
fi

#### ——— 10. 作业完成信息 ———
echo "=== 作业完成信息 ==="
echo "作业结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo "动作名称: $ACTION_NAME"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "📁 主要输出文件位置:"
echo "  🏗️  训练模型: output/dnerf/$ACTION_NAME/point_cloud/iteration_20000/"
echo "  🎨 渲染图像: output/dnerf/$ACTION_NAME/{train,test,video}/ours_20000/renders/"
echo "  📦 逐帧模型: output/dnerf/$ACTION_NAME/gaussian_pertimestamp/"
echo ""
echo "✅ 4DGaussians 训练流程全部完成！"

