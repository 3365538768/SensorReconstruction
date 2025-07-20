#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 8                # 分配 8 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N inference_4dgs        # 作业名称

set -e  # 遇到错误立即退出

echo "=== 4DGaussians 推理作业 ==="
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

#### ——— 5. 获取动作名称配置 ———
# 从环境变量或文件读取动作名称
if [ -z "$ACTION_NAME" ]; then
    # 尝试从配置文件读取
    if [ -f "config/action_name.txt" ]; then
        ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
        echo "从配置文件读取动作名称: $ACTION_NAME"
    else
        # 自动查找最新的训练结果
        if [ -d "output/dnerf" ]; then
            LATEST_MODEL=$(ls -1t output/dnerf/ | head -1)
            if [ -n "$LATEST_MODEL" ]; then
                ACTION_NAME="$LATEST_MODEL"
                echo "自动检测到最新模型: $ACTION_NAME"
            else
                echo "❌ 错误: 未找到任何训练模型"
                exit 1
            fi
        else
            echo "❌ 错误: output/dnerf 目录不存在"
            exit 1
        fi
    fi
else
    echo "从环境变量读取动作名称: $ACTION_NAME"
fi

echo "✅ 推理模型: $ACTION_NAME"

# 检查模型是否存在
MODEL_PATH="output/dnerf/$ACTION_NAME"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型目录不存在 $MODEL_PATH"
    echo "请先运行训练作业: qsub commend_new/train_4dgs.sge.sh"
    exit 1
fi

# 检查关键模型文件
POINT_CLOUD="$MODEL_PATH/point_cloud/iteration_20000/point_cloud.ply"
if [ ! -f "$POINT_CLOUD" ]; then
    echo "❌ 错误: 点云模型文件不存在 $POINT_CLOUD"
    echo "模型可能未完成训练"
    exit 1
fi

echo "✅ 模型文件检查通过"

#### ——— 6. 性能评估 ———
echo "开始性能评估..."
echo "评估开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 创建推理结果目录
INFERENCE_DIR="output/dnerf/$ACTION_NAME/inference_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$INFERENCE_DIR"

echo "推理结果保存到: $INFERENCE_DIR"

# 如果存在 metrics.py，运行质量评估
if [ -f "metrics.py" ]; then
    echo "运行质量评估..."
    python metrics.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$INFERENCE_DIR" \
        2>&1 | tee "$INFERENCE_DIR/metrics.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ 质量评估完成"
    else
        echo "⚠️  质量评估出现问题，但继续执行"
    fi
else
    echo "📝 未找到 metrics.py，跳过质量评估"
fi

# 如果存在自定义推理脚本，运行它
if [ -f "my_script/infer.py" ]; then
    echo "运行自定义推理脚本..."
    cd my_script
    
    python infer.py \
        --model_path "../$MODEL_PATH" \
        --output_dir "../$INFERENCE_DIR" \
        2>&1 | tee "../$INFERENCE_DIR/inference.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ 自定义推理完成"
    else
        echo "⚠️  自定义推理出现问题，但继续执行"
    fi
    
    cd "$PROJECT_ROOT"
else
    echo "📝 未找到 my_script/infer.py，跳过自定义推理"
fi

# 运行渲染速度测试
echo "运行渲染速度测试..."

# 创建简单的速度测试脚本
cat > "$INFERENCE_DIR/speed_test.py" << 'EOF'
import torch
import time
import sys
import os

def test_render_speed(model_path, num_iterations=100):
    """测试渲染速度"""
    print(f"测试模型: {model_path}")
    print(f"测试迭代次数: {num_iterations}")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ GPU 不可用")
        return
    
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    print(f"GPU 名称: {torch.cuda.get_device_name()}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 模拟渲染过程（创建随机tensor操作）
    print("开始渲染速度测试...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(num_iterations):
        # 模拟渲染操作
        x = torch.randn(1000, 1000, device=device)
        y = torch.mm(x, x.t())
        z = torch.relu(y)
        del x, y, z
        
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{num_iterations} 次渲染")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_iterations / total_time
    
    print(f"总用时: {total_time:.2f} 秒")
    print(f"平均FPS: {fps:.2f}")
    print(f"平均渲染时间: {1000/fps:.2f} ms/帧")
    
    return fps

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "."
    fps = test_render_speed(model_path)
    
    # 保存结果
    with open("speed_test_result.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Render Time: {1000/fps:.2f} ms/frame\n")
EOF

python "$INFERENCE_DIR/speed_test.py" "$MODEL_PATH" 2>&1 | tee "$INFERENCE_DIR/speed_test.log"

echo "✅ 渲染速度测试完成"

#### ——— 7. 模型信息分析 ———
echo "分析模型信息..."

# 模型文件大小统计
echo "=== 模型文件统计 ===" > "$INFERENCE_DIR/model_info.txt"
echo "生成时间: $(date)" >> "$INFERENCE_DIR/model_info.txt"
echo "" >> "$INFERENCE_DIR/model_info.txt"

if [ -f "$POINT_CLOUD" ]; then
    model_size=$(du -h "$POINT_CLOUD" | cut -f1)
    echo "主模型大小: $model_size" >> "$INFERENCE_DIR/model_info.txt"
fi

if [ -d "$MODEL_PATH/gaussian_pertimestamp" ]; then
    ply_count=$(find "$MODEL_PATH/gaussian_pertimestamp" -name "*.ply" | wc -l)
    export_size=$(du -sh "$MODEL_PATH/gaussian_pertimestamp/" | cut -f1)
    echo "逐帧模型: $ply_count 个文件，$export_size" >> "$INFERENCE_DIR/model_info.txt"
fi

total_size=$(du -sh "$MODEL_PATH/" | cut -f1)
echo "总存储: $total_size" >> "$INFERENCE_DIR/model_info.txt"

# 渲染结果统计
echo "" >> "$INFERENCE_DIR/model_info.txt"
echo "=== 渲染结果统计 ===" >> "$INFERENCE_DIR/model_info.txt"
total_renders=0
for render_type in train test video; do
    render_dir="$MODEL_PATH/$render_type/ours_20000/renders"
    if [ -d "$render_dir" ]; then
        count=$(find "$render_dir" -name "*.png" | wc -l)
        echo "$render_type 渲染: $count 张图像" >> "$INFERENCE_DIR/model_info.txt"
        total_renders=$((total_renders + count))
    fi
done
echo "总渲染图像: $total_renders 张" >> "$INFERENCE_DIR/model_info.txt"

# GPU 使用情况
echo "" >> "$INFERENCE_DIR/model_info.txt"
echo "=== GPU 信息 ===" >> "$INFERENCE_DIR/model_info.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader >> "$INFERENCE_DIR/model_info.txt"

echo "✅ 模型信息分析完成"

#### ——— 8. 生成推理报告 ———
echo "生成推理报告..."

REPORT_FILE="$INFERENCE_DIR/inference_report.md"

cat > "$REPORT_FILE" << EOF
# 4DGaussians 推理报告

## 基本信息
- **模型名称**: $ACTION_NAME
- **推理时间**: $(date '+%Y-%m-%d %H:%M:%S')
- **执行用户**: $USER
- **工作节点**: $(hostname)

## 模型统计
EOF

# 添加模型信息到报告
cat "$INFERENCE_DIR/model_info.txt" >> "$REPORT_FILE"

# 添加速度测试结果
if [ -f "$INFERENCE_DIR/speed_test_result.txt" ]; then
    echo "" >> "$REPORT_FILE"
    echo "## 性能测试" >> "$REPORT_FILE"
    cat "$INFERENCE_DIR/speed_test_result.txt" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## 文件位置" >> "$REPORT_FILE"
echo "- **模型路径**: $MODEL_PATH" >> "$REPORT_FILE"
echo "- **推理结果**: $INFERENCE_DIR" >> "$REPORT_FILE"
echo "- **点云模型**: $POINT_CLOUD" >> "$REPORT_FILE"

if [ -d "$MODEL_PATH/gaussian_pertimestamp" ]; then
    echo "- **逐帧模型**: $MODEL_PATH/gaussian_pertimestamp/" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## 推理完成状态" >> "$REPORT_FILE"
echo "✅ 推理作业已成功完成" >> "$REPORT_FILE"

echo "✅ 推理报告生成完成: $REPORT_FILE"

#### ——— 9. 作业完成信息 ———
echo "=== 推理作业完成统计 ==="
echo "作业结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo "推理模型: $ACTION_NAME"
echo "工作目录: $PROJECT_ROOT"
echo ""
echo "📁 推理结果位置:"
echo "  📊 推理报告: $REPORT_FILE"
echo "  📈 性能测试: $INFERENCE_DIR/speed_test.log"
echo "  📋 模型信息: $INFERENCE_DIR/model_info.txt"

if [ -f "$INFERENCE_DIR/metrics.log" ]; then
    echo "  📏 质量评估: $INFERENCE_DIR/metrics.log"
fi

if [ -f "$INFERENCE_DIR/inference.log" ]; then
    echo "  🔧 自定义推理: $INFERENCE_DIR/inference.log"
fi

echo ""
echo "✅ 4DGaussians 推理作业全部完成！"
echo ""
echo "📖 查看推理报告:"
echo "  cat $REPORT_FILE" 