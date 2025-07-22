#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 4                # 分配 4 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N static_inference_exec # 作业名称

set -e  # 遇到错误立即退出

echo "========================================="
echo "静态场景推理执行作业"
echo "作业开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo "工作节点: $(hostname)"
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "========================================="

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
MY_SCRIPT_DIR="$PROJECT_ROOT/my_script"

echo "项目根目录: $PROJECT_ROOT"
echo "my_script目录: $MY_SCRIPT_DIR"

# 检查项目目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 错误: 项目目录不存在 $PROJECT_ROOT"
    exit 1
fi

if [ ! -d "$MY_SCRIPT_DIR" ]; then
    echo "❌ 错误: my_script目录不存在 $MY_SCRIPT_DIR"
    exit 1
fi

cd "$PROJECT_ROOT"

#### ——— 5. 读取静态推理配置 ———
echo "读取静态推理动作名称配置..."

if [ ! -f "config/action_name.txt" ]; then
    echo "❌ 错误: config/action_name.txt 文件不存在！"
    echo "请先运行静态推理准备脚本: qsub commend_new/static_inference_preparation.sge.sh"
    exit 1
fi

ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
echo "静态推理动作名称: $ACTION_NAME"

# 验证动作名称格式（应该是static_开头）
if [[ ! "$ACTION_NAME" =~ ^static_ ]]; then
    echo "❌ 错误: 动作名称格式不正确，应该以'static_'开头"
    echo "当前动作名称: $ACTION_NAME"
    exit 1
fi

# 设置关键路径
STATIC_OUTPUT_DIR="output/dnerf/$ACTION_NAME"
GAUSSIAN_DIR="$STATIC_OUTPUT_DIR/gaussian_pertimestamp"
MY_SCRIPT_ACTION_DIR="$MY_SCRIPT_DIR/$ACTION_NAME"
INFERENCE_OUTPUT_DIR="inference_outputs/$ACTION_NAME"

echo "静态输出目录: $STATIC_OUTPUT_DIR"
echo "高斯模型目录: $GAUSSIAN_DIR"
echo "my_script动作目录: $MY_SCRIPT_ACTION_DIR"
echo "推理输出目录: $INFERENCE_OUTPUT_DIR"

#### ——— 6. 检查静态推理准备结果 ———
echo "检查静态推理准备结果..."

if [ ! -d "$STATIC_OUTPUT_DIR" ]; then
    echo "❌ 错误: 静态输出目录不存在 $STATIC_OUTPUT_DIR"
    echo "请先运行静态推理准备脚本"
    exit 1
fi

if [ ! -d "$GAUSSIAN_DIR" ]; then
    echo "❌ 错误: 高斯模型目录不存在 $GAUSSIAN_DIR"
    echo "静态推理准备可能未完成"
    exit 1
fi

# 查找静态PLY文件
STATIC_PLY_FILES=($(find "$GAUSSIAN_DIR" -name "*.ply" | sort))
STATIC_PLY_COUNT=${#STATIC_PLY_FILES[@]}

echo "发现 $STATIC_PLY_COUNT 个静态PLY文件"

if [ "$STATIC_PLY_COUNT" -eq 0 ]; then
    echo "❌ 错误: 未找到静态PLY文件"
    exit 1
elif [ "$STATIC_PLY_COUNT" -ne 1 ]; then
    echo "⚠️  警告: 期望1个静态PLY文件，实际发现 $STATIC_PLY_COUNT 个"
    echo "将使用第一个文件: ${STATIC_PLY_FILES[0]}"
fi

STATIC_PLY_FILE="${STATIC_PLY_FILES[0]}"
STATIC_PLY_NAME=$(basename "$STATIC_PLY_FILE")
echo "✅ 使用静态PLY文件: $STATIC_PLY_NAME"

#### ——— 7. 检查导入文档状态 ———
echo "检查两个文档的导入状态..."

# 检查my_script/data/ACTION_NAME目录下的文件
DATA_DIR="$MY_SCRIPT_DIR/data/$ACTION_NAME"
echo "检查数据目录: $DATA_DIR"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 数据目录不存在 $DATA_DIR"
    echo "请确认笼节点模型训练数据准备已完成"
    exit 1
fi

# 检查region.json文件
REGION_JSON="$DATA_DIR/region.json"
if [ ! -f "$REGION_JSON" ]; then
    echo "❌ 错误: region.json 文件不存在 $REGION_JSON"
    echo "请使用my_script/user/user.py导入region.json文件"
    exit 1
fi

echo "✅ region.json 文件检查通过"

# 验证region.json格式
if ! python -c "import json; json.load(open('$REGION_JSON'))" 2>/dev/null; then
    echo "❌ 错误: region.json 文件格式不正确"
    exit 1
fi

echo "✅ region.json 格式验证通过"

# 检查sensor.csv文件
SENSOR_CSV="$DATA_DIR/sensor.csv"
if [ ! -f "$SENSOR_CSV" ]; then
    echo "❌ 错误: sensor.csv 文件不存在 $SENSOR_CSV"
    echo "请使用my_script/user/user.py导入sensor.csv文件"
    exit 1
fi

echo "✅ sensor.csv 文件检查通过"

# 验证sensor.csv格式（检查是否有数据且列数合理）
CSV_LINES=$(wc -l < "$SENSOR_CSV")
CSV_COLUMNS=$(head -1 "$SENSOR_CSV" | tr ',' '\n' | wc -l)

echo "sensor.csv 统计: $CSV_LINES 行, $CSV_COLUMNS 列"

if [ "$CSV_LINES" -lt 2 ]; then
    echo "❌ 错误: sensor.csv 数据行数过少（少于2行）"
    exit 1
fi

if [ "$CSV_COLUMNS" -lt 10 ]; then
    echo "❌ 错误: sensor.csv 列数过少（少于10列）"
    exit 1
fi

echo "✅ sensor.csv 格式验证通过"

#### ——— 8. 准备my_script/ACTION_NAME目录 ———
echo "准备my_script/$ACTION_NAME目录..."

# 创建目标目录
mkdir -p "$MY_SCRIPT_ACTION_DIR"
echo "✅ 创建目录: $MY_SCRIPT_ACTION_DIR"

# 将静态PLY文件复制并改名为init.ply
INIT_PLY_PATH="$MY_SCRIPT_ACTION_DIR/init.ply"
cp "$STATIC_PLY_FILE" "$INIT_PLY_PATH"

echo "✅ PLY文件处理完成:"
echo "  源文件: $STATIC_PLY_FILE"
echo "  目标文件: $INIT_PLY_PATH"

# 验证init.ply文件
if [ ! -f "$INIT_PLY_PATH" ]; then
    echo "❌ 错误: init.ply 文件创建失败"
    exit 1
fi

INIT_PLY_SIZE=$(stat -c%s "$INIT_PLY_PATH")
echo "✅ init.ply 文件大小: $INIT_PLY_SIZE 字节"

if [ "$INIT_PLY_SIZE" -eq 0 ]; then
    echo "❌ 错误: init.ply 文件为空"
    exit 1
fi

#### ——— 9. 创建模型输出目录 ———
echo "创建模型输出目录..."

MODEL_OUTPUT_DIR="$MY_SCRIPT_DIR/outputs/$ACTION_NAME"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "✅ 模型输出目录: $MODEL_OUTPUT_DIR"

# 检查是否有训练好的模型
MODEL_PATH="$MODEL_OUTPUT_DIR/deform_model_final.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  警告: 训练模型不存在 $MODEL_PATH"
    echo "将尝试创建示例模型文件"
    
    # 创建模型目录并生成占位符
    echo "创建模型占位符文件..."
    touch "$MODEL_PATH"
    echo "✅ 模型占位符已创建（实际使用需要训练完成的模型）"
fi

#### ——— 10. 创建推理输出目录 ———
echo "创建推理输出目录..."

mkdir -p "$INFERENCE_OUTPUT_DIR"
echo "✅ 推理输出目录: $INFERENCE_OUTPUT_DIR"

#### ——— 11. 运行推理脚本 ———
echo "开始运行推理脚本..."

cd "$MY_SCRIPT_DIR"

# 设置推理参数
DATA_DIR_PARAM="$ACTION_NAME"                    # my_script/ACTION_NAME
INIT_PLY_PARAM="$INIT_PLY_PATH"                  # my_script/ACTION_NAME/init.ply
MODEL_PATH_PARAM="outputs/$ACTION_NAME/deform_model_final.pth"  # my_script/outputs/ACTION_NAME
OUT_DIR_PARAM="../$INFERENCE_OUTPUT_DIR"         # inference_outputs/ACTION_NAME

echo "推理参数配置:"
echo "  --data_dir: $DATA_DIR_PARAM"
echo "  --init_ply_path: $INIT_PLY_PARAM"
echo "  --model_path: $MODEL_PATH_PARAM"
echo "  --out_dir: $OUT_DIR_PARAM"

# 检查推理脚本是否存在
if [ ! -f "infer.py" ]; then
    echo "❌ 错误: infer.py 脚本不存在于 $MY_SCRIPT_DIR"
    exit 1
fi

echo "✅ 推理脚本检查通过"

# 运行推理
echo "正在执行推理..."
python infer.py \
    --data_dir "$DATA_DIR_PARAM" \
    --init_ply_path "$INIT_PLY_PARAM" \
    --model_path "$MODEL_PATH_PARAM" \
    --out_dir "$OUT_DIR_PARAM" \
    --sensor_dim 512 \
    --cage_res 15 15 15 \
    --sensor_res 10 10 \
    --num_fourier_bands 8 \
    --num_time_bands 6 \
    --falloff_distance 0.0

echo "✅ 推理执行完成"

#### ——— 12. 验证推理结果 ———
echo "验证推理结果..."

CAGES_OUTPUT_DIR="$INFERENCE_OUTPUT_DIR/cages_pred"
OBJECTS_OUTPUT_DIR="$INFERENCE_OUTPUT_DIR/objects_world"

if [ ! -d "$CAGES_OUTPUT_DIR" ]; then
    echo "❌ 错误: 笼预测结果目录不存在 $CAGES_OUTPUT_DIR"
    exit 1
fi

if [ ! -d "$OBJECTS_OUTPUT_DIR" ]; then
    echo "❌ 错误: 物体推理结果目录不存在 $OBJECTS_OUTPUT_DIR"
    exit 1
fi

CAGE_PLY_COUNT=$(find "$CAGES_OUTPUT_DIR" -name "*.ply" | wc -l)
OBJECT_PLY_COUNT=$(find "$OBJECTS_OUTPUT_DIR" -name "*.ply" | wc -l)

echo "推理结果统计:"
echo "  笼预测PLY文件: $CAGE_PLY_COUNT 个"
echo "  物体推理PLY文件: $OBJECT_PLY_COUNT 个"

if [ "$CAGE_PLY_COUNT" -eq 0 ] && [ "$OBJECT_PLY_COUNT" -eq 0 ]; then
    echo "❌ 错误: 未生成任何推理结果"
    exit 1
fi

echo "✅ 推理结果验证通过"

#### ——— 13. 生成推理执行报告 ———
REPORT_FILE="$INFERENCE_OUTPUT_DIR/inference_execution_report.md"
echo "生成推理执行报告: $REPORT_FILE"

cat > "$REPORT_FILE" << EOF
# 静态场景推理执行完成报告

## 基本信息
- 动作名称: $ACTION_NAME
- 执行时间: $(date '+%Y-%m-%d %H:%M:%S')
- 工作节点: $(hostname)
- 执行模式: 静态场景任意物体推理

## 输入文件状态
- 静态PLY文件: $STATIC_PLY_NAME (已改名为init.ply)
- region.json: ✅ 导入成功 (格式验证通过)
- sensor.csv: ✅ 导入成功 ($CSV_LINES 行, $CSV_COLUMNS 列)

## 推理配置
- 数据目录: my_script/$ACTION_NAME
- 初始PLY: my_script/$ACTION_NAME/init.ply
- 模型路径: my_script/outputs/$ACTION_NAME/deform_model_final.pth
- 输出目录: inference_outputs/$ACTION_NAME

## 推理参数
- 传感器维度: 512
- 笼分辨率: 15×15×15
- 传感器分辨率: 10×10
- 傅里叶频带: 8
- 时间编码频带: 6
- 衰减距离: 0.0

## 输出结果
- 笼预测PLY: $CAGE_PLY_COUNT 个文件
- 物体推理PLY: $OBJECT_PLY_COUNT 个文件
- 输出目录: $INFERENCE_OUTPUT_DIR

## 文件结构
\`\`\`
my_script/$ACTION_NAME/
├── init.ply                    # 静态参考模型
├── region.json                 # 边界框配置
└── sensor.csv                  # 传感器数据

inference_outputs/$ACTION_NAME/
├── cages_pred/                 # 笼预测结果
│   ├── cage_00000.ply
│   └── ...
├── objects_world/              # 物体推理结果
│   ├── object_00000.ply
│   └── ...
└── inference_execution_report.md
\`\`\`

## 后续步骤

### 1. 查看推理结果
\`\`\`bash
# 查看物体推理结果
ls -la inference_outputs/$ACTION_NAME/objects_world/

# 使用MeshLab查看
meshlab inference_outputs/$ACTION_NAME/objects_world/object_00000.ply
\`\`\`

### 2. 渲染运动视频
可继续执行第5步渲染运动视频脚本。

### 3. 结果分析
- 对比静态参考模型与推理结果
- 分析传感器数据对变形的影响
- 评估推理质量和时序一致性

## 技术说明
- 基于静态场景的任意物体推理
- 使用笼节点模型控制变形
- 支持传感器数据驱动的实时推理
- 保持高斯球属性的完整性

EOF

cd "$PROJECT_ROOT"

echo "========================================="
echo "静态场景推理执行完成"
echo "动作名称: $ACTION_NAME"
echo "初始PLY: $INIT_PLY_PATH"
echo "推理结果: $INFERENCE_OUTPUT_DIR"
echo "笼预测: $CAGE_PLY_COUNT 个文件"
echo "物体推理: $OBJECT_PLY_COUNT 个文件"
echo "报告文件: $REPORT_FILE"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "✅ 静态场景推理执行成功完成！" 