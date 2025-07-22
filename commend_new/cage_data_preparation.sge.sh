#!/bin/bash
#$ -N cage_data_prep
#$ -pe smp 4
#$ -l gpu_card=1
#$ -o cage_data_prep.o$JOB_ID
#$ -e cage_data_prep.e$JOB_ID
#$ -q gpu
#$ -cwd

# SGE脚本：笼节点模型训练数据准备（第一步）
# 功能：读取action_name，复制gaussian_pertimestamp数据，运行get_movepoint.py筛选动态点

set -e  # 错误立即退出

echo "========================================="
echo "笼节点模型训练数据准备开始"
echo "作业ID: $JOB_ID"
echo "开始时间: $(date)"
echo "========================================="

# 激活环境
echo "激活 Gaussians4D 环境..."
source ~/.bashrc
conda activate Gaussians4D

# 读取action_name配置
echo "读取动作名称配置..."
if [ ! -f "config/action_name.txt" ]; then
    echo "错误: config/action_name.txt 文件不存在！"
    echo "请先执行4DGaussians训练步骤以生成动作名称配置。"
    exit 1
fi

ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
if [ -z "$ACTION_NAME" ]; then
    echo "错误: 动作名称为空！"
    exit 1
fi

echo "读取到动作名称: $ACTION_NAME"

# 检查4DGaussians训练输出
GAUSSIAN_SOURCE="output/dnerf/$ACTION_NAME/gaussian_pertimestamp"
if [ ! -d "$GAUSSIAN_SOURCE" ]; then
    echo "错误: 4DGaussians训练输出不存在！"
    echo "期望路径: $GAUSSIAN_SOURCE"
    echo "请确保4DGaussians训练已完成。"
    exit 1
fi

# 统计源数据
PLY_COUNT=$(find "$GAUSSIAN_SOURCE" -name "*.ply" | wc -l)
echo "发现 $PLY_COUNT 个PLY文件在 $GAUSSIAN_SOURCE"

# 创建目标目录结构
TARGET_DIR="my_script/data/$ACTION_NAME"
echo "创建目标目录: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 复制gaussian_pertimestamp数据
echo "复制gaussian_pertimestamp数据..."
if [ -d "$TARGET_DIR/gaussian_pertimestamp" ]; then
    echo "警告: 目标目录已存在，将先删除..."
    rm -rf "$TARGET_DIR/gaussian_pertimestamp"
fi

cp -r "$GAUSSIAN_SOURCE" "$TARGET_DIR/"
echo "数据复制完成"

# 验证复制结果
COPIED_PLY_COUNT=$(find "$TARGET_DIR/gaussian_pertimestamp" -name "*.ply" | wc -l)
echo "复制验证: $COPIED_PLY_COUNT 个PLY文件复制到目标位置"

if [ "$PLY_COUNT" -ne "$COPIED_PLY_COUNT" ]; then
    echo "错误: 复制的文件数量不匹配！"
    echo "源文件数: $PLY_COUNT, 复制文件数: $COPIED_PLY_COUNT"
    exit 1
fi

# 设置get_movepoint.py参数
INPUT_DIR="$TARGET_DIR/gaussian_pertimestamp"
OUTPUT_DIR="$TARGET_DIR/frames"
FILTER_PERCENT=${FILTER_PERCENT:-0.1}  # 默认10%，可通过环境变量覆盖

echo "配置动态点筛选参数:"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  筛选比例: $FILTER_PERCENT"

# 运行get_movepoint.py筛选动态点
echo "开始筛选核心动态点..."
cd my_script

python get_movepoint.py \
    --input_dir "../$INPUT_DIR" \
    --output_dir "../$OUTPUT_DIR" \
    --percent $FILTER_PERCENT

cd ..

# 验证筛选结果
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 筛选输出目录未创建！"
    exit 1
fi

FILTERED_PLY_COUNT=$(find "$OUTPUT_DIR" -name "*.ply" | wc -l)
echo "筛选完成: $FILTERED_PLY_COUNT 个筛选后的PLY文件生成"

if [ "$FILTERED_PLY_COUNT" -ne "$PLY_COUNT" ]; then
    echo "警告: 筛选后文件数量与输入文件数量不匹配"
    echo "输入: $PLY_COUNT, 输出: $FILTERED_PLY_COUNT"
fi

# 生成后续处理指导文件
INSTRUCTION_FILE="$TARGET_DIR/local_processing_instructions.md"
echo "生成本地处理指导文件: $INSTRUCTION_FILE"

cat > "$INSTRUCTION_FILE" << EOF
# 笼节点模型训练 - 本地处理指导

## 当前状态
- 动作名称: $ACTION_NAME
- 筛选后点云: $OUTPUT_DIR ($FILTERED_PLY_COUNT 个文件)
- 筛选比例: $FILTER_PERCENT

## 下一步：本地Windows端处理

### 1. 环境准备
\`\`\`bash
# 在本地Windows环境中
cd D:\\4DGaussians\\my_script\\user
pip install dash plotly plyfile numpy torch dash-bootstrap-components
\`\`\`

### 2. 启动交互界面
\`\`\`bash
python user.py
\`\`\`

### 3. 访问界面
- 打开浏览器访问: http://localhost:8050
- 上传筛选后的PLY文件 (从服务器的 $OUTPUT_DIR 选择任一文件)
- 使用界面框选笼节点范围
- 调节法向量方向(theta/phi)
- 点击"Save & Predict"生成region.json

### 4. 文件传输
- 将生成的region.json传输到服务器的: $TARGET_DIR/
- 准备sensor.csv文件或使用示例数据

### 5. 继续服务器端训练
- 确保region.json和sensor.csv文件就位
- 运行第二个SGE脚本进行模型训练

## 文件结构
\`\`\`
$TARGET_DIR/
├── gaussian_pertimestamp/     # 原始4DGS输出 ($PLY_COUNT 文件)
├── frames/                    # 筛选后动态点 ($FILTERED_PLY_COUNT 文件)
├── region.json               # [待生成] 笼节点区域定义
├── sensor.csv                # [待生成] 传感器数据
└── local_processing_instructions.md  # 本文件
\`\`\`

EOF

echo "========================================="
echo "笼节点模型训练数据准备完成"
echo "动作名称: $ACTION_NAME"
echo "源数据: $PLY_COUNT 个PLY文件"
echo "筛选数据: $FILTERED_PLY_COUNT 个PLY文件"
echo "筛选比例: $FILTER_PERCENT"
echo "目标目录: $TARGET_DIR"
echo "后续指导: $INSTRUCTION_FILE"
echo "完成时间: $(date)"
echo "========================================="

echo "✅ 数据准备完成，请查看指导文件进行下一步本地处理" 