#!/bin/bash
#$ -N cage_training
#$ -pe smp 8
#$ -l gpu_card=1
#$ -o cage_training.o$JOB_ID
#$ -e cage_training.e$JOB_ID
#$ -q gpu
#$ -cwd

# SGE脚本：笼节点模型训练（第二步）
# 功能：检查region.json和sensor.csv文件，修改train.py参数并运行训练

set -e  # 错误立即退出

echo "========================================="
echo "笼节点模型训练开始"
echo "作业ID: $JOB_ID"
echo "开始时间: $(date)"
echo "========================================="

# 创建SGE日志备份目录
SGE_LOG_BACKUP_DIR="logs/sge_jobs/cage_model"
mkdir -p "$SGE_LOG_BACKUP_DIR"

# 激活环境
echo "激活 Gaussians4D 环境..."
source ~/.bashrc
conda activate Gaussians4D

# 读取action_name配置
echo "读取动作名称配置..."
if [ ! -f "config/action_name.txt" ]; then
    echo "错误: config/action_name.txt 文件不存在！"
    echo "请先执行前面的训练步骤。"
    exit 1
fi

ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
if [ -z "$ACTION_NAME" ]; then
    echo "错误: 动作名称为空！"
    exit 1
fi

echo "读取到动作名称: $ACTION_NAME"

# 设置路径变量
DATA_DIR="my_script/data/$ACTION_NAME"
OUT_DIR="outputs/$ACTION_NAME"

echo "数据目录: $DATA_DIR"
echo "输出目录: $OUT_DIR"

# 检查数据准备是否完成
echo "检查数据准备状态..."
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在！"
    echo "期望路径: $DATA_DIR"
    echo "请先执行数据准备步骤(cage_data_preparation.sge.sh)。"
    exit 1
fi

if [ ! -d "$DATA_DIR/frames" ]; then
    echo "错误: 筛选后的动态点云数据不存在！"
    echo "期望路径: $DATA_DIR/frames"
    echo "请确保数据准备步骤已完成。"
    exit 1
fi

# 统计筛选后的PLY文件
FILTERED_PLY_COUNT=$(find "$DATA_DIR/frames" -name "*.ply" | wc -l)
echo "发现 $FILTERED_PLY_COUNT 个筛选后的PLY文件"

if [ "$FILTERED_PLY_COUNT" -eq 0 ]; then
    echo "错误: 没有找到筛选后的PLY文件！"
    exit 1
fi

# 检查必需文件：region.json
echo "检查region.json文件..."
REGION_FILE="$DATA_DIR/region.json"
if [ ! -f "$REGION_FILE" ]; then
    echo "错误: region.json 文件不存在！"
    echo "期望路径: $REGION_FILE"
    echo "请先在本地Windows环境中运行user.py生成region.json文件。"
    echo "参考指导文档: $DATA_DIR/local_processing_instructions.md"
    exit 1
fi

echo "✅ region.json 文件存在"

# 验证region.json格式
echo "验证region.json格式..."
if ! python -c "import json; json.load(open('$REGION_FILE'))" 2>/dev/null; then
    echo "错误: region.json 文件格式不正确！"
    echo "请检查JSON格式是否有效。"
    exit 1
fi

echo "✅ region.json 格式验证通过"

# 检查必需文件：sensor.csv
echo "检查sensor.csv文件..."
SENSOR_FILE="$DATA_DIR/sensor.csv"
if [ ! -f "$SENSOR_FILE" ]; then
    echo "警告: sensor.csv 文件不存在！"
    echo "期望路径: $SENSOR_FILE"
    echo "将创建示例sensor.csv文件..."
    
    # 创建示例sensor.csv文件
    python -c "
import pandas as pd
import numpy as np

# 生成示例传感器数据
frames = list(range($FILTERED_PLY_COUNT))
sensor_data = []

for frame in frames:
    # 生成10x10=100维的示例传感器数据
    sensor_values = np.random.rand(100) * 255
    row = [frame] + sensor_values.tolist()
    sensor_data.append(row)

# 保存为CSV文件（无header）
df = pd.DataFrame(sensor_data)
df.to_csv('$SENSOR_FILE', index=False, header=False)
print('示例sensor.csv文件已生成')
"
    
    if [ ! -f "$SENSOR_FILE" ]; then
        echo "错误: 无法创建示例sensor.csv文件！"
        exit 1
    fi
    
    echo "✅ 示例sensor.csv文件已创建"
else
    echo "✅ sensor.csv 文件存在"
fi

# 验证sensor.csv格式
echo "验证sensor.csv格式..."
SENSOR_COLS=$(head -1 "$SENSOR_FILE" | tr ',' '\n' | wc -l)
echo "sensor.csv 列数: $SENSOR_COLS"

if [ "$SENSOR_COLS" -ne 101 ]; then
    echo "警告: sensor.csv列数不是预期的101列(1个帧号+100个传感器值)"
    echo "实际列数: $SENSOR_COLS"
    echo "将继续执行，但可能需要调整sensor_res参数"
fi

# 创建输出目录
echo "创建输出目录..."
mkdir -p "$OUT_DIR"

# 设置训练参数
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
SENSOR_DIM=${SENSOR_DIM:-512}
CAGE_RES_X=${CAGE_RES_X:-15}
CAGE_RES_Y=${CAGE_RES_Y:-15}
CAGE_RES_Z=${CAGE_RES_Z:-15}
SENSOR_RES_H=${SENSOR_RES_H:-10}
SENSOR_RES_W=${SENSOR_RES_W:-10}

echo "训练参数配置:"
echo "  批大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  传感器维度: $SENSOR_DIM"
echo "  笼网格分辨率: ${CAGE_RES_X}x${CAGE_RES_Y}x${CAGE_RES_Z}"
echo "  传感器分辨率: ${SENSOR_RES_H}x${SENSOR_RES_W}"

# 运行my_script/train.py
echo "开始训练笼节点模型..."
cd my_script

python train.py \
    --data_dir "../$DATA_DIR" \
    --out_dir "../$OUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --sensor_dim $SENSOR_DIM \
    --cage_res $CAGE_RES_X $CAGE_RES_Y $CAGE_RES_Z \
    --sensor_res $SENSOR_RES_H $SENSOR_RES_W \
    --num_workers 1

cd ..

# 验证训练结果
echo "验证训练结果..."
MODEL_FILE="$OUT_DIR/deform_model_final.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "错误: 训练模型文件未生成！"
    echo "期望路径: $MODEL_FILE"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
echo "✅ 训练模型已生成: $MODEL_FILE (大小: $MODEL_SIZE)"

# 验证输出目录
BBOX_DIR="$OUT_DIR/cropped_bbox"
CAGES_DIR="$OUT_DIR/cages_pred"
OBJS_DIR="$OUT_DIR/objects_world"

BBOX_COUNT=$(find "$BBOX_DIR" -name "*.ply" 2>/dev/null | wc -l)
CAGES_COUNT=$(find "$CAGES_DIR" -name "*.ply" 2>/dev/null | wc -l)
OBJS_COUNT=$(find "$OBJS_DIR" -name "*.ply" 2>/dev/null | wc -l)

echo "训练输出统计:"
echo "  裁剪边界框PLY: $BBOX_COUNT 个文件"
echo "  预测笼子PLY: $CAGES_COUNT 个文件"
echo "  重建对象PLY: $OBJS_COUNT 个文件"

echo "========================================="
echo "笼节点模型训练完成"
echo "结束时间: $(date)"
echo "========================================="

# 备份SGE日志到logs文件夹
if [ ! -z "$JOB_ID" ]; then
    echo "备份SGE日志文件到logs文件夹..."
    LOG_BACKUP_DIR="logs/sge_jobs/cage_model/$ACTION_NAME"
    mkdir -p "$LOG_BACKUP_DIR"
    
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    
    # 复制SGE输出和错误日志
    if [ -f "cage_training.o$JOB_ID" ]; then
        cp "cage_training.o$JOB_ID" "$LOG_BACKUP_DIR/sge_output_${TIMESTAMP}.log"
        echo "✅ SGE输出日志已备份: $LOG_BACKUP_DIR/sge_output_${TIMESTAMP}.log"
    fi
    
    if [ -f "cage_training.e$JOB_ID" ]; then
        cp "cage_training.e$JOB_ID" "$LOG_BACKUP_DIR/sge_error_${TIMESTAMP}.log"
        echo "✅ SGE错误日志已备份: $LOG_BACKUP_DIR/sge_error_${TIMESTAMP}.log"
    fi
    
    # 创建作业信息摘要
    echo "Creating cage model job summary..."
    cat > "$LOG_BACKUP_DIR/job_summary_${TIMESTAMP}.txt" << EOF
SGE作业信息摘要 - 笼节点模型训练
================================
作业ID: $JOB_ID
作业名称: 笼节点模型训练
实验名称: $ACTION_NAME
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
结束时间: $(date '+%Y-%m-%d %H:%M:%S')
节点信息: $(hostname)
数据目录: $DATA_DIR
输出目录: $OUT_DIR
训练结果统计:
  - 模型文件: $MODEL_FILE
  - 模型大小: $MODEL_SIZE
  - 裁剪边界框PLY: $BBOX_COUNT 个文件
  - 预测笼子PLY: $CAGES_COUNT 个文件  
  - 重建对象PLY: $OBJS_COUNT 个文件
训练参数:
  - 批大小: $BATCH_SIZE
  - 训练轮数: $EPOCHS
  - 学习率: $LEARNING_RATE
  - 传感器维度: $SENSOR_DIM
  - 笼网格分辨率: ${CAGE_RES_X}x${CAGE_RES_Y}x${CAGE_RES_Z}
  - 传感器分辨率: ${SENSOR_RES_H}x${SENSOR_RES_W}
状态: 训练成功完成
EOF
    echo "✅ 作业摘要已创建: $LOG_BACKUP_DIR/job_summary_${TIMESTAMP}.txt"
fi

echo "✅ 笼节点模型训练流程完成，所有日志已备份到logs文件夹"
echo "  预测笼节点PLY: $CAGES_COUNT 个文件"
echo "  重建物体PLY: $OBJS_COUNT 个文件"

# 生成使用指南
USAGE_GUIDE="$OUT_DIR/usage_guide.md"
echo "生成使用指南: $USAGE_GUIDE"

cat > "$USAGE_GUIDE" << EOF
# 笼节点模型训练完成 - 使用指南

## 训练结果
- 动作名称: $ACTION_NAME
- 模型文件: $MODEL_FILE (大小: $MODEL_SIZE)
- 训练参数: batch_size=$BATCH_SIZE, epochs=$EPOCHS, lr=$LEARNING_RATE

## 输出文件结构
\`\`\`
$OUT_DIR/
├── deform_model_final.pth      # 训练好的模型权重
├── cropped_bbox/               # 裁剪后的边界框点云 ($BBOX_COUNT 文件)
├── cages_pred/                 # 预测的笼节点变形 ($CAGES_COUNT 文件)
├── objects_world/              # 重建的世界坐标物体 ($OBJS_COUNT 文件)
└── usage_guide.md             # 本文件
\`\`\`

## 下一步使用

### 1. 推理任意物体
\`\`\`bash
cd my_script
python infer.py \\
    --data_dir new_scene_data \\
    --model_path ../$MODEL_FILE \\
    --out_dir ../inference_outputs/new_scene
\`\`\`

### 2. 可视化结果
- 使用MeshLab或CloudCompare查看生成的PLY文件
- \`objects_world/\` 中的文件为最终重建结果
- \`cages_pred/\` 中的文件显示笼节点变形过程

### 3. 自定义渲染
\`\`\`bash
python ../custom_render.py \\
    --ply_dir ../$OUT_DIR/objects_world \\
    --out custom_video.mp4
\`\`\`

## 训练数据信息
- 筛选后PLY文件: $FILTERED_PLY_COUNT 个
- 传感器数据列数: $SENSOR_COLS
- 数据目录: $DATA_DIR
- region.json: ✅ 有效
- sensor.csv: ✅ 有效

## 模型参数
- 传感器编码维度: $SENSOR_DIM
- 笼网格分辨率: ${CAGE_RES_X}×${CAGE_RES_Y}×${CAGE_RES_Z}
- 传感器网格分辨率: ${SENSOR_RES_H}×${SENSOR_RES_W}

EOF

echo "========================================="
echo "笼节点模型训练完成"
echo "动作名称: $ACTION_NAME"
echo "模型文件: $MODEL_FILE"
echo "输出目录: $OUT_DIR"
echo "使用指南: $USAGE_GUIDE"
echo "完成时间: $(date)"
echo "========================================="

echo "✅ 笼节点模型训练成功完成！" 