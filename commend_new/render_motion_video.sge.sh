#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 4                # 分配 4 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N render_motion_video   # 作业名称

set -e  # 遇到错误立即退出

echo "========================================="
echo "渲染运动视频作业"
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
module load ffmpeg  # 加载ffmpeg用于视频合成

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

# 验证ffmpeg
echo "FFmpeg 版本:"
ffmpeg -version | head -1

#### ——— 4. 设置工作目录 ———
PROJECT_ROOT="/users/$USER/SensorReconstruction"

echo "项目根目录: $PROJECT_ROOT"

# 检查项目目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 错误: 项目目录不存在 $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

#### ——— 5. 读取推理动作配置 ———
echo "读取推理动作名称配置..."

if [ ! -f "config/action_name.txt" ]; then
    echo "❌ 错误: config/action_name.txt 文件不存在！"
    echo "请先运行推理任意物体脚本"
    exit 1
fi

ACTION_NAME=$(cat config/action_name.txt | tr -d '[:space:]')
echo "推理动作名称: $ACTION_NAME"

# 设置关键路径
TRAIN_DATA_DIR="$PROJECT_ROOT/data/dnerf/SPLITS/train"
INFERENCE_OUTPUT_DIR="$PROJECT_ROOT/my_script/inference_outputs/$ACTION_NAME/objects_world"
MODEL_PATH="$PROJECT_ROOT/output/dnerf/$ACTION_NAME"
SOURCE_PATH="$PROJECT_ROOT/data/dnerf/$ACTION_NAME"

echo "训练数据目录: $TRAIN_DATA_DIR"
echo "推理输出目录: $INFERENCE_OUTPUT_DIR" 
echo "模型路径: $MODEL_PATH"
echo "源数据路径: $SOURCE_PATH"

#### ——— 6. 检查数据和推理结果 ———
echo "检查训练数据和推理结果..."

# 检查训练数据目录
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "❌ 错误: 训练数据目录不存在 $TRAIN_DATA_DIR"
    exit 1
fi

# 统计训练照片数量
PHOTO_COUNT=$(find "$TRAIN_DATA_DIR" -name "*.png" | wc -l)
echo "发现训练照片数量: $PHOTO_COUNT 张"

if [ "$PHOTO_COUNT" -eq 0 ]; then
    echo "❌ 错误: 训练数据目录中没有PNG文件"
    exit 1
fi

# 获取照片编号范围
MIN_PHOTO=$(find "$TRAIN_DATA_DIR" -name "r_*.png" | sed 's/.*r_\([0-9]*\)\.png/\1/' | sort -n | head -1)
MAX_PHOTO=$(find "$TRAIN_DATA_DIR" -name "r_*.png" | sed 's/.*r_\([0-9]*\)\.png/\1/' | sort -n | tail -1)
MIN_PHOTO=$((10#$MIN_PHOTO))  # 去除前导零
MAX_PHOTO=$((10#$MAX_PHOTO))  # 去除前导零

echo "照片编号范围: $MIN_PHOTO - $MAX_PHOTO"
echo "✅ 训练数据检查通过"

# 检查推理结果
if [ ! -d "$INFERENCE_OUTPUT_DIR" ]; then
    echo "❌ 错误: 推理输出目录不存在 $INFERENCE_OUTPUT_DIR"
    echo "请先运行推理任意物体脚本"
    exit 1
fi

PLY_COUNT=$(find "$INFERENCE_OUTPUT_DIR" -name "*.ply" | wc -l)
echo "发现推理PLY文件数量: $PLY_COUNT 个"

if [ "$PLY_COUNT" -eq 0 ]; then
    echo "❌ 错误: 推理输出目录中没有PLY文件"
    echo "请确认推理任意物体步骤已完成"
    exit 1
fi

echo "✅ 推理结果检查通过"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在 $MODEL_PATH"
    exit 1
fi

echo "✅ 模型路径检查通过"

#### ——— 7. 读取照片编号配置 ———
echo "读取照片编号配置..."

if [ ! -f "$PROJECT_ROOT/config/camera_number.txt" ]; then
    echo "❌ 错误: config/camera_number.txt 文件不存在！"
    echo "请先运行以下命令设置照片编号:"
    echo "  read -p \"请输入照片编号（0-688范围内，如 344）: \" CAMERA_NUMBER"
    echo "  echo \"\$CAMERA_NUMBER\" > config/camera_number.txt"
    exit 1
fi

USER_PHOTO_NUM=$(cat "$PROJECT_ROOT/config/camera_number.txt" | tr -d '[:space:]')
echo "从配置文件读取照片编号: $USER_PHOTO_NUM"

# 提供一些建议编号供参考
QUARTER_1=$((MIN_PHOTO + (MAX_PHOTO - MIN_PHOTO) / 4))
HALF=$((MIN_PHOTO + (MAX_PHOTO - MIN_PHOTO) / 2))
QUARTER_3=$((MIN_PHOTO + 3 * (MAX_PHOTO - MIN_PHOTO) / 4))

echo "========================================="
echo "照片编号配置验证"
echo "可用照片编号范围: $MIN_PHOTO - $MAX_PHOTO (共 $PHOTO_COUNT 张)"
echo "配置的照片编号: $USER_PHOTO_NUM"
echo "建议编号参考:"
echo "  - 1/4 位置: $QUARTER_1"
echo "  - 中间位置: $HALF"
echo "  - 3/4 位置: $QUARTER_3"
echo "========================================="

# 验证输入是否为数字
if ! [[ "$USER_PHOTO_NUM" =~ ^[0-9]+$ ]]; then
    echo "❌ 错误: 照片编号必须是有效的数字"
    echo "当前配置: $USER_PHOTO_NUM"
    echo "请重新设置正确的照片编号"
    exit 1
fi

# 验证输入范围
if [ "$USER_PHOTO_NUM" -lt "$MIN_PHOTO" ] || [ "$USER_PHOTO_NUM" -gt "$MAX_PHOTO" ]; then
    echo "❌ 错误: 照片编号必须在 $MIN_PHOTO-$MAX_PHOTO 范围内"
    echo "当前配置: $USER_PHOTO_NUM"
    echo "请重新设置正确的照片编号"
    exit 1
fi

# 检查对应的文件是否存在
PHOTO_FILE=$(printf "$TRAIN_DATA_DIR/r_%03d.png" "$USER_PHOTO_NUM")
if [ ! -f "$PHOTO_FILE" ]; then
    echo "❌ 错误: 照片文件不存在 $PHOTO_FILE"
    echo "请重新设置正确的照片编号"
    exit 1
fi

echo "✅ 照片编号验证通过: $USER_PHOTO_NUM"
echo "✅ 对应文件: $(basename "$PHOTO_FILE")"

# 计算相机编号范围 [USER_PHOTO_NUM-1, USER_PHOTO_NUM]
# 注意：相机编号从0开始，所以照片编号即为相机编号
CAMERA_START=$((USER_PHOTO_NUM - 1))
CAMERA_END=$USER_PHOTO_NUM

# 确保相机编号不小于0
if [ "$CAMERA_START" -lt 0 ]; then
    CAMERA_START=0
    CAMERA_END=1
fi

echo "计算的相机编号范围: [$CAMERA_START:$CAMERA_END]"

#### ——— 8. 备份并修改custom_render.py ———
echo "备份并修改custom_render.py..."

CUSTOM_RENDER_FILE="$PROJECT_ROOT/custom_render.py"

# 检查custom_render.py是否存在
if [ ! -f "$CUSTOM_RENDER_FILE" ]; then
    echo "❌ 错误: custom_render.py 文件不存在 $CUSTOM_RENDER_FILE"
    exit 1
fi

# 备份原始文件
BACKUP_FILE="$CUSTOM_RENDER_FILE.backup_$(date '+%Y%m%d_%H%M%S')"
cp "$CUSTOM_RENDER_FILE" "$BACKUP_FILE"
echo "✅ 已备份原始文件到: $BACKUP_FILE"

# 修改ply_dir参数默认值
echo "修改ply_dir参数..."
sed -i "s|default=r'/users/zchen27/SensorReconstruction/my_script/inference_outputs/experiment2/objects_world'|default=r'$INFERENCE_OUTPUT_DIR'|g" "$CUSTOM_RENDER_FILE"

# 修改cameras行
echo "修改cameras参数..."
# 查找并替换cameras行，支持各种可能的格式
sed -i "s/cameras = list(scene.getVideoCameras())\[[0-9]*:[0-9]*\]/cameras = list(scene.getVideoCameras())[$CAMERA_START:$CAMERA_END]/g" "$CUSTOM_RENDER_FILE"

echo "✅ custom_render.py 修改完成"

# 验证修改结果
echo "验证修改结果:"
echo "  ply_dir设置:"
grep -n "ply_dir.*default=" "$CUSTOM_RENDER_FILE" | head -1
echo "  cameras设置:"
grep -n "cameras = list" "$CUSTOM_RENDER_FILE" | head -1

#### ——— 9. 运行渲染命令 ———
echo "========================================="
echo "开始运行渲染命令..."
echo "  选择的照片: r_$(printf '%03d' $USER_PHOTO_NUM).png"
echo "  相机范围: [$CAMERA_START:$CAMERA_END]"
echo "  推理PLY文件: $PLY_COUNT 个"
echo "========================================="

# 设置输出视频文件名
OUTPUT_VIDEO="$PROJECT_ROOT/motion_video_${ACTION_NAME}_camera${USER_PHOTO_NUM}.mp4"

# 运行custom_render.py
echo "正在执行渲染..."
python custom_render.py \
    --model_path "$MODEL_PATH" \
    --source_path "$SOURCE_PATH" \
    --ply_dir "$INFERENCE_OUTPUT_DIR" \
    --out "$OUTPUT_VIDEO" \
    --fps 30 \
    --width 1920 \
    --height 1080 \
    --ffmpeg_crf 18 \
    --ffmpeg_preset slow \
    --bitrate 8000k

echo "✅ 渲染执行完成"

#### ——— 10. 验证渲染结果 ———
echo "验证渲染结果..."

if [ ! -f "$OUTPUT_VIDEO" ]; then
    echo "❌ 错误: 输出视频文件未生成 $OUTPUT_VIDEO"
    exit 1
fi

VIDEO_SIZE=$(stat -c%s "$OUTPUT_VIDEO")
VIDEO_SIZE_MB=$((VIDEO_SIZE / 1024 / 1024))

echo "✅ 视频文件生成成功:"
echo "  文件路径: $OUTPUT_VIDEO"
echo "  文件大小: ${VIDEO_SIZE_MB}MB"

# 检查帧序列目录
FRAMES_DIR="${OUTPUT_VIDEO%.*}_frames"
if [ -d "$FRAMES_DIR" ]; then
    FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.png" | wc -l)
    echo "  生成帧数: $FRAME_COUNT 帧"
    echo "  帧目录: $FRAMES_DIR"
fi

# 使用ffprobe获取视频信息（如果可用）
if command -v ffprobe >/dev/null 2>&1; then
    echo "视频信息:"
    ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_VIDEO" | grep -E '"duration"|"width"|"height"|"codec_name"' | head -6
fi

#### ——— 11. 恢复custom_render.py ———
echo "恢复custom_render.py原始配置..."

if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" "$CUSTOM_RENDER_FILE"
    echo "✅ 已恢复原始配置文件"
else
    echo "⚠️  警告: 备份文件不存在，无法恢复原始配置"
fi

#### ——— 12. 生成渲染报告 ———
REPORT_FILE="$PROJECT_ROOT/motion_video_${ACTION_NAME}_report.md"
echo "生成渲染报告: $REPORT_FILE"

cat > "$REPORT_FILE" << EOF
# 运动视频渲染完成报告

## 基本信息
- 动作名称: $ACTION_NAME
- 渲染时间: $(date '+%Y-%m-%d %H:%M:%S')
- 工作节点: $(hostname)
- 选择照片: r_$(printf '%03d' $USER_PHOTO_NUM).png (编号 $USER_PHOTO_NUM)

## 数据源信息
- 训练照片总数: $PHOTO_COUNT 张
- 照片编号范围: $MIN_PHOTO - $MAX_PHOTO
- 推理PLY文件: $PLY_COUNT 个
- 相机视角: [$CAMERA_START:$CAMERA_END]

## 渲染配置
- 模型路径: $MODEL_PATH
- 源数据路径: $SOURCE_PATH
- PLY目录: $INFERENCE_OUTPUT_DIR
- 输出视频: $OUTPUT_VIDEO

## 视频参数
- 分辨率: 1920×1080
- 帧率: 30 FPS
- 编码器: H.264 (libx264)
- CRF质量: 18 (高质量)
- 预设: slow
- 比特率: 8000k

## 输出结果
- 视频文件: $OUTPUT_VIDEO
- 文件大小: ${VIDEO_SIZE_MB}MB
- 帧序列: $FRAMES_DIR ($FRAME_COUNT 帧)

## 渲染质量分析
### 用户选择说明
- **选择编号**: $USER_PHOTO_NUM (在 $MIN_PHOTO-$MAX_PHOTO 范围内)
- **视角位置**: 基于训练数据中的第 $USER_PHOTO_NUM 号照片视角
- **建议对比**: 可尝试不同编号以获得最佳视觉效果

### 技术参数解释
- **相机映射**: 照片编号直接对应相机编号
- **视角连续性**: 使用[$CAMERA_START:$CAMERA_END]确保视角平滑过渡
- **PLY序列**: 基于推理结果生成的动态点云序列

## 后续操作建议

### 1. 查看渲染结果
\`\`\`bash
# 播放视频
vlc $OUTPUT_VIDEO

# 查看帧序列
ls -la $FRAMES_DIR
\`\`\`

### 2. 尝试不同视角
可以重新运行脚本并选择不同的照片编号：
- 前景视角: $QUARTER_1
- 中景视角: $HALF  
- 远景视角: $QUARTER_3

### 3. 调整渲染参数
可以修改以下参数以获得不同效果：
- 帧率: --fps (建议 24-60)
- 分辨率: --width --height
- 质量: --ffmpeg_crf (数值越低质量越高)
- 比特率: --bitrate

### 4. 批量渲染
如需生成多个视角的视频，可修改脚本支持批量处理。

## 故障排除

### 常见问题
1. **视频文件过大**: 降低比特率或分辨率
2. **质量不佳**: 降低CRF值或使用slower预设
3. **渲染时间过长**: 提高CRF值或使用faster预设
4. **内存不足**: 降低分辨率或减少并行度

### 性能优化建议
- 对于快速预览: CRF=28, preset=fast
- 对于高质量输出: CRF=15, preset=veryslow
- 对于平衡设置: CRF=20, preset=medium

EOF

echo "========================================="
echo "运动视频渲染完成"
echo "动作名称: $ACTION_NAME"
echo "选择照片: r_$(printf '%03d' $USER_PHOTO_NUM).png"
echo "输出视频: $OUTPUT_VIDEO"
echo "视频大小: ${VIDEO_SIZE_MB}MB"
echo "报告文件: $REPORT_FILE"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "✅ 运动视频渲染成功完成！" 