#!/bin/bash
#$ -M $USER@nd.edu          # 自动使用当前用户邮箱
#$ -m abe                   # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 2                # 分配 2 个 CPU 核心
#$ -q gpu                   # 提交到 GPU 队列
#$ -l gpu_card=1            # 请求 1 张 GPU 卡
#$ -N cage_model_video      # 作业名称

set -e  # 遇到错误立即退出

echo "========================================="
echo "笼节点模型运动视频生成作业"
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
module load ffmpeg  # 加载ffmpeg用于视频生成

#### ——— 2. 设置环境变量 ———
export OMP_NUM_THREADS=$NSLOTS
export CUDA_HOME=/opt/crc/c/cuda/11.8

#### ——— 3. 激活 Conda 环境 ———
source ~/.bashrc
conda activate Gaussians4D

# 验证环境
echo "Python 环境验证:"
python --version
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 验证ffmpeg
echo "FFmpeg 版本:"
ffmpeg -version | head -1

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

#### ——— 5. 读取推理动作配置 ———
echo "读取推理动作名称配置..."

if [ ! -f "$PROJECT_ROOT/config/action_name.txt" ]; then
    echo "❌ 错误: config/action_name.txt 文件不存在！"
    echo "请先运行推理任意物体脚本"
    exit 1
fi

ACTION_NAME=$(cat "$PROJECT_ROOT/config/action_name.txt" | tr -d '[:space:]')
echo "推理动作名称: $ACTION_NAME"

# 设置关键路径
CAGES_OUTPUT_DIR="$MY_SCRIPT_DIR/inference_outputs/$ACTION_NAME/cages_pred"
SHOW_CAGE_FILE="$MY_SCRIPT_DIR/show_cage.py"

echo "笼预测目录: $CAGES_OUTPUT_DIR"
echo "show_cage.py文件: $SHOW_CAGE_FILE"

#### ——— 6. 检查笼预测结果 ———
echo "检查笼预测结果..."

if [ ! -d "$CAGES_OUTPUT_DIR" ]; then
    echo "❌ 错误: 笼预测目录不存在 $CAGES_OUTPUT_DIR"
    echo "请先运行推理任意物体脚本"
    exit 1
fi

CAGE_PLY_COUNT=$(find "$CAGES_OUTPUT_DIR" -name "*.ply" | wc -l)
echo "发现笼预测PLY文件数量: $CAGE_PLY_COUNT 个"

if [ "$CAGE_PLY_COUNT" -eq 0 ]; then
    echo "❌ 错误: 笼预测目录中没有PLY文件"
    echo "请确认推理任意物体步骤已完成"
    exit 1
fi

echo "✅ 笼预测结果检查通过"

# 检查show_cage.py文件
if [ ! -f "$SHOW_CAGE_FILE" ]; then
    echo "❌ 错误: show_cage.py 文件不存在 $SHOW_CAGE_FILE"
    exit 1
fi

echo "✅ show_cage.py文件检查通过"

#### ——— 7. 备份并修改show_cage.py ———
echo "备份并修改show_cage.py..."

# 备份原始文件
BACKUP_FILE="$SHOW_CAGE_FILE.backup_$(date '+%Y%m%d_%H%M%S')"
cp "$SHOW_CAGE_FILE" "$BACKUP_FILE"
echo "✅ 已备份原始文件到: $BACKUP_FILE"

# 修改plydir参数
echo "修改plydir参数..."
# 使用相对路径（因为我们会cd到my_script目录）
RELATIVE_CAGES_DIR="inference_outputs/$ACTION_NAME/cages_pred"

# 查找并替换plydir行
sed -i "s|ply_dir = r\".*\"|ply_dir = r\"$RELATIVE_CAGES_DIR\"|g" "$SHOW_CAGE_FILE"

echo "✅ show_cage.py 修改完成"

# 验证修改结果
echo "验证修改结果:"
grep -n "ply_dir = r" "$SHOW_CAGE_FILE"

#### ——— 8. 切换到my_script目录并运行 ———
echo "========================================="
echo "切换到my_script目录并运行show_cage.py..."
echo "工作目录: $MY_SCRIPT_DIR"
echo "笼预测文件: $CAGE_PLY_COUNT 个"
echo "========================================="

# 切换到my_script目录
cd "$MY_SCRIPT_DIR"

# 验证当前目录
echo "当前工作目录: $(pwd)"

# 验证文件路径
if [ ! -d "$RELATIVE_CAGES_DIR" ]; then
    echo "❌ 错误: 相对路径下的笼预测目录不存在 $RELATIVE_CAGES_DIR"
    exit 1
fi

echo "✅ 相对路径验证通过"

# 运行show_cage.py
echo "正在运行show_cage.py..."
python show_cage.py

echo "✅ show_cage.py 执行完成"

#### ——— 9. 验证输出结果 ———
echo "验证输出结果..."

OUTPUT_VIDEO="$RELATIVE_CAGES_DIR/cage_nodes_only.mp4"

if [ ! -f "$OUTPUT_VIDEO" ]; then
    echo "❌ 错误: 输出视频文件未生成 $OUTPUT_VIDEO"
    exit 1
fi

VIDEO_SIZE=$(stat -c%s "$OUTPUT_VIDEO")
VIDEO_SIZE_MB=$((VIDEO_SIZE / 1024 / 1024))

echo "✅ 笼节点视频生成成功:"
echo "  文件路径: $OUTPUT_VIDEO"
echo "  文件大小: ${VIDEO_SIZE_MB}MB"

# 使用ffprobe获取视频信息（如果可用）
if command -v ffprobe >/dev/null 2>&1; then
    echo "视频信息:"
    ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_VIDEO" | grep -E '"duration"|"width"|"height"|"codec_name"' | head -6
fi

#### ——— 10. 恢复show_cage.py ———
echo "恢复show_cage.py原始配置..."

if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" "$SHOW_CAGE_FILE"
    echo "✅ 已恢复原始配置文件"
else
    echo "⚠️  警告: 备份文件不存在，无法恢复原始配置"
fi

#### ——— 11. 生成笼节点视频报告 ———
cd "$PROJECT_ROOT"  # 回到项目根目录

REPORT_FILE="$PROJECT_ROOT/cage_model_video_${ACTION_NAME}_report.md"
echo "生成笼节点视频报告: $REPORT_FILE"

cat > "$REPORT_FILE" << EOF
# 笼节点模型运动视频生成完成报告

## 基本信息
- 动作名称: $ACTION_NAME
- 生成时间: $(date '+%Y-%m-%d %H:%M:%S')
- 工作节点: $(hostname)
- 脚本类型: 笼节点模型可视化

## 数据源信息
- 笼预测PLY文件: $CAGE_PLY_COUNT 个
- 数据来源: 推理任意物体步骤输出
- PLY文件目录: my_script/inference_outputs/$ACTION_NAME/cages_pred/

## 视频生成配置
- 脚本文件: my_script/show_cage.py
- 工作目录: my_script/
- 修改参数: plydir → inference_outputs/$ACTION_NAME/cages_pred
- 输出格式: MP4视频

## 可视化参数
- 动画帧率: 6 FPS
- 点大小: 2像素
- 点颜色: 蓝色
- 渲染DPI: 300
- 图形大小: 10×10英寸

## 输出结果
- 视频文件: my_script/inference_outputs/$ACTION_NAME/cages_pred/cage_nodes_only.mp4
- 文件大小: ${VIDEO_SIZE_MB}MB
- 动画帧数: $CAGE_PLY_COUNT 帧

## 技术说明

### 笼节点模型
- **定义**: 控制物体变形的稀疏控制点集合
- **作用**: 通过移动笼节点实现物体的非刚性变形
- **可视化**: 展示笼节点在时间序列中的运动轨迹

### 视频生成流程
1. **数据读取**: 从PLY文件中提取笼节点坐标
2. **坐标系统一**: 计算全局坐标范围确保视角一致
3. **动画生成**: 使用matplotlib生成3D散点图动画
4. **视频编码**: 使用FFMpegWriter输出MP4视频

### 关键特性
- **时序连续性**: 笼节点运动轨迹平滑连贯
- **空间一致性**: 固定视角范围避免视角跳跃
- **高质量输出**: 300 DPI确保细节清晰

## 应用场景

### 1. 变形分析
- 观察笼节点的运动模式
- 分析变形的时空特征
- 验证笼节点模型的有效性

### 2. 调试工具
- 检查笼节点训练结果
- 识别异常的笼节点运动
- 优化笼节点配置参数

### 3. 演示展示
- 可视化笼节点模型概念
- 展示变形控制机制
- 制作技术演示材料

## 后续操作建议

### 1. 查看笼节点视频
\`\`\`bash
# 播放视频
vlc my_script/inference_outputs/$ACTION_NAME/cages_pred/cage_nodes_only.mp4

# 或使用其他播放器
mpv my_script/inference_outputs/$ACTION_NAME/cages_pred/cage_nodes_only.mp4
\`\`\`

### 2. 对比分析
可以与第5步生成的物体运动视频进行对比：
- 物体变形视频: motion_video_${ACTION_NAME}_camera编号.mp4
- 笼节点运动视频: cage_nodes_only.mp4

### 3. 参数调整
如需调整可视化效果，可修改show_cage.py中的参数：
- 帧率: writer = FFMpegWriter(fps=6) 
- 点大小: s=2
- 点颜色: c='blue'
- DPI: dpi=300

### 4. 批量处理
如需处理多个动作的笼节点视频，可修改脚本支持批量模式。

## 故障排除

### 常见问题
1. **视频文件过小**: 检查PLY文件是否正确加载
2. **动画卡顿**: 尝试降低帧率或减少点数
3. **文件未生成**: 确认matplotlib和ffmpeg环境正确
4. **路径错误**: 验证相对路径设置是否正确

### 性能优化建议
- 对于大量笼节点: 考虑降采样显示
- 对于高质量输出: 增加DPI和图形尺寸
- 对于快速预览: 降低帧率和DPI

## 技术细节

### 文件格式
- **输入**: PLY格式的笼节点坐标文件
- **输出**: MP4格式的动画视频
- **中间格式**: matplotlib 3D散点图序列

### 坐标处理
- **全局范围**: 基于所有帧计算统一的坐标轴范围
- **坐标轴**: 隐藏坐标轴标记以突出笼节点运动
- **视角固定**: 使用一致的3D视角避免视觉干扰

EOF

echo "========================================="
echo "笼节点模型运动视频生成完成"
echo "动作名称: $ACTION_NAME"
echo "输出视频: my_script/inference_outputs/$ACTION_NAME/cages_pred/cage_nodes_only.mp4"
echo "视频大小: ${VIDEO_SIZE_MB}MB"
echo "笼节点数据: $CAGE_PLY_COUNT 帧"
echo "报告文件: $REPORT_FILE"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "✅ 笼节点模型运动视频生成成功完成！" 