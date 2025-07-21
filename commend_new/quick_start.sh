#!/bin/bash
# quick_start.sh - 4DGaussians SGE 脚本快速开始演示

set -e

echo "=== 4DGaussians SGE 脚本快速开始演示 ==="
echo "执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "执行用户: $USER"
echo ""

# 检查基本环境
echo "🔍 检查基本环境..."

if [ ! -d "/users/$USER/SensorReconstruction" ]; then
    echo "❌ 错误: 项目目录不存在"
    echo "请确保项目位于: /users/$USER/SensorReconstruction"
    exit 1
fi

if [ ! -d "ECCV2022-RIFE/originframe" ]; then
    echo "❌ 错误: 未找到 originframe 数据"
    echo "请确保 Blender 输出数据已放置在 ECCV2022-RIFE/originframe/"
    exit 1
fi

echo "✅ 基本环境检查通过"
echo ""

# 显示使用说明
echo "📋 SGE 脚本使用流程:"
echo ""
echo "步骤 1: 数据预处理"
echo "  qsub commend_new/data_preprocessing.sge.sh"
echo ""
echo "步骤 2: 设置动作名称并训练"
echo "  export ACTION_NAME=\"your_action_name\""
echo "  qsub commend_new/train_4dgs.sge.sh"
echo ""
echo "步骤 3: 推理测试"
echo "  qsub commend_new/inference_4dgs.sge.sh"
echo ""

# 自动执行数据预处理（移除用户交互）
echo "🚀 自动开始执行数据预处理..."
echo ""

if command -v qsub &> /dev/null; then
    echo "✅ 检测到 SGE 环境，提交数据预处理作业..."
    job_id=$(qsub commend_new/data_preprocessing.sge.sh)
    echo "✅ 数据预处理作业已提交: $job_id"
    echo ""
    echo "📊 监控作业状态:"
    echo "  qstat -u $USER"
    echo ""
    echo "📄 查看作业日志:"
    echo "  tail -f data_preprocessing.o*"
    echo ""
    echo "⏭️  等数据预处理完成后，运行训练:"
    echo "  export ACTION_NAME=\"your_action_name\""
    echo "  qsub commend_new/train_4dgs.sge.sh"
    echo ""
    echo "📖 或者使用默认动作名称:"
    echo "  qsub commend_new/train_4dgs.sge.sh"
else
    echo "⚠️  警告: qsub 命令不可用"
    echo "请在 CRC 集群的提交节点上运行此脚本"
    echo ""
    echo "🔧 手动提交命令:"
    echo "  qsub commend_new/data_preprocessing.sge.sh"
    echo ""
    echo "📖 完整流程:"
    echo ""
    echo "1. 数据预处理:"
    echo "   qsub commend_new/data_preprocessing.sge.sh"
    echo ""
    echo "2. 训练 (等数据预处理完成后):"
    echo "   export ACTION_NAME=\"walking_01\"  # 替换为你的动作名称"
    echo "   qsub commend_new/train_4dgs.sge.sh"
    echo ""
    echo "3. 推理 (等训练完成后):"
    echo "   qsub commend_new/inference_4dgs.sge.sh"
    echo ""
    echo "4. 监控作业:"
    echo "   qstat -u $USER"
    echo "   tail -f <script_name>.o<job_id>"
fi

echo ""
echo "📚 更多信息请查看:"
echo "  - 详细使用指南: cat commend_new/README.md"
echo "  - 交互式流程: instruction/auto.md"
echo "  - 项目文档: development_record.md"
echo ""
echo "✅ 快速开始演示完成，数据预处理已自动启动" 