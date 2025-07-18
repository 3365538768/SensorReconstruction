# 4DGaussians 环境设置指南

## 📋 环境配置文件说明

### 主要配置文件
- **`4DGaussians_environment.yml`** - 4DGaussians项目专用conda环境配置
- **`../my_environment.yml`** - 原始项目环境配置（位于项目根目录）
- **`../requirements.txt`** - pip依赖列表（位于项目根目录）

## 🚀 快速开始

### 方法1: 使用4DGaussians专用配置 (推荐)
```bash
# 创建4DGaussians环境
conda env create -f docs/4DGaussians_environment.yml

# 激活环境
conda activate 4DGaussians

# 验证安装
python -c "import torch, open3d, mmcv; print('✅ Environment ready!')"
```

### 方法2: 使用原始配置
```bash
# 使用原始环境配置
conda env create -f my_environment.yml

# 激活环境
conda activate NeRF

# 手动安装额外依赖
pip install simple-knn lpips
```

## 📦 主要依赖包说明

### 核心框架
- **PyTorch 1.13.1** - 深度学习框架
- **OpenCV** - 计算机视觉库
- **Open3D 0.17.0** - 3D数据处理
- **NumPy 1.21.6** - 数值计算

### 4DGaussians专用库
- **mmcv 1.6.0** - 计算机视觉工具链
- **simple-knn** - KNN计算加速
- **lpips** - 感知损失计算
- **plyfile** - PLY文件处理
- **pytorch_msssim** - 结构相似性损失

### 辅助工具
- **matplotlib** - 数据可视化
- **scikit-image** - 图像处理
- **imageio[ffmpeg]** - 视频处理
- **tensorboard** - 训练监控

## ⚠️ 注意事项

### GPU支持
- 当前配置包含CUDA支持的PyTorch
- 如果系统无GPU，某些功能可能受限
- `diff-gaussian-rasterization`需要CUDA环境编译

### 常见问题解决
```bash
# 如果遇到CUDA相关错误
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu

# 如果缺少rasterization模块
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

# 验证环境完整性
python -c "
import torch
import open3d
import mmcv
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Open3D: {open3d.__version__}')
print(f'MMCV: {mmcv.__version__}')
"
```

## 📚 相关文档
- **项目初始化报告**: `../PROJECT_INIT_SUMMARY.md`
- **开发记录**: `../development_record.md`
- **项目目标**: `../objective.md`
- **主要README**: `../README.md` 