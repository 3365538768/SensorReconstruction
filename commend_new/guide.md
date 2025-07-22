# 4DGaussians 简洁使用指南

## 🚀 一键启动（别用，还没搞好）

```bash
./commend_new/quick_start.sh
```

## 📋 手动执行

```bash
# 1. 数据预处理（ECCV插帧）
# blender导出的文件夹已经改名为originframe并放在在ECCV2022-RIFE/下
qsub commend_new/data_preprocessing.sge.sh

# 2. 4DGaussians训练（等预处理完成）
#命令行输入
read -p "请输入动作名称（如 walking_01, jumping_02）: " ACTION_NAME
echo "$ACTION_NAME" > config/action_name.txt
qsub commend_new/train_4dgs.sge.sh

# 3. 训练笼节点模型（等4DGaussians训练完成）
# 第一步：数据准备和动态点筛选
qsub commend_new/cage_data_preparation.sge.sh

# 第二步：本地Windows端框选笼节点（等数据准备完成）
# 在本地Windows环境中运行
cd my_script/user && python user.py

#在my_script/data/{ACTION_NAME}/路径下
#导入region.json
#导入sensor.csv

# 第三步：笼节点模型训练（等本地处理完成）
qsub commend_new/cage_model_training.sge.sh

# 4. 推理任意物体（等笼节点模型训练完成）
# 第一步：静态场景数据准备与训练（整合步骤1+2的修改版）
qsub commend_new/static_inference_preparation.sge.sh

cd my_script/user && python user.py
# 在my_script/data/{ACTION_NAME}/路径下
#导入region.json
#导入sensor.csv

# 第二步：推理执行（等静态准备完成）
qsub commend_new/static_inference_execution.sge.sh

# 5. 渲染运动视频（等推理完成）
#命令行输入
read -p "请输入照片编号（0-688范围内，如 344）: " CAMERA_NUMBER
echo "$CAMERA_NUMBER" > config/camera_number.txt
# 注意：此脚本会读取配置文件中的照片编号
qsub commend_new/render_motion_video.sge.sh

# 6. 生产笼节点模型运动视频（等渲染完成）
# 基于笼节点模型的专用运动视频
qsub commend_new/cage_model_video.sge.sh
```

### 混合流程：服务器+本地处理

```bash
# 1. 服务器端：数据筛选和准备
./commend_new/lightweight_cage_training.sh walking_01
# (脚本会在需要本地处理时暂停)

# 2. 本地Windows端：框选笼节点
# - 启动: D:\4DGaussians\my_script\user\user.py
# - 访问: http://localhost:8050
# - 生成: region.json

# 3. 继续服务器端训练
./commend_new/lightweight_cage_training.sh walking_01 continue
```

## 📊 监控作业

```bash
qstat -u $USER                    # 查看作业状态
tail -f $(qstat -u $USER | grep " r " | awk '{print $3".o"$1}' | head -1)   # 自动查看运行中任务的日志
```

## 📁 前提条件

### 4DGaussians 标准流程

- 项目位于: `/users/$USER/SensorReconstruction/`
- 数据位于: `ECCV2022-RIFE/originframe/`
- 环境: `Gaussians4D` conda 环境

### 轻量笼节点模型训练

- 项目位于: `/users/$USER/SensorReconstruction/`
- 4DGaussians 已完成，存在: `output/dnerf/{SCENE_NAME}/gaussian_pertimestamp/`
- 环境: `Gaussians4D` conda 环境
- 本地环境: Windows 端 `D:\4DGaussians\my_script\user\user.py` 可用
- 依赖: `dash plotly plyfile numpy torch dash-bootstrap-components`

## 📈 输出结果

### 4DGaussians 标准训练输出

- 训练模型: `output/dnerf/{ACTION_NAME}/point_cloud/iteration_20000/`
- 渲染图像: `output/dnerf/{ACTION_NAME}/{train,test,video}/ours_20000/renders/`
- 逐帧模型: `output/dnerf/{ACTION_NAME}/gaussian_pertimestamp/`

### 轻量笼节点模型训练输出

- 数据目录: `my_script/data/{SCENE_NAME}/`
  - `frames/` - 筛选后的动态点云
  - `region.json` - 笼节点区域定义
  - `sensor.csv` - 传感器数据
- 训练模型: `outputs/{SCENE_NAME}/deform_model_final.pth`
- 推理结果: `inference_outputs/{SCENE_NAME}/`
- 可视化视频: `{SCENE_NAME}.mp4`
