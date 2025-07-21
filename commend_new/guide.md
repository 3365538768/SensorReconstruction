# 4DGaussians 简洁使用指南

## 🚀 一键启动

```bash
./commend_new/quick_start.sh
```

## 📋 手动执行

```bash
# 1. 数据预处理
qsub commend_new/data_preprocessing.sge.sh

# 2. 训练（等预处理完成）
export ACTION_NAME="walking_01"
qsub commend_new/train_4dgs.sge.sh

# 3. 推理（等训练完成）
qsub commend_new/inference_4dgs.sge.sh
```

## 📊 监控作业

```bash
qstat -u $USER                    # 查看作业状态
tail -f <script_name>.o<job_id>   # 查看日志
```

## 📁 前提条件

- 项目位于: `/users/$USER/SensorReconstruction/`
- 数据位于: `ECCV2022-RIFE/originframe/`
- 环境: `Gaussians4D` conda 环境

## 📈 输出结果

- 训练模型: `output/dnerf/{ACTION_NAME}/point_cloud/iteration_20000/`
- 渲染图像: `output/dnerf/{ACTION_NAME}/{train,test,video}/ours_20000/renders/`
- 逐帧模型: `output/dnerf/{ACTION_NAME}/gaussian_pertimestamp/`
