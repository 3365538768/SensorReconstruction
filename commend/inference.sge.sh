#!/bin/bash
#$ -M lshou@nd.edu         # 作业状态邮件通知地址
#$ -m abe                  # 在作业开始（a）、结束（b）、中止（e）时发送邮件
#$ -pe smp 8               # 分配 1 个 CPU core (NSLOTS=1)
#$ -q gpu                  # 提交到 GPU 队列
#$ -l gpu_card=1           # 请求 1 张 GPU 卡
#$ -N job     # 作业名，可以改成你想要的名字

#### ——— 1. 加载模块 ———
module load cmake/3.22.1
module load cuda/11.8
module load intel/24.2

#### ——— 2. 设置 OpenMP 线程数 ———
# NSLOTS 由 SGE 根据 -pe smp 1 分配，本例是 1
export OMP_NUM_THREADS=$NSLOTS

#### ——— 3. 激活 Conda 环境 ———
# 假设你的 conda 已经在 ~/.bashrc 或模块里初始化过
# 如果之前执行过 'conda init bash'，下面这行会生效
source ~/.bashrc  

# 激活你要用的 conda 环境
conda activate Gaussians4D
#### ——— 4. 跳转到训练脚本所在目录 ———
# （可根据实际项目路径修改）
cd /users/lshou/4DGaussians/my_script
#### ——— 5. 运行 Python 脚本 ———
python  infer.py
#### ——— 6. 可选：打印日志路径等信息 ———
echo "Job finished at $(date)"
