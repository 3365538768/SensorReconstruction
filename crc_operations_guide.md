# Notre Dame CRC 快速命令手册

快速查找复制 CRC 集群常用命令。
# 查看存储配额
quota                                    # AFS永久存储
pan_df -H /scratch365/zchen27           # 临时存储

# 加载软件环境
module load conda
```

## 🐍 Conda 环境管理

```bash
# 环境操作
conda create -n myenv python=3.10 -y   # 创建环境
conda activate myenv                    # 激活环境
conda deactivate                        # 退出环境
conda env list                          # 查看所有环境
conda list                              # 查看当前环境包

# 包管理
conda install numpy pandas matplotlib -y
pip install package_name
conda env export --from-history > environment.yml
```
---

## 🎯 GPU 资源管理

### 检查 GPU 可用性

```bash
free_gpus.sh @crc_gpu                   # 查看GPU节点空闲情况
free_nodes.sh -G                        # 查看空闲CPU节点
nvidia-smi                              # 查看当前节点GPU状态
```

### 交互式 GPU 会话

```bash
# 申请交互式GPU
qrsh -q gpu -l gpu_card=1 -pe smp 8
# 使用完毕释放资源
exit
```

### 批处理 GPU 作业

```bash
# 创建作业脚本 gpu_job.sh
cat > gpu_job.sh << 'EOF'
#!/bin/bash
#$ -M zchen27@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N my_gpu_job

module load conda
conda activate myenv
python train.py
EOF

# 提交和管理作业
qsub gpu_job.sh                         # 提交作业
qstat -u zchen27                        # 查看我的作业
qstat -j <job_id>                       # 查看作业详情
qdel <job_id>                           # 删除作业
```

---

## 📊 系统监控

```bash
# 资源监控
ps aux | grep zchen27                   # 查看我的进程
top                                     # 实时系统监控
htop                                    # 交互式监控
qhost -h <hostname>                     # 查看节点信息
```

---

## 📁 文件传输

```bash
# 上传文件到CRC
scp local_file.zip zchen27@crcfe02.crc.nd.edu:~/
rsync -avP local_dir/ zchen27@crcfe02.crc.nd.edu:~/remote_dir/

# 从CRC下载文件
scp zchen27@crcfe02.crc.nd.edu:~/remote_file.zip ./
rsync -avP zchen27@crcfe02.crc.nd.edu:~/remote_dir/ ./local_dir/
```

---

## 🔧 常见问题解决

```bash
# conda命令找不到
module load conda

# 检查作业为什么失败
qstat -j <job_id>

# 查看作业输出
cat <job_name>.o<job_id>
cat <job_name>.e<job_id>

# 强制删除卡住的作业
qdel -f <job_id>
```

---

## 📋 快速启动流程

```bash
# 1. 登录
ssh zchen27@crcfe01.crc.nd.edu

# 2. 设置环境
module load conda
conda activate myproject

# 3. 检查GPU资源
gpu-free

# 4. 申请GPU测试
gpu-qrsh
nvidia-smi
exit

# 5. 提交正式作业
qsub gpu_job.sh
myjobs
```

---

## 📝 作业脚本模板

### 基础 GPU 作业

```bash
#!/bin/bash
#$ -M your_email@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N job_name

module load conda
conda activate env_name
python your_script.py
```

### 多 GPU 作业

```bash
#!/bin/bash
#$ -M your_email@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=2
#$ -N multi_gpu_job

module load conda
conda activate env_name
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

---

_最后更新：2025-07-16 | 维护者：zchen27_
