# export PYTHONPATH=$PYTHONPATH:root/to/your_project

# Distributed Training
# 设置OMP_NUM_THREADS 是为了不让CPU炸
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnode=1 --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 main.py
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 30000 main.py

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=0 python3 main.py --local_rank -1
