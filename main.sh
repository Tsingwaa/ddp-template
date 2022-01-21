# export PYTHONPATH=$PYTHONPATH:root/to/your_project

# Distributed Training
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnode=1 --nproc_per_node=2 --master_addr 127.0.0.111 --master_port 30000 main.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnode=1 --nproc_per_node=2 main.py

# Single-GPU Training
# CUDA_VISIBLE_DEVICES=0 python3 main.py --local_rank -1
