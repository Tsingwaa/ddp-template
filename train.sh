export PYTHONPATH=$PYTHONPATH:root/to/your_project

# Distributed Training
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 --master_addr 127.0.0.111 --master_port 30000 train.py

# Single-GPU Training
CUDA_VISIBLE_DEVICES=0 python3 train.py --local_rank -1
