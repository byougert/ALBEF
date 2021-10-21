#!/bin/bash

cd /home/byyoung/python_space/ALBEF

#export CUDA_VISIBLE_DEVICES=4,5,6,7

config='./configs/Retrieval_flickr.yaml'
output_dir='output/Retrieval_flickr'
checkpoint='/data/ALBEF/ALBEF.pth'

retrieval='Retrieval.py'

python -m torch.distributed.launch --nproc_per_node=4 --use_env \
$retrieval \
--config $config \
--output_dir $output_dir \
--checkpoint $checkpoint \
--dist_url tcp://127.0.0.1:2222