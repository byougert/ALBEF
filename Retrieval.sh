#!/bin/bash

cd /home/byyoung/python_space/ALBEF

export CUDA_VISIBLE_DEVICES=4,5,6,7

config='./configs/Retrieval_coco.yaml'
output_dir='output/Retrieval_coco'
checkpoint='/data/ALBEF/mscoco.pth'

retrieval='Retrieval.py'

if [ "$3" == 'topk' ]; then
  retrieval='Retrieval.py'
else
  retrieval='Retrieval_inv.py'
fi


if [ "$1" == 'flickr' ]; then
  output_dir='output/Retrieval_flickr'

  if [ "$3" == 'topk' ]; then
    config='./configs/Retrieval_flickr.yaml'
  else
    config='./configs/Retrieval_flickr_inv.yaml'
  fi

  if [ "$2" == 'finetune' ]; then
    checkpoint='/data/ALBEF/flickr.pth'
  else
    checkpoint='/data/ALBEF/mscoco.pth'
  fi

else
  output_dir='output/Retrieval_coco'
  if [ "$3" == 'topk' ]; then
    config='./configs/Retrieval_coco.yaml'
  else
    config='./configs/Retrieval_coco_inv.yaml'
  fi

  if [ "$2" == 'finetune' ]; then
    checkpoint='/data/ALBEF/mscoco.pth'
  else
    echo "NOT Support zero-shot on coco"
    exit 1
  fi
fi

python -m torch.distributed.launch --nproc_per_node=4 --use_env \
$retrieval \
--config $config \
--output_dir $output_dir \
--checkpoint $checkpoint \
--dist_url tcp://127.0.0.1:2222 \
--evaluate