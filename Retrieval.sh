cd /home/byyoung/python_space/ALBEF

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch --nproc_per_node=4 --use_env \
Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir output/Retrieval_coco \
--checkpoint /data/ALBEF/mscoco.pth \
--dist_url tcp://127.0.0.1:2222 \
--evaluate