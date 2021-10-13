cd /home/byyoung/python_space/ALBEF

python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/Retrieval_flickr \
--checkpoint /data/ALBEF/flickr30k.pth \
--evaluate