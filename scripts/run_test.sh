#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

python -m torch.distributed.launch --master_port 1238 --nproc_per_node=1 \
    test.py --config ${config} --weights ${weight} ${@:3}

    ./configs/ucf101/ucf_zero_shot_video.yaml ./checkpoints/k400-vitb-32-f16.pt
    /home/sergio/Documents/code/project/text4vis/configs/ucf101/ucf_zero_shot_video.yaml /home/sergio/Documents/code/project/text4vis/checkpoints/k400_vitb-32-f16.pt