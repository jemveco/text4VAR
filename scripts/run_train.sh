#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
python -m torch.distributed.launch --master_port 3333 --nproc_per_node=1 \
         train.py  --config ${config} --log_time $now