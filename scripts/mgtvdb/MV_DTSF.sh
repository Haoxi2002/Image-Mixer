#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_MV_DTSF \
      --model MV_DTSF \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((288)) \
      --lw 2 \
      --expand 1 \
      --channel 1 \
      --use_multi_gpu \
      > mgtvdb_MV_DTSF.txt 2>&1 &
