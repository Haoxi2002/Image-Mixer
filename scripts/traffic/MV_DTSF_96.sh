#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id traffic_MV_DTSF_96 \
      --model MV_DTSF \
      --data traffic \
      --dir_path ./data/traffic \
      --data_path traffic.csv \
      --freq h \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --h 24 \
      --lw 1 \
      --channel 3 \
      --hidden_dim 16 \
      --patch_size 8 8 \
      --token_mlp_dim 256 \
      --channel_mlp_dim 128 \
      --n_blocks 6 \
      --dropout 0 \
      --batch_size 32 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > traffic_MV_DTSF_96.txt 2>&1 &
