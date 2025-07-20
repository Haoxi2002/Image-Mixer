#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id traffic_PatchTST_336 \
      --model PatchTST \
      --data traffic \
      --dir_path ./data/traffic \
      --data_path traffic.csv \
      --freq h \
      --seq_len 336 \
      --label_len 168 \
      --pred_len 336 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 64 \
      --learning_rate 0.005 \
      > traffic_PatchTST_336.txt 2>&1 &
