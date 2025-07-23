#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id traffic_Pyraformer_720 \
      --model Pyraformer \
      --data traffic \
      --dir_path ./data/traffic \
      --data_path traffic.csv \
      --freq h \
      --seq_len 720 \
      --label_len 360 \
      --pred_len 720 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 128 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > traffic_Pyraformer_720.txt 2>&1 &
