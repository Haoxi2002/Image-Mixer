#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id traffic_TimesNet_192 \
      --model TimesNet \
      --data traffic \
      --dir_path ./data/traffic \
      --data_path traffic.csv \
      --freq h \
      --seq_len 192 \
      --label_len 96 \
      --pred_len 192 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 64 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > traffic_TimesNet_192.txt 2>&1 &
