#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id electricity_TSMixer_192 \
      --model TSMixer \
      --data electricity \
      --dir_path ./data/electricity \
      --data_path electricity.csv \
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
      > electricity_TSMixer_192.txt 2>&1 &
