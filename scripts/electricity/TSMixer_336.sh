#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id electricity_TSMixer_336 \
      --model TSMixer \
      --data electricity \
      --dir_path ./data/electricity \
      --data_path electricity.csv \
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
      --use_multi_gpu \
      > electricity_TSMixer_336.txt 2>&1 &
