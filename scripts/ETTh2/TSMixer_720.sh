#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTh2_TSMixer_720 \
      --model TSMixer \
      --data ETTh2 \
      --dir_path ./data/ETT-small \
      --data_path ETTh2.csv \
      --freq h \
      --seq_len 720 \
      --label_len 360 \
      --pred_len 720 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 64 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ETTh2_TSMixer_720.txt 2>&1 &
