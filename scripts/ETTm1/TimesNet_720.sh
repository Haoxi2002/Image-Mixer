#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTm1_TimesNet_720 \
      --model TimesNet \
      --data ETTm1 \
      --dir_path ./data/ETT-small \
      --data_path ETTm1.csv \
      --freq 15min \
      --seq_len 720 \
      --label_len 360 \
      --pred_len 720 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 32 \
      --learning_rate 0.005 \
      > ETTm1_TimesNet_720.txt 2>&1 &
