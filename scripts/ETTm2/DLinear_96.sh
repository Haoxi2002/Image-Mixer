#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTm2_DLinear_96 \
      --model DLinear \
      --data ETTm2 \
      --dir_path ./data/ETT-small \
      --data_path ETTm2.csv \
      --freq 15min \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 128 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ETTm2_DLinear_96.txt 2>&1 &
