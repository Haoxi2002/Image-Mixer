#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id weather_DLinear_96 \
      --model DLinear \
      --data weather \
      --dir_path ./data/weather \
      --data_path weather.csv \
      --freq 10min \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 64 \
      --learning_rate 0.005 \
      > weather_DLinear_96.txt 2>&1 &
