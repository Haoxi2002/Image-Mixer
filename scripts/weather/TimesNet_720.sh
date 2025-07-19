#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id weather_TimesNet_720 \
      --model TimesNet \
      --data weather \
      --dir_path ./data/weather \
      --data_path weather.csv \
      --freq 10min \
      --seq_len 720 \
      --label_len 360 \
      --pred_len 720 \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 32 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > weather_TimesNet_720.txt 2>&1 &
