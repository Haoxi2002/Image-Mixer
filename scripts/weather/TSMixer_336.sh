#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id weather_TSMixer_336 \
      --model TSMixer \
      --data weather \
      --dir_path ./data/weather \
      --data_path weather.csv \
      --freq 10min \
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
      > weather_TSMixer_336.txt 2>&1 &
