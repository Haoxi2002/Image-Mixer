#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTh1_08_Autoformer \
      --model Autoformer \
      --data ETTh1 \
      --dir_path ./data/ETT-small \
      --data_path ETTh1.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ETTh1_Autoformer.txt 2>&1 &
