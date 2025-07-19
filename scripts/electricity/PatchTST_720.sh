#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id electricity_PatchTST_720 \
      --model PatchTST \
      --data electricity \
      --dir_path ./data/electricity \
      --data_path electricity.csv \
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
      > electricity_PatchTST_720.txt 2>&1 &
