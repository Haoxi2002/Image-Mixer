#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id electricity_ImageMixer_720 \
      --model ImageMixer \
      --data electricity \
      --dir_path ./data/electricity \
      --data_path electricity.csv \
      --freq h \
      --seq_len 720 \
      --label_len 360 \
      --pred_len 720 \
      --h 24 \
      --lw 1 \
      --channel 3 \
      --hidden_dim 16 \
      --patch_size 8 8 \
      --token_mlp_dim 256 \
      --channel_mlp_dim 128 \
      --n_blocks 6 \
      --dropout 0 \
      --batch_size 64 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > electricity_ImageMixer_720.txt 2>&1 &
