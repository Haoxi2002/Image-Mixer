#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTm1_ImageMixer_96 \
      --model ImageMixer \
      --data ETTm1 \
      --dir_path ./data/ETT-small \
      --data_path ETTm1.csv \
      --freq 15min \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
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
      > ETTm1_ImageMixer_96.txt 2>&1 &
