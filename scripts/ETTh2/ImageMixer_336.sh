#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTh2_ImageMixer_336 \
      --model ImageMixer \
      --data ETTh2 \
      --dir_path ./data/ETT-small \
      --data_path ETTh2.csv \
      --freq h \
      --seq_len 336 \
      --label_len 168 \
      --pred_len 336 \
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
      > ETTh2_ImageMixer_336.txt 2>&1 &
