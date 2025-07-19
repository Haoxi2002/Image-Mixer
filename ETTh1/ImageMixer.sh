#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ETTh1_08_ImageMixer \
      --model ImageMixer \
      --data ETTh1 \
      --dir_path ./data/ETT-small \
      --data_path ETTh1.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((48 * 2)) \
      --lw 1 \
      --channel 3 \
      --hidden_dim 16 \
      --patch_size 6 6 \
      --token_mlp_dim 512 \
      --channel_mlp_dim 32 \
      --n_blocks 6 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ETTh1_ImageMixer.txt 2>&1 &
