#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_ImageMixer \
      --model ImageMixer \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((144)) \
      --lw 4 \
      --expand 1 \
      --hidden_dim 16 \
      --channel_mlp_dim 64 \
      --patch_size 48 48 \
      --token_mlp_dim 64 \
      --n_blocks 9 \
      --dropout 0 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > mgtvdb_ImageMixer.txt 2>&1 &
