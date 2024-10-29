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
      --h $((288)) \
      --lw 3 \
      --expand 1 \
      --hidden_dim 8 \
      --channel_mlp_dim 512 \
      --patch_size 16 16 \
      --token_mlp_dim 256 \
      --n_blocks 10 \
      --dropout 0.15 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > mgtvdb_ImageMixer.txt 2>&1 &
