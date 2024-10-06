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
      --h $((288 * 2)) \
      --lw 3 \
      --channel 1 \
      --hidden_dim 4 \
      --patch_size 36 36 \
      --token_mlp_dim 256 \
      --channel_mlp_dim 8 \
      --n_blocks 2 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > mgtvdb_ImageMixer.txt 2>&1 &
