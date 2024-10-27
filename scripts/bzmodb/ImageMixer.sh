#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_ImageMixer \
      --model ImageMixer \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((288)) \
      --lw 4 \
      --expand 1 \
      --hidden_dim 16 \
      --channel_mlp_dim 16 \
      --patch_size 12 12 \
      --token_mlp_dim 128 \
      --n_blocks 3 \
      --dropout 0.05 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > bzmodb_ImageMixer.txt 2>&1 &
