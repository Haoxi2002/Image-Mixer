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
      --hidden_dim 64 \
      --channel_mlp_dim 512 \
      --patch_size 16 16 \
      --token_mlp_dim 256 \
      --n_blocks 7 \
      --dropout 0 \
      --learning_rate 0.0015 \
      --use_multi_gpu \
      > bzmodb_ImageMixer.txt 2>&1 &