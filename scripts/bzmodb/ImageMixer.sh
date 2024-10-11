#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_ImageMixer \
      --model ImageMixer \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((288 * 2)) \
      --lw 3 \
      --channel 1 \
      --hidden_dim 32 \
      --patch_size 24 24 \
      --token_mlp_dim 288 \
      --channel_mlp_dim 32 \
      --n_blocks 3 \
      --dropout 0.1 \
      --use_multi_gpu \
      > bzmodb_ImageMixer.txt 2>&1 &
