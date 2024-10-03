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
      --h $((288 * 2)) \
      --lw 4 \
      --channel 1 \
      --hidden_dim 8 \
      --patch_size 24 24 \
      --token_mlp_dim 512 \
      --channel_mlp_dim 16 \
      --n_blocks 2 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > bzmodb_ImageMixer.txt 2>&1 &
