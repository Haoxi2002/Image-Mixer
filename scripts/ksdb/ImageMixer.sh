#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ksdb_ImageMixer \
      --model ImageMixer \
      --data ksdb \
      --dir_path ./data/PPIO \
      --data_path ksdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((288 * 2)) \
      --lw 3 \
      --expand 1 \
      --channel 1 \
      --hidden_dim 32 \
      --channel_mlp_dim 128 \
      --patch_size 16 16 \
      --token_mlp_dim 256 \
      --n_blocks 7 \
      --dropout 0.05 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > ksdb_ImageMixer.txt 2>&1 &
