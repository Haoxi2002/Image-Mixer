#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_MV_DTSF \
      --model MV_DTSF \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --h $((288 * 2)) \
      --lw 3 \
      --channel 1 \
      --hidden_dim 8 \
      --patch_size 24 24 \
      --token_mlp_dim 512 \
      --channel_mlp_dim 16 \
      --n_blocks 2 \
      --dropout 0 \
      --batch_size 64 \
      --use_multi_gpu \
      > mgtvdb_MV_DTSF.txt 2>&1 &
