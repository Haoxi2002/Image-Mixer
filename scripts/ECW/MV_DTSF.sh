#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_08_MV_DTSF \
      --model MV_DTSF \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((48 * 2)) \
      --lw 0.5 \
      --channel 1 \
      --hidden_dim 8 \
      --patch_size 24 24 \
      --token_mlp_dim 512 \
      --channel_mlp_dim 16 \
      --n_blocks 2 \
      --dropout 0 \
      --use_multi_gpu \
      > ECW_08_MV_DTSF.txt 2>&1 &
