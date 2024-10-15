#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_ImageMixer \
      --model ImageMixer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((48)) \
      --hidden_dim 16 \
      --channel_mlp_dim 32 \
      --patch_size 6 6 \
      --token_mlp_dim 256 \
      --n_blocks 2 \
      --dropout 0 \
      --learning_rate 0.0003 \
      --use_multi_gpu \
      > ECW_08_ImageMixer.txt 2>&1 &
