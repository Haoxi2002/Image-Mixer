#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_08_ImageMixer \
      --model ImageMixer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((48 * 2)) \
      --lw 1 \
      --channel 3 \
      --hidden_dim 16 \
      --patch_size 6 6 \
      --token_mlp_dim 512 \
      --channel_mlp_dim 32 \
      --n_blocks 6 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ECW_08_ImageMixer.txt 2>&1 &
