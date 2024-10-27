#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_ImageMixer \
      --model ImageMixer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newapp.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --hidden_dim 16 \
      --channel_mlp_dim 128 \
      --patch_size 8 8 \
      --token_mlp_dim 256 \
      --n_blocks 6 \
      --dropout 0.15 \
      --learning_rate 0.003 \
      --use_multi_gpu \
      > ECW_newapp_ImageMixer.txt 2>&1 &

nohup python run.py \
      --is_training 1 \
      --task_id ECW_ImageMixer \
      --model ImageMixer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newmac.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --hidden_dim 16 \
      --channel_mlp_dim 128 \
      --patch_size 8 8 \
      --token_mlp_dim 256 \
      --n_blocks 6 \
      --dropout 0.15 \
      --learning_rate 0.003 \
      --use_multi_gpu \
      > ECW_newmac_ImageMixer.txt 2>&1 &

nohup python run.py \
      --is_training 1 \
      --task_id ECW_ImageMixer \
      --model ImageMixer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_switch.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --hidden_dim 16 \
      --channel_mlp_dim 128 \
      --patch_size 8 8 \
      --token_mlp_dim 256 \
      --n_blocks 6 \
      --dropout 0.15 \
      --learning_rate 0.003 \
      --use_multi_gpu \
      > ECW_switch_ImageMixer.txt 2>&1 &