#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_08_Autoformer \
      --model Autoformer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --use_multi_gpu \
      > ECW_08_Autoformer.txt 2>&1 &
