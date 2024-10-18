#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_Autoformer \
      --model Autoformer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --d_model 32 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 128 \
      --use_multi_gpu \
      > ECW_08_Autoformer.txt 2>&1 &
