#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_Autoformer \
      --model Autoformer \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
		  --d_model 512 \
		  --n_heads 8 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > mgtvdb_Autoformer.txt 2>&1 &
