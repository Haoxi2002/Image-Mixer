#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_TimesNet \
      --model TimesNet \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 16 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --factor 3 \
		  --d_ff 32 \
		  --top_k 5 \
      --use_multi_gpu \
      > mgtvdb_TimesNet.txt 2>&1 &
