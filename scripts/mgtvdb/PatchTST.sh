#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_PatchTST \
      --model PatchTST \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 16 \
		  --n_heads 4 \
		  --e_layers 3 \
		  --d_layers 1 \
		  --d_ff 128 \
      --use_multi_gpu \
      > mgtvdb_PatchTST.txt 2>&1 &
