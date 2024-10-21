#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_PatchTST \
      --model PatchTST \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
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
      > bzmodb_08_PatchTST.txt 2>&1 &
