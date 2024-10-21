#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_KAE_Informer \
      --model KAE_Informer \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 512 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 2048 \
		  --factor 5 \
      --use_multi_gpu \
      > bzmodb_KAE_Informer.txt 2>&1 &
