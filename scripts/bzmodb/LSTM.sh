#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_LSTM \
      --model LSTM \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 1024 \
		  --e_layers 2 \
      --use_multi_gpu \
      > bzmodb_LSTM.txt 2>&1 &
