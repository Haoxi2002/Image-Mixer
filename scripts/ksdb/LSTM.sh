#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ksdb_LSTM \
      --model LSTM \
      --data ksdb \
      --dir_path ./data/PPIO \
      --data_path ksdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 1024 \
		  --e_layers 2 \
      --use_multi_gpu \
      > ksdb_LSTM.txt 2>&1 &
