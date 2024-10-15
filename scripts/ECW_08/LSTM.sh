#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_LSTM \
      --model LSTM \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24 \
		  --inverse \
		  --d_model 2048 \
		  --e_layers 3 \
      --use_multi_gpu \
      > ECW_08_LSTM.txt 2>&1 &
