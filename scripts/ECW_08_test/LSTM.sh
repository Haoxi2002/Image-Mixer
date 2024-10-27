#!/bin/bash

nohup python run.py \
      --is_training 0 \
      --task_id ECW_LSTM \
      --model LSTM \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newapp.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24 \
		  --inverse \
		  --d_model 1024 \
		  --e_layers 2 \
      --use_multi_gpu \
      > ECW_newapp_LSTM.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_LSTM \
      --model LSTM \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newmac.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24 \
		  --inverse \
		  --d_model 1024 \
		  --e_layers 2 \
      --use_multi_gpu \
      > ECW_newmac_LSTM.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_LSTM \
      --model LSTM \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_switch.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24 \
		  --inverse \
		  --d_model 1024 \
		  --e_layers 2 \
      --use_multi_gpu \
      > ECW_switch_LSTM.txt 2>&1 &
