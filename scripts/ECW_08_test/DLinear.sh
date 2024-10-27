#!/bin/bash

nohup python run.py \
      --is_training 0 \
      --task_id ECW_DLinear \
      --model DLinear \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newapp.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --d_model 512 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_newapp_DLinear.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_DLinear \
      --model DLinear \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newmac.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --d_model 512 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_newmac_DLinear.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_DLinear \
      --model DLinear \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_switch.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --d_model 512 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_switch_DLinear.txt 2>&1 &
