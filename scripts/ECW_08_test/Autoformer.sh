#!/bin/bash

nohup python run.py \
      --is_training 0 \
      --task_id ECW_Autoformer \
      --model Autoformer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newapp.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
		  --d_model 512 \
		  --n_heads 8 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_newapp_Autoformer.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_Autoformer \
      --model Autoformer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newmac.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
		  --d_model 512 \
		  --n_heads 8 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_newmac_Autoformer.txt 2>&1 &

nohup python run.py \
      --is_training 0 \
      --task_id ECW_Autoformer \
      --model Autoformer \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_switch.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
		  --d_model 512 \
		  --n_heads 8 \
		  --d_ff 2048 \
      --use_multi_gpu \
      > ECW_switch_Autoformer.txt 2>&1 &