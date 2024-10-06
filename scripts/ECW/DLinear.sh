#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_08_DLinear \
      --model DLinear \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > ECW_08_DLinear.txt 2>&1 &
