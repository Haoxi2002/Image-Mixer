#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id bzmodb_Pyraformer \
      --model Pyraformer \
      --data bzmodb \
      --dir_path ./data/PPIO \
      --data_path bzmodb.csv \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.005 \
      --use_multi_gpu \
      > bzmodb_Pyraformer.txt 2>&1 &
