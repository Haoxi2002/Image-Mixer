#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ksdb_Autoformer \
      --model Autoformer \
      --data ksdb \
      --dir_path ./data/PPIO \
      --data_path ksdb.csv \
      --freq 5min \
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
      > ksdb_Autoformer.txt 2>&1 &
