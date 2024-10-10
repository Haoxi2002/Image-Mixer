#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ksdb_PatchTST \
      --model PatchTST \
      --data ksdb \
      --dir_path ./data/PPIO \
      --data_path ksdb.csv \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --freq 5min \
      --d_model 256 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --batch_size 512 \
      --learning_rate 0.0001 \
      --use_multi_gpu \
      > ksdb_PatchTST.txt 2>&1 &
