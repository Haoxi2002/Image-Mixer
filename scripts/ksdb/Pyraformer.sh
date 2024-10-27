#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ksdb_Pyraformer \
      --model Pyraformer \
      --data ksdb \
      --dir_path ./data/PPIO \
      --data_path ksdb.csv \
      --freq 5min \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
		  --d_model 512 \
		  --n_heads 6 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --d_ff 2048 \
		  --batch_size 256 \
      --use_multi_gpu \
      > ksdb_Pyraformer.txt 2>&1 &
