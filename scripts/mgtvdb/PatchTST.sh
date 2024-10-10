#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id mgtvdb_PatchTST \
      --model PatchTST \
      --data mgtvdb \
      --dir_path ./data/PPIO \
      --data_path mgtvdb.csv \
      --seq_len 288 \
      --label_len 144 \
      --pred_len 288  \
      --freq 5min \
      --d_model 64 \
      --n_heads 2 \
      --d_ff 2048 \
      --dropout 0 \
      --use_multi_gpu \
      > mgtvdb_PatchTST.txt 2>&1 &
