#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_08_PatchTST \
      --model PatchTST \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --d_model 64 \
      --n_heads 8 \
      --d_ff 2048 \
      --dropout 0.01 \
      --batch_size 512 \
      --learning_rate 0.001 \
      --use_multi_gpu \
      > ECW_08_PatchTST.txt 2>&1 &
