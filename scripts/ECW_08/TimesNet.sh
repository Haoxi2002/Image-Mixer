#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_TimesNet \
      --model TimesNet \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_08.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
		  --d_model 16 \
		  --n_heads 8 \
		  --e_layers 2 \
		  --d_layers 1 \
		  --factor 3 \
		  --d_ff 32 \
		   --top_k 5 \
      --use_multi_gpu \
      > ECW_08_TimesNet.txt 2>&1 &
