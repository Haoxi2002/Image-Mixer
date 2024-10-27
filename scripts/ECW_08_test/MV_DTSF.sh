#!/bin/bash

nohup python run.py \
      --is_training 1 \
      --task_id ECW_MV_DTSF \
      --model MV_DTSF \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newapp.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --use_multi_gpu \
      > ECW_newapp_MV_DTSF.txt 2>&1 &

nohup python run.py \
      --is_training 1 \
      --task_id ECW_MV_DTSF \
      --model MV_DTSF \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_newmac.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --use_multi_gpu \
      > ECW_newmac_MV_DTSF.txt 2>&1 &

nohup python run.py \
      --is_training 1 \
      --task_id ECW_MV_DTSF \
      --model MV_DTSF \
      --data ECW \
      --dir_path ./data/ECW \
      --data_path ECW_switch.csv \
      --freq h \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 24  \
		  --inverse \
      --h $((24)) \
      --lw 1 \
      --expand 1 \
      --channel 3 \
      --use_multi_gpu \
      > ECW_switch_MV_DTSF.txt 2>&1 &