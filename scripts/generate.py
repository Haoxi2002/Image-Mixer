import os.path

models = ['ImageMixer', 'PatchTST']
datasets = [
    # {
    #     'data': 'ECW',
    #     'dir_path': './data/ECW',
    #     'data_path': 'ECW_08.csv',
    #     'is_training': 1
    # },
    {
        'data': 'ksdb',
        'dir_path': './data/PPIO',
        'data_path': 'ksdb.csv',
        'is_training': 1
    },
    {
        'data': 'mgtvdb',
        'dir_path': './data/PPIO',
        'data_path': 'mgtvdb.csv',
        'is_training': 1
    },
    {
        'data': 'bzmodb',
        'dir_path': './data/PPIO',
        'data_path': 'bzmodb.csv',
        'is_training': 1
    }
]

command = """#!/bin/bash

nohup python run.py \\
      --is_training {} \\
      --task_id {}_{} \\
      --model {} \\
      --data {} \\
      --dir_path {} \\
      --data_path {} \\
      --freq {} \\
      --seq_len {} \\
      --label_len {} \\
      --pred_len {} {} \\
      {}
      --dropout 0 \\
      --batch_size 512 \\
      --learning_rate 0.005 \\
      --use_multi_gpu \\
      > {}_{}.txt 2>&1 &
"""

command_num = """--d_model 512 \\
      --n_heads 8 \\
      --d_ff 2048 \\
      --c_out 1 \\"""

command_img = """--h 288 * 2 \\
      --lw 4 \\
      --channel 1 \\
      --hidden_dim 8 \\
      --patch_size 24 24 \\
      --token_mlp_dim 512 \\
      --channel_mlp_dim 16 \\
      --n_blocks 2 \\"""

for dataset in datasets:
    for model in models:
        if not os.path.exists(f"./{dataset['data']}"):
            os.makedirs(f"./{dataset['data']}")
        file = open(f"./{dataset['data']}/{model}.sh", "w")
        seq_len = 48 if dataset['data'] == 'ECW' else 288
        pred_len = 24 if dataset['data'] == 'ECW' else 288
        inverse = " \\\n\t\t  --inverse" if dataset['data'] == 'ECW' else ""
        freq = 'h' if dataset['data'] == 'ECW' else "5min"
        file.write(command.format(dataset['is_training'], dataset['data_path'][:-4], model, model, dataset['data'], dataset['dir_path'], dataset['data_path'], freq, seq_len, seq_len // 2, pred_len, inverse, command_img if model == 'ImageMixer' else command_num,dataset['data_path'][:-4], model))
        file.close()
