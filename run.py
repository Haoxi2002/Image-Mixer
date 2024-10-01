import argparse
import random

import numpy as np
import torch

from exp import Exp_Long_Term_Forecast_VI

if __name__ == '__main__':
    fix_seed = 2021
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Multimodal TSF')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='1 for train or 0 for test')
    parser.add_argument('--draw_test', type=int, default=1, help='draw test result')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='Image-Mixer', help='model name')
    parser.add_argument('--model_type', type=int, default=0, help='0 for image-based model, 1 for numerical-based model')

    # data loader
    parser.add_argument('--data', type=str, default='ECW_08', help='data type')
    parser.add_argument('--dir_path', type=str, default='./data/ECW', help='dir path')
    parser.add_argument('--data_path', type=str, default='ECW_08.csv', help='data file name')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='mps', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h ]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', default=True, help='inverse output data')

    # numerical config
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # fig config
    parser.add_argument('--h', type=int, default=96, help='h')
    parser.add_argument('--lw', type=float, default=0.5, help='line width')
    parser.add_argument('--expand', type=int, default=1, help='expansion rate')
    parser.add_argument('--lc', type=float, nargs='+', default=(0, 0, 0), help='line color')
    parser.add_argument('--bc', type=float, nargs='+', default=(1, 1, 1), help='background color')
    parser.add_argument('--channel', type=int, default=1, help='3 for RGB and 1 for Grey')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimension')
    parser.add_argument('--patch_size', type=int, nargs='+', default=(8, 8), help='patch_size')
    parser.add_argument('--token_mlp_dim', type=int, default=128, help='token_mlp_dim')
    parser.add_argument('--channel_mlp_dim', type=int, default=32, help='channel_mlp_dim')
    parser.add_argument('--n_blocks', type=int, default=3, help='h')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='optim', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='gpu ids of multiple gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args: {}'.format(args))

    exp = Exp_Long_Term_Forecast_VI(args)
    if args.is_training:
        setting = '{}_{}_{}_seq_len{}_hd{}_ps{}_tmd{}_cmd{}_nb{}_drop{}_{}_lradj{}'.format(
            args.task_id,
            args.model,
            args.data,
            args.seq_len,
            args.hidden_dim,
            args.patch_size,
            args.token_mlp_dim,
            args.channel_mlp_dim,
            args.n_blocks,
            args.dropout,
            args.features,
            args.lradj
        )

        print('>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting)
    else:
        setting = '{}_{}_{}_seq_len{}_hd{}_ps{}_tmd{}_cmd{}_nb{}_drop{}_{}_lradj{}'.format(
            args.task_id,
            args.model,
            args.data,
            args.seq_len,
            args.hidden_dim,
            args.patch_size,
            args.token_mlp_dim,
            args.channel_mlp_dim,
            args.n_blocks,
            args.dropout,
            args.features,
            args.lradj
        )

        print('>>>>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting, test=1)
