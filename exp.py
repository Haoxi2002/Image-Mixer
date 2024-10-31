import os
import sys
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from model import ImageMixer, PatchTST, LSTM, Informer, Autoformer, DLinear, TSMixer, TimesNet, Pyraformer, MV_DTSF, \
    KAE_Informer
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


class Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_type = self.args.model_type
        self.model_dict = {
            'ImageMixer': ImageMixer,
            'PatchTST': PatchTST,
            'LSTM': LSTM,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'TSMixer': TSMixer,
            'TimesNet': TimesNet,
            'Pyraformer': Pyraformer,
            'MV_DTSF': MV_DTSF,
            'KAE_Informer': KAE_Informer,
        }
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(
                self.args.gpu if not self.args.use_multi_gpu else self.args.devices)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(list(self.model.parameters()), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps, epochs=self.args.train_epochs, max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (seq_x, seq_y, fig_x, static, seq_x_mark, seq_y_mark) in enumerate(train_loader):
                iter_count += 1
                seq_y = seq_y.float().to(self.device)
                model_optim.zero_grad()
                if self.model_type == 0:
                    fig_x = fig_x.float().to(self.device)
                    static = static.float().to(self.device)
                    y_pred = self.model(fig_x, static)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                loss = criterion(y_pred, seq_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if self.args.lradj == 'optim':
                    adjust_learning_rate(model_optim, epoch, self.args, scheduler, printnot=False)
                    scheduler.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch + 1) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    sys.stdout.flush()

            cost_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}  speed: {:.4f}s/iter".format(epoch + 1, cost_time, cost_time / train_steps))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            sys.stdout.flush()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'optim':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, seq_y, fig_x, static, seq_x_mark, seq_y_mark) in enumerate(val_loader):
                seq_y = seq_y.float().to(self.device)
                if self.model_type == 0:
                    fig_x = fig_x.float().to(self.device)
                    static = static.float().to(self.device)
                    y_pred = self.model(fig_x, static)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                loss = criterion(y_pred, seq_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            state_dict = torch.load(os.path.join(self.args.checkpoints, setting) + '/' + 'checkpoint.pth')
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     new_key = k.replace('module.', '')  # 移除module前缀
            #     new_state_dict[new_key] = v
            # self.model.load_state_dict(new_state_dict)

        seq_xs = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if self.args.draw_test and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, seq_y, fig_x, static, seq_x_mark, seq_y_mark) in enumerate(test_loader):
                seq_y = seq_y.float().to(self.device)
                if self.model_type == 0:
                    fig_x = fig_x.float().to(self.device)
                    static = static.float().to(self.device)
                    y_pred = self.model(fig_x, static)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                outputs = y_pred.cpu().detach().numpy()
                batch_y = seq_y.cpu().detach().numpy()

                if self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                seq_xs.append(seq_x.cpu().detach().numpy())
                preds.append(pred)
                trues.append(true)
                if self.args.draw_test and i % 100 == 0:
                    input = seq_x.cpu().detach().numpy()
                    if self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        print('Inference time: {:.4f}s, Model parameters: {:.2f} MB'.format((time.time() - start_time) / len(test_loader), sum(p.numel() for p in self.model.parameters()) * 4 / (1024 ** 2)))
        seq_xs = np.asarray(seq_xs)
        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe = metric(preds, trues)
        print('mean_mse:{}, var_mse:{}, top_mse:{}, bottom_mse:{}, peak_mse:{}\nmean_mae:{}, var_mae:{}, top_mae:{}, bottom_mae:{}, peak_mae:{}'.format(mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mean_mse:{}, var_mse:{}, top_mse:{}, bottom_mse:{}, peak_mse:{}\nmean_mae:{}, var_mae:{}, top_mae:{}, bottom_mae:{}, peak_mae:{}'.format(mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae))

        np.save(folder_path + 'metrics.npy', np.array([mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe]))
        np.save(folder_path + 'seq_x.npy', seq_xs)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        f.write('\n')
        f.close()
        return
