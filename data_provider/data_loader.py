import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset


class Dataset_Basic(Dataset):
    def __init__(self, args, flag):
        self.args = args
        self.dir_path = self.args.dir_path
        self.data_path = self.args.data_path
        self.seq_len = self.args.seq_len
        self.label_len = self.args.label_len
        self.pred_len = self.args.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = self.args.target
        self.features = self.args.features
        assert self.features in ['S', 'MS', 'M']

        self.scaler = MinMaxScaler()

        self.h = self.args.h
        self.channel = self.args.channel
        assert self.channel in [1, 3]
        self.bc = self.args.bc
        self.lw = self.args.lw
        self.lc = self.args.lc
        self.expand = self.args.expand
        self.model_type = self.args.model_type

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(str(os.path.join(self.dir_path, self.data_path)))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        if 'ETTh' in self.data_path:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ETTm' in self.data_path:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            if self.pred_len < 96:  # short-term TSF
                num_train = int(len(df_raw) * 0.6)
            else:  # long-term TSF
                num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            if 'ECW' in self.data_path and '08' not in self.data_path:  # only test
                num_test = len(df_raw)
            border1s = [0, num_train, len(df_raw) - num_test]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS' or 'ECW' in self.args.data:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # self.features == 'S'
            df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values.reshape(-1, 1))
        feature = df_data.shape[1]
        data = self.scaler.transform(df_data.values.reshape(-1, 1)).reshape(-1, feature)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.model_type == 0:
            self.mu = np.mean(data, axis=0)[:, np.newaxis]
            self.std = np.std(data, axis=0)[:, np.newaxis] + 1e-7

            self.fig_path = os.path.join(self.dir_path, self.data_path.split('.')[0])
            if not os.path.exists(self.fig_path):
                os.makedirs(self.fig_path)
            fig_name = f'{self.args.data}_{self.set_type}_h{self.h}_lw{self.lw}_lc{self.lc}_bc{self.bc}_channel{self.channel}_seq_len{self.seq_len}_ep{self.expand}_f{self.features}.npy'
            fig_path = os.path.join(str(self.fig_path), fig_name)
            # if os.path.exists(fig_path):
            #     self.fig_data_x = np.load(fig_path)
            # else:
            self.fig_data_x = self.data2Pixel(data)[:, border1 * self.expand:border2 * self.expand, :]
            np.save(fig_path, self.fig_data_x)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
            self.data_stamp = data_stamp

    def data2Pixel(self, dataXIn):
        dataX = np.copy(dataXIn.T)
        dataX = (dataX - self.mu) / self.std
        feature = dataX.shape[0]
        lenX = dataX.shape[1]

        imgX = np.zeros([feature * self.channel, lenX * self.expand, self.h * self.expand])
        for i in range(feature):
            canvas = FigureCanvasAgg(
                plt.figure(figsize=(lenX * self.expand / 100, self.h * self.expand / 100), facecolor=self.bc))
            plt.plot(dataX[i], linewidth=self.lw, color=self.lc)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.margins(0, 0)
            # plt.savefig(f'fig/{self.set_type}_{i}.png')
            canvas.draw()
            buf = canvas.buffer_rgba()
            if self.channel == 1:
                img = np.dot(np.asarray(buf)[:, :, :3] / 255, [0.2989, 0.5870, 0.1140])
                imgX[i, :img.shape[1], :] = img.T
            else:  # self.channel == 3:
                img = np.asarray(buf)[:, :, :3] / 255
                imgX[i * self.channel:(i + 1) * self.channel, :img.shape[1], :] = np.transpose(img, (2, 1, 0))
            plt.close()
        return imgX

    def __getitem__(self, index):
        device = index // ((self.data_x.shape[0] - self.seq_len - self.pred_len) // 12)
        index = index % ((self.data_x.shape[0] - self.seq_len - self.pred_len) // 12)
        s_begin = index * 12
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, device][:, np.newaxis]
        seq_y = self.data_y[r_begin:r_end, device][:, np.newaxis]

        if self.model_type == 0:
            fig_x = self.fig_data_x[device * self.channel:(device + 1) * self.channel, s_begin * self.expand:s_end * self.expand, :]
            mu = self.mu[device, :][np.newaxis, :]
            std = self.std[device, :][np.newaxis, :]
            return seq_x, seq_y, fig_x, mu, std, 0, 0
        else:
            seq_x_mark = self.data_stamp[s_begin:s_end, :]
            seq_y_mark = self.data_stamp[r_begin:r_end, :]
            return seq_x, seq_y, 0, 0, 0, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_x.shape[1] * (self.data_x.shape[0] - self.seq_len - self.pred_len) // 12

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
