import argparse
import os
from collections import defaultdict
import concurrent.futures

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import StandardScaler
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

        self.h = self.args.h
        self.channel = self.args.channel
        assert self.channel in [1, 3]
        self.bc = self.args.bc
        self.lw = self.args.lw
        self.lc = self.args.lc
        self.expand = self.args.expand
        self.model_type = self.args.model_type

        self.__read_data__()

    def data2Pixel(self, dataXIn, draw_type='opencv'):
        assert draw_type in ['matplotlib', 'opencv', 'sampling']
        dataX = np.copy(dataXIn.T)
        feature = dataX.shape[0]
        lenX = dataX.shape[1]

        imgX = np.zeros([feature * self.channel, lenX * self.expand, self.h * self.expand], dtype=np.float32)
        for i in range(feature):
            if draw_type == 'matplotlib':
                canvas = FigureCanvasAgg(
                    plt.figure(figsize=(lenX / 100 * self.expand, self.h / 100 * self.expand), facecolor=self.bc))
                plt.plot(dataX[i], linewidth=self.lw, color=self.lc)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.margins(0, 0)
                canvas.draw()
                buf = canvas.buffer_rgba()
                if self.channel == 1:
                    img = np.dot(np.asarray(buf)[:, :, :3] / 255, [0.2989, 0.5870, 0.1140])
                    imgX[i, :img.shape[1], :] = img.T
                else:  # self.channel == 3:
                    img = np.asarray(buf)[:, :, :3] / 255
                    imgX[i * self.channel:(i + 1) * self.channel, :img.shape[1], :] = np.transpose(img, (2, 1, 0))
                plt.close()
            else:
                img = (np.ones((self.h * self.expand, lenX * self.expand, 3), dtype=np.uint8) * (
                    int(self.bc[0] * 255), int(self.bc[1] * 255), int(self.bc[2] * 255))).astype(np.uint8)
                if np.max(dataX[i]) == np.min(dataX[i]):
                    data_line = np.ones(len(dataX[i])) - 0.5
                else:
                    data_line = 1 - (dataX[i] - np.min(dataX[i])) / (np.max(dataX[i]) - np.min(dataX[i]))
                if draw_type == 'opencv':
                    for j in range(lenX - 1):
                        pt1 = (int(j * self.expand), round(data_line[j] * (self.h * self.expand - 1)))
                        pt2 = (int((j + 1) * self.expand), round(data_line[j + 1] * (self.h * self.expand - 1)))
                        cv2.line(img, pt1, pt2, (int(self.lc[0] * 255), int(self.lc[1] * 255), int(self.lc[2] * 255)),
                                 self.lw if isinstance(self.lw, int) else 1)
                else:  # if draw_type == 'sampling'  self.lw is not used
                    data_line = np.round(data_line * (self.h - 1)).astype(int)
                    for j in range(self.expand):
                        img[np.repeat(data_line, self.expand) * self.expand + j, np.arange(len(data_line) * self.expand), :] = (int(self.lc[0] * 255), int(self.lc[1] * 255), int(self.lc[2] * 255))
                if self.channel == 1:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgX[i, :gray_img.shape[1], :] = np.transpose(np.expand_dims(gray_img, axis=0) / 255, (0, 2, 1))
                else:  # self.channel == 3:
                    imgX[i * self.channel:(i + 1) * self.channel, :, :] = np.transpose(img / 255, (2, 1, 0))
        return np.transpose(imgX, (0, 2, 1))

    def __read_data__(self):
        df_raw = pd.read_csv(str(os.path.join(self.dir_path, self.data_path)))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)  # long-term TSF
        num_test = int(len(df_raw) * 0.2) if self.args.is_training else len(df_raw)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # self.features == 'S'
            df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        data = data[border1:border2]
        self.data = defaultdict(list)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.astype(object).apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.astype(object).apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.astype(object).apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.astype(object).apply(lambda row: row.hour)
        df_stamp['minute'] = df_stamp.date.astype(object).apply(lambda row: row.minute)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
        data_stamp = df_stamp.drop(columns=['date']).values
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-4) as executor:
            futures = []
            for i in range(len(data) - self.seq_len - self.pred_len + 1):
                data_x = data[i:i + self.seq_len]
                self.data['x'].append(data_x)
                self.data['y'].append(data[i + self.seq_len:i + self.seq_len + self.pred_len])
                if self.model_type == 0:
                    self.data['max'].append(np.amax(data_x, axis=0))
                    self.data['min'].append(np.amin(data_x, axis=0))
                    self.data['median'].append(np.median(data_x, axis=0))
                    self.data['mean'].append(np.mean(data_x, axis=0))
                    self.data['25th'].append(np.percentile(data_x, 25, axis=0))
                    self.data['75th'].append(np.percentile(data_x, 75, axis=0))
                    self.data['ptp'].append(np.ptp(data_x, axis=0))
                    self.data['std'].append(np.std(data_x, axis=0))
                    self.data['var'].append(np.var(data_x, axis=0))
                    futures.append(executor.submit(self.data2Pixel, data_x))
                else:
                    self.data['x_mark'].append(data_stamp[i:i + self.seq_len])
                    self.data['y_mark'].append(
                        data_stamp[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len])
            for future in futures:
                self.data['fig'].append(future.result())
        if self.model_type == 0:
            self.static = np.concatenate([np.array(self.data['max'])[:, :, np.newaxis],
                                          np.array(self.data['min'])[:, :, np.newaxis],
                                          np.array(self.data['median'])[:, :, np.newaxis],
                                          np.array(self.data['mean'])[:, :, np.newaxis],
                                          np.array(self.data['25th'])[:, :, np.newaxis],
                                          np.array(self.data['75th'])[:, :, np.newaxis],
                                          np.array(self.data['ptp'])[:, :, np.newaxis],
                                          np.array(self.data['std'])[:, :, np.newaxis],
                                          np.array(self.data['var'])[:, :, np.newaxis]], axis=2)

    def __getitem__(self, index):
        if self.model_type == 0:
            return self.data['x'][index], self.data['y'][index], self.data['fig'][index], self.static[index], 0, 0
        else:
            return self.data['x'][index], self.data['y'][index], 0, 0, self.data['x_mark'][index], self.data['y_mark'][
                index]

    def __len__(self):
        return len(self.data['x'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')
    args = parser.parse_args()
    args.is_training = 1
    args.dir_path = '../data/ETT-small/'
    args.data_path = 'ETTm1.csv'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.target = 'OT'
    args.features = 'M'
    args.h = 24
    args.lw = 1
    args.expand = 3
    args.channel = 3
    args.lc = (0, 0, 0)
    args.bc = (1, 1, 1)
    args.model_type = 1

    data_set = Dataset_Basic(args=args, flag='test')
    print(len(data_set))

    data = []
    for i in range(96):
        data.append(np.sin(i))
    fig1 = data_set.data2Pixel(np.asarray(data)[:, np.newaxis], draw_type='opencv')
    fig2 = data_set.data2Pixel(np.asarray(data)[:, np.newaxis], draw_type='matplotlib')
    fig3 = data_set.data2Pixel(np.asarray(data)[:, np.newaxis], draw_type='sampling')
    pass
