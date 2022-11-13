import os
import pandas as pd
import numpy as np
from utils import DataRange, Cluster

DATASET = "题目/B/附件"
Au_DATASET = DATASET + "/Au20_OPT_1000"
B_DATASET = DATASET + "/B45-_OPT_3751"


class DataSet:
    def __init__(self, element):
        self.data = []
        self.distance = []
        self.np2d_x = None
        self.np3d_x = None
        self.np_y = None
        self.range_x = None
        self.range_y = None
        self.range_z = None
        if element == 'Au':
            dataset = Au_DATASET
        elif element == 'B':
            dataset = B_DATASET
        else:
            raise ValueError
        for file in os.listdir(dataset):
            c = Cluster(file)
            c.read_xyz(dataset)
            self.data.append(c)

    @staticmethod
    def mapping(array, min_n, max_n, scale=256):
        return np.round((array - min_n)/(max_n - min_n) * scale)

    def datarange(self):
        self.np2d_x, self.np3d_x, self.np_y = self.list2ndarray()
        self.range_x = DataRange(self.np2d_x[:, 0::3])
        self.range_y = DataRange(self.np2d_x[:, 1::3])
        self.range_z = DataRange(self.np2d_x[:, 2::3])

    def pic(self):
        self.datarange()
        pic = np.zeros_like(self.np3d_x)
        pic[:, :, 0] = self.mapping(self.np3d_x[:, :, 0], self.range_x.min, self.range_x.max)
        pic[:, :, 1] = self.mapping(self.np3d_x[:, :, 1], self.range_y.min, self.range_y.max)
        pic[:, :, 2] = self.mapping(self.np3d_x[:, :, 2], self.range_z.min, self.range_z.max)
        return pic

    def write_excel(self):
        filename = []
        pos = []
        energy = []
        for cluster in self.data:
            filename.append(int(cluster.filename.split('.')[0]))
            pos.append(cluster.seq(1))
            energy.append(cluster.energy)
        fn = pd.DataFrame(filename)
        X = pd.DataFrame(pos)
        Y = pd.DataFrame(energy)
        data = pd.concat([fn, Y, X], axis=1)
        data.to_excel('B.xlsx', header=None, index=False)

    def write_csv(self):
        X = pd.DataFrame(self.np2d_x)
        Y = pd.DataFrame(self.np_y)
        data = pd.concat([X, Y], axis=1)
        data[:700].to_csv('train.csv', index=False, header=False)
        data[700:].to_csv('test.csv', index=False, header=False)

    def cnt_distance(self, mode):
        for cluster in self.data:
            cluster.cnt_distance(mode)
            self.distance.append(cluster.distance)

    def data2seq(self, dim=2):
        pos = []
        energy = []
        for cluster in self.data:
            pos.append(cluster.seq(dim - 1))
            energy.append(cluster.energy)
        return np.array(pos), np.array(energy)

    def list2ndarray(self):
        np2d_x, np_y = self.data2seq()
        np3d_x, _ = self.data2seq(3)
        return np2d_x, np3d_x, np_y

    def list2dataframe(self):
        pd_x, pd_y = self.data2seq()
        return pd.DataFrame(pd_x), pd.DataFrame(pd_y)
