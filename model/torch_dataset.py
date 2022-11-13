import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_xyz, list2ndarray

DATASET = "题目/B/附件"
Au_DATASET = DATASET + "/Au20_OPT_1000"
B_DATASET = DATASET + "/B45-_OPT_3751"


class ClusterDataset(Dataset):
    def __init__(self, element, mode='train', transform=None):
        self.transform = transform
        if element == 'Au':
            dataset = Au_DATASET
        elif element == 'B':
            dataset = B_DATASET
        else:
            raise ValueError
        self.data = []
        for file in os.listdir(dataset):
            self.data.append(read_xyz(dataset, file))
        _, self.np_x, self.np_y = list2ndarray(self.data)
        if mode == 'train':
            self.np_x = self.np_x[int(self.__len__() * 0.7):]
            self.np_y = self.np_y[int(self.__len__() * 0.7):]
        elif mode == 'val':
            self.np_x = self.np_x[:int(self.__len__() * 0.3)]
            self.np_y = self.np_y[:int(self.__len__() * 0.3)]

    def __len__(self):
        return len(self.np_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.np_x[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.DoubleTensor(sample), self.np_y[idx]
