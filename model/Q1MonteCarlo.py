from dataset import DataSet
import numpy as np
import random
import tqdm
from utils import Atom, Cluster
from gupta import Gupta


class MonteCarlo:
    def __init__(self, iter=100000):
        dataset = DataSet('Au')
        dataset.datarange()
        self.x_range = dataset.range_x
        self.y_range = dataset.range_y
        self.z_range = dataset.range_z
        self.cluster = []
        self.distance = []
        self.energy = []
        for self.i in tqdm.tqdm(range(iter)):
            cluster = self.random_pos()
            cluster.cnt_distance('md')
            self.cluster.append(cluster)
            self.distance.append(cluster.distance)
            self.energy.append(Gupta(np.array(cluster.distance)).ug)
        self.result()

    def random_pos(self):
        cluster = Cluster(str(self.i), 40, None)
        for i in range(40):
            x = random.uniform(self.x_range.min, self.x_range.max)
            y = random.uniform(self.y_range.min, self.y_range.max)
            z = random.uniform(self.z_range.min, self.z_range.max)
            cluster.__add__(Atom('Au', x, y, z))
        return cluster

    def result(self):
        self.energy = np.array(self.energy)
        self.y_min = self.energy.min()
        self.min_index = np.argmin(self.energy)
        self.x_min = self.cluster[self.min_index]
        self.x_min.energy = self.y_min


x = MonteCarlo()

x.x_min.write_xyz()

i = 0
