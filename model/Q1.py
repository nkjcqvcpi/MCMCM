from dataset import DataSet
import numpy as np
from gupta import Gupta
import math


ds = DataSet('B')
ds.cnt_distance('md')
dis = np.array(ds.distance)
_, __, np_y = ds.list2ndarray()
pred = []
for cluster in dis:
    pred.append(Gupta(cluster).ug)

pred = np.array(pred)
loss = math.sqrt(np.mean(np.power(np.subtract(pred, np_y), 2)))
i=0
