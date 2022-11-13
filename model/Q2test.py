from utils import Cluster
from gupta import Gupta
import numpy as np

cluster = Cluster('15364.xyz')
cluster.read_xyz('')
cluster.cnt_distance('md')
seq = cluster.seq(2)

gu = Gupta(np.array(cluster.distance))

ug = gu.ug

i = 0