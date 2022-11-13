from dataset import DataSet
from utils import Cluster, Atom
import numpy as np
import pandas as pd
import random
from tqdm import tqdm


def com(cluster):
    x = 0
    y = 0
    z = 0
    for atom in cluster.atoms:
        x += atom.x
        y += atom.y
        z += atom.z
    x /= cluster.num_atoms
    y /= cluster.num_atoms
    z /= cluster.num_atoms
    return x, y, z


def correct(cluster, x, y, z):
    tc = Cluster('offspring', 20, None)
    for atom in cluster.atoms:
        ta = Atom('Au', atom.x - x, atom.y - y, atom.z - z)
        tc.__add__(ta)
    return cluster


class GACluster:
    def __init__(self):
        self.atoms_par = []
        self.atoms_pos = []
        self.atoms_dot = []
        self.nv = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        self.pos_fa = []
        self.neg_fa = []
        self.pos_mo = []
        self.neg_mo = []

    def count(self):
        for i, value in enumerate(self.atoms_dot):
            if value > 0 and self.atoms_par[i] == 'father':
                self.pos_fa.append(self.atoms_pos[i])
            elif value < 0 and self.atoms_par[i] == 'father':
                self.neg_fa.append(self.atoms_pos[i])
            elif value > 0 and self.atoms_par[i] == 'mother':
                self.pos_mo.append(self.atoms_pos[i])
            elif value < 0 and self.atoms_par[i] == 'mother':
                self.neg_mo.append(self.atoms_pos[i])
        sit1 = self.pos_fa + self.neg_mo
        sit2 = self.neg_fa + self.pos_mo
        if len(sit1) == 32:
            return sit1
        elif len(sit2) == 32:
            return sit2
        else:
            return None

    def cal_pos(self, cluster, par):
        for atom in cluster.atoms:
            pos = np.array([atom.x, atom.y, atom.z])
            self.atoms_pos.append(pos)
            self.atoms_par.append(par)
            self.atoms_dot.append(np.dot(self.nv, pos))

    def write_xyz(self, name):
        file = []
        file.append(str(len(self.atoms_pos)) + '\n')
        file.append('\n')
        for i, atom in enumerate(self.atoms_pos):
            if self.atoms_par[i] == 'father':
                file.append('Au ' + str(atom[0]) + ' ' + str(atom[1]) + ' ' + str(atom[2]) + '\n')
            else:
                file.append('B ' + str(atom[0]) + ' ' + str(atom[1]) + ' ' + str(atom[2]) + '\n')
        with open(name + '.xyz', mode='w') as xyz:
            xyz.writelines(file)


clu = pd.read_excel('聚类结果.xlsx', header=None, index_col=0)

ds = DataSet('Au')

fathers = []
mothers = []

for xyz in ds.data:
    nb = int(xyz.filename.split('.')[0])
    if clu.at[nb, 1] - 1:
        fathers.append(xyz)
    else:
        mothers.append(xyz)

fathers.sort()
mothers.sort()

father = fathers[:25]
mother = mothers[:25]

poss = None
preview_energy = 0

for i in tqdm(range(1000000)):
    fa1 = father[random.randint(0, 24)]
    mo1 = mother[random.randint(0, 24)]
    fa2 = father[random.randint(0, 24)]
    mo2 = mother[random.randint(0, 24)]
    gac = GACluster()
    x, y, z = com(fa1)
    fa1 = correct(fa1, x, y, z)
    x, y, z = com(fa2)
    fa2 = correct(fa2, x, y, z)
    x, y, z = com(mo1)
    mo1 = correct(mo1, x, y, z)
    x, y, z = com(mo2)
    mo2 = correct(mo2, x, y, z)
    gac.cal_pos(fa1, 'father')
    gac.cal_pos(mo1, 'mother')
    gac.cal_pos(fa2, 'father')
    gac.cal_pos(mo2, 'mother')
    te = gac.count()
    if te != None:
        cluster = Cluster(str(i))
        cluster.num_atoms = 32
        for pos in te:
            cluster.__add__(Atom('Au', pos[0], pos[1], pos[2]))
        cluster.cnt_distance('md')
        cluster.cnt_energy()
        print(cluster.energy)
        if cluster.energy <= preview_energy:
            poss = cluster
            preview_energy = cluster.energy
            cluster.write_xyz()

i = 0
