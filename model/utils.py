import numpy as np
import re
import os
import math
from gupta import Gupta


class DataRange:
    def __init__(self, data):
        self.Max = data.max()
        self.max = self.Max.max()
        self.Min = data.min()
        self.min = self.Min.min()
        self.range = self.max - self.min


class Atom:
    def __init__(self, element, pos_x, pos_y, pos_z):
        self.element = element
        self.x = pos_x
        self.y = pos_y
        self.z = pos_z

    def seq(self):
        return np.array([self.x, self.y, self.z])


class Cluster:
    def __init__(self, filename, num_atoms=0, energy=0):
        self.filename = filename
        self.num_atoms = num_atoms
        self.energy = energy
        self.atoms = []
        self.distance = []

    def seq(self, dim=1):
        atoms = []
        for atom in self.atoms:
            atoms.append(atom.seq())
        if dim == 1:
            return np.concatenate(atoms)
        elif dim == 2:
            return np.array(atoms)

    def __eq__(self, other):
        return self.energy == other.energy

    def __lt__(self, other):
        return self.energy < other.energy

    def __gt__(self, other):
        return self.energy > other.energy

    def __add__(self, other):
        self.atoms.append(other)

    def write_xyz(self):
        file = []
        file.append(str(self.num_atoms)+'\n')
        file.append(str(self.energy)+'\n')
        for atom in self.atoms:
            file.append(atom.element + ' ' + str(atom.x) + ' ' + str(atom.y) + ' ' + str(atom.z) + '\n')
        with open(self.filename + '.xyz', mode='w') as xyz:
            xyz.writelines(file)

    def read_xyz(self, file_path):
        if file_path != '':
            file_path += os.sep
        with open(file_path + self.filename, mode='r') as curr_xyz:
            curr_xyz = curr_xyz.readlines()
        self.num_atoms = int(curr_xyz[0])
        try:
            self.energy = float(curr_xyz[1].split(':')[-1])
        except Exception:
            pass
        for atom in curr_xyz[2:]:
            atom = re.split(r" +", atom)
            self.__add__(Atom(atom[0], float(atom[1]), float(atom[2]), float(atom[3])))

    def cnt_distance(self, mode):
        """
        :param mode: m=matrix,s=sequence;
                    u=unique,d=duplicate;
        """
        if mode == 'mu':
            for i, atom1 in enumerate(self.atoms):
                dis_atom = []
                for atom2 in self.atoms[i + 1:]:
                    dis_atom.append(ed(atom1, atom2))
                self.distance.append(dis_atom)
        elif mode == 'md':
            for atom1 in self.atoms:
                dis_atom = []
                for atom2 in self.atoms:
                    dis_atom.append(ed(atom1, atom2))
                self.distance.append(dis_atom)
        elif mode == 'su':
            for i, atom1 in enumerate(self.atoms):
                for atom2 in self.atoms[i + 1:]:
                    self.distance.append(ed(atom1, atom2))
        elif mode == 'sd':
            for atom1 in self.atoms:
                for atom2 in self.atoms:
                    self.distance.append(ed(atom1, atom2))

    def cnt_energy(self):
        self.energy = Gupta(np.array(self.distance)).ug


def ed(atom1, atom2):
    return math.sqrt(
        math.pow(atom1.x - atom2.x, 2) + math.pow(atom1.y - atom2.y, 2) + math.pow(atom1.z - atom2.z, 2))


def get_split_data(data, size=0.7):
    test_data = data[int(len(data) * size):]
    train_data = data[:int(len(data) * size)]
    return train_data, test_data

