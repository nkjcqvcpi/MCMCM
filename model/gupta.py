import math


class Gupta:
    def __init__(self, r):
        self.A = 0.2061
        self.xi = 1.79
        self.d = 2.884
        self.p = 10.229
        self.q = 4.036
        self.C = -432.93731945158004
        self.N = len(r)
        self.r = r
        self.ug = self.u()

    def rho(self, i):
        rhori = 0
        for j in range(self.N):
            rhori += self.xi**2 * math.exp(-2*self.q*(self.r[i, j]/self.d-1))
        return rhori

    def u(self):
        ug = 0
        for i in range(self.N):
            ae = 0
            for j in range(self.N):
                if j != i:
                    ae += (math.exp(-self.p*(self.r[i, j]/self.d-1)) * self.A)
            ug += ae - math.sqrt(self.rho(i))
        return ug


# class Tersoff:
#     def __init__(self, r):
#         self.r = r
#         self.A = 277.02
#
#     def E(self):
#         e = 0
#         for i in range(N):
#             for j in range(N):
#                 if j!=i:
#                     e += V[i, j]
#
#     def V(self):
#         return fC(self.r[i, j]+delta)*(fR(self.r[i, j]+delta)+b[i,j]*fA(self.r[i, j]+delta))
#
#     def fC(self, r):
#         if r < R-D:
#             return 1
#         elif r>R+D:
#             return 0
#         else:
#             return 1/2-math.sin((math.pi/2)*((r-R)/D)/2)
#
#     def fR(self):
#         return A*math.exp(-lambda1*r)
#
#     def fA(self):
#         return -B*math.exp(-lambda2*r)
#
#     def bij(self):
#         return math.pow(1+math.pow(beta, n)*math.pow(xi, n), -1/(2*n))
#
#     def xiij(self):
#         xi = 0
#         for k in range(N):
#             if k!=i and k!= j:
#                 xi += fC(r[i, k]+delta)*g(theta(r[i,j],r[i,k]))*math.exp(math.pow(lambda3*(r[i,j]-r[i,k]), m))
#
#     def g(self,theta):


"""
Pair    A (eV)   B (eV)   lambda (AA^-1)   mu (AA^-1)  beta           m       c        d         h        R   S    chi     omega
B-B        183.49   1.9922           1.5856      0.0000016      3.9929  0.52629  0.001587  0.5      1.8 2.1  1.0     1.0
"""
# from utils import Cluster, Atom
# import re

# with open('zncluster.xyz', mode='r') as curr_xyz:
#     curr_xyz = curr_xyz.readlines()
# cluster = Cluster('zncluster', int(curr_xyz[0]), float(re.split(r" +", curr_xyz[1])[2]))
# for atom in curr_xyz[2:]:
#     atom = re.split(r" +", atom)
#     cluster.__add__(Atom(atom[1], float(atom[2]), float(atom[3]), float(atom[4])))
#
#
# def cnt_distance(cluster):
#     distance = []
#     for i, atom1 in enumerate(cluster.atoms):
#         dis_atom = []
#         for atom2 in cluster.atoms:
#             dis_atom.append(ed(atom1, atom2))
#         distance.append(dis_atom)
#     return distance


# dis1 = np.array(cnt_distance(cluster))

# clu = Cluster('28916.xyz')
# clu.read_xyz('')
# clu.cnt_distance('md')
#
# dis = np.array(clu.distance)
#
# temp = Gupta(dis)
#
# i=0


