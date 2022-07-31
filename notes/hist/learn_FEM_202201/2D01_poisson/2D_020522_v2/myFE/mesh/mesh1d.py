"""
01.25.2022
"""
import numpy as np
import matplotlib.pyplot as plt

class UniformMesh1d:
    def __init__(self, xmin=0.0, xmax=1.0, nx=16):
        """
        generate P and T matrix
        :param xmin: the left point
        :type xmin: double
        :param xmax: the right point
        :type xmax: double
        :param nx: the number of elements
        :type nx: int
        """
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.h = (xmax - xmin) / self.nx

    def getMatP(self):
        N = self.nx  # mesh element number
        Nm = N+1  # mesh node number
        P = np.zeros(Nm)
        for i in range(1, Nm+1):  # [1, Nm]
            idx = i-1  # index used in python
            P[idx] = self.xmin + self.h * idx
        return P

    def getMatT(self):
        N = self.nx
        Nm = N+1
        T = np.zeros((2, N), dtype=np.int)
        for i in range(1, N+1): # [1,N]
            idx = i-1
            T[0, idx] = i
            T[1, idx] = i+1
        return T
    ######################################################
    def printMatP(self):
        print('*** matrix P is ')
        print(self.getMatP())

    def printMatT(self):
        print('*** matrix T is ')
        print(self.getMatT())

    def plotmesh(self, showNode=True, showNodeNum=True):
        y = np.zeros(self.nx + 1)
        P = self.getMatP()
        plt.figure()
        plt.plot(P, y, 'k')
        if showNode:
            plt.plot(P, y, 'r*')
        if showNodeNum:
            for i in range(len(P)):
                x = P[i]
                plt.text(x, 0., '%d'%(i+1))
        plt.xlabel('X')
        plt.ylabel('Y')
