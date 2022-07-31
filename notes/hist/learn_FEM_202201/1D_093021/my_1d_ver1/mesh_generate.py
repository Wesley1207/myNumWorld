import numpy as np
from scipy.sparse import csc_matrix as sci_csc_matrix


def gen_mesh_1d(start=0., end=1., num=10):
    """ generate 1d mesh """
    return np.linspace(start, end, num)

def gen_P_1d(mesh_mat):
    """ generate 1d node coordinate matrix """
    return mesh_mat

def gen_T_1d(mesh_mat):
    """ generate 1d element node number matrix """
    N = len(mesh_mat) - 1   # element number
    T = sci_csc_matrix((2, N), dtype=np.int16).toarray()
    for i in range(N):
        n = i+1  # n_th element, 1,2,...,N
        T[0, i] = n
        T[1, i] = n+1
    return T


def gen_Pb_1d(mesh_mat):
    """ generate 1d FE node coordinate matrix """
    return mesh_mat

def gen_Tb_1d(mesh_mat):
    """ generate 1d FE element node number matrix """
    N = len(mesh_mat) - 1   # element number
    T = sci_csc_matrix((2, N), dtype=np.int16).toarray()
    for i in range(N):
        n = i+1  # n_th element, 1,2,...,N
        T[0, i] = n
        T[1, i] = n+1
    return T


if __name__ == "__main__":
    pass
