import numpy as np
from scipy.sparse import csc_matrix as sci_csc_matrix

def gen_mesh_P_T_1d(left, right, h, basis_type):
    """ generate matrix P and T for a mesh in 1d domain
    * basis_type: 101: 1d linear FE
                  102: 1d quadratic FE
    * P : node coordinate matrix
          ith row : ith coordinate
          jth column: jth node
          in 1d case, P only has 1 row.
    * T: element node (global) index matrix
          ith row: global index of the ith local node
          jth column: jth global element
    """
    if basis_type == 101:
        N_grid = int((right - left)/h)
        P = np.zeros(N_grid+1, dtype=np.double)
        T = np.zeros((2, N_grid), dtype=np.int32)

        # generate P matrix
        for i in range(N_grid+1):
            P[i] = left + i*h

        # generate T matrix
        for i in range(N_grid):
            T[0, i] = i+1
            T[1, i] = i+2

    elif basis_type == 102:
        N_grid = int((right-left)/h)
        h_FE = h/2.
        N_FE = N_grid*2
        P = np.zeros(N_FE+1, dtype=np.double)
        T = np.zeros((3, N_grid), dtype=np.int32)

        # generate P matrix
        for i in range(N_FE+1):
            P[i] = left + i*h_FE

        # generate T matrix
        for i in range(N_grid):
            temp = i+1
            T[0, i] = 2*temp - 1
            T[1, i] = 2*temp + 1
            T[2, i] = 2*temp
    else:
        raise ValueError("basis_type is not 101 or 102")
    return P, T
