import numpy as np
from scipy.sparse import csc_matrix as sci_csc_matrix
import scipy.integrate as sci_integrate


def gen_stiff_mat_A_1d(Pb_mat, Tb_mat, func_c):
    """ generate stiff matrix A """
    N = np.shape(Tb_mat)[1]   # element number
    A = sci_csc_matrix((N+1, N+1), dtype=np.double).toarray()
    # # define local basis functions
    # def psi_n1(x, x0, xN, h):
    #     n = int((x - x0)/h)  # n = 0,1,2...,N-1
    #     x_r = x0 + n*h + h
    #     return (x_r - x) / h
    #
    # def psi_n2(x, x0, xN, h):
    #     n = int((x-x0)/h)
    #     x_l = x0 + n*h
    #     return (x - x_l) / h
    #
    # def local_basis_1d(x, x0, xN, h, kind):
    #     if kind == 0:
    #         return psi_n1(x, x0, xN, h)
    #     elif kind == 1:
    #         return psi_n2(x, x0, xN, h)
    #     else:
    #         raise IndexError('kind is not 0 or 1')
    #
    x0 = Pb_mat[0]
    xN = Pb_mat[-1]
    h = (xN-x0) / N

    def d_basis(h, kind):
        if kind == 0:
            return -1./h
        elif kind == 1:
            return 1./h

    for n in range(N):
        for alpha in range(2):
            for beta in range(2):
                xl = Tb_mat[0,n]
                xr = Tb_mat[1,n]
                xl = Pb_mat[xl - 1]
                xr = Pb_mat[xr - 1]
                r = sci_integrate.quad(lambda x: func_c(x)*d_basis(h,alpha)*d_basis(h,beta),xl, xr)
                r = r[0]
                A[Tb_mat[beta, n]-1, Tb_mat[alpha, n]-1] += r
    return A

def gen_load_vec_b_1d(Pb_mat, Tb_mat, func_f):
    """ generate load vector b """
    N = np.shape(Tb_mat)[1]  # element number
    b = np.zeros(N+1)

    x0 = Pb_mat[0]
    xN = Pb_mat[-1]
    h = (xN-x0) / N
    # define local basis functions
    def psi_n1(x, x0, xN, h):
        n = int((x - x0)/h)  # n = 0,1,2...,N-1
        x_r = x0 + n*h + h
        return (x_r - x) / h

    def psi_n2(x, x0, xN, h):
        n = int((x-x0)/h)
        x_l = x0 + n*h
        return (x - x_l) / h

    def local_basis_1d(x, x0, xN, h, kind):
        if kind == 0:
            return psi_n1(x, x0, xN, h)
        elif kind == 1:
            return psi_n2(x, x0, xN, h)
        else:
            raise IndexError('kind is not 0 or 1')

    for n in range(N):
        for beta in range(2):
            xl = Tb_mat[0,n]
            xr = Tb_mat[1,n]
            xl = Pb_mat[xl - 1]
            xr = Pb_mat[xr - 1]
            r = sci_integrate.quad(lambda x: func_f(x) * local_basis_1d(x, x0, xN, h, beta),xl, xr)
            r = r[0]
            b[Tb_mat[beta, n]-1] += r
    return b


def gen_bc_mat_1d(kind, Tb_mat):
    """ generate 1d boundary condition """
    N = np.shape(Tb_mat)[1]
    bc_mat = np.zeros((2,2), dtype=np.int16)
    if kind == 0: # Dirichlet BC
        bc_mat[0, :] = 0
    bc_mat[1,0] = Tb_mat[0, 0]
    bc_mat[1,1] = Tb_mat[1,-1]
    return bc_mat

def add_Dirichlet_BC(A_mat, b_mat, bc_mat, Pb_mat ,g_func):
    for k in range(np.shape(bc_mat)[1]):
        if bc_mat[0,k] == 0: # dirichlet bc
            i = bc_mat[1, k]
            A_mat[i-1, :] = 0
            A_mat[i-1, i-1] = 1
            b_mat[i-1] = g_func(Pb_mat[i-1])

def solve_Ax_b(A_mat, b_mat):
    return np.linalg.solve(A_mat, b_mat)


if __name__ == "__main__":
    pass