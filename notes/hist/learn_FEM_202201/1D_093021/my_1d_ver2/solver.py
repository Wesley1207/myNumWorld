import mesh as msh
import FE_mat
import numpy as np
# from scipy.sparse import csc_matrix as sci_csc_matrix
# import scipy.integrate as sci_integrate


def poisson_solver_1d(left, right, h, basis_type):
    """ 1d poisson problem solver
    * problem domain: [left, right]
    * h: step size of partition
    * basis type: 101: linear FE basis;
                  102: quadratic FE basis.

    internal vars:
    ** N : number of element
    * N_grid: N for grid
    * N_FE: N for FE basis function
    * func_c: function on the l.h.s. of Poisson eqn
    * func_f: function on the r.h.s. of Poisson eqn
    * func_g: Dirichlet boundary function
    """
    # all internal vars
    # N_grid
    # N_FE
    # P_grid
    # T_grid
    # P_FE
    # T_FE
    # num_trial_local_basis
    # num_test_local_basis

    # set N_grid and N_FE
    N_grid = int((right - left) / h)
    if basis_type == 101:
        N_FE = N_grid
    elif basis_type == 102:
        N_FE = N_grid * 2
    else:
        raise ValueError("basis type is not 101 or 102")

    # mesh info
    P_grid, T_grid = msh.gen_mesh_P_T_1d(left, right, h, 101)

    if basis_type == 101:
        P_FE = P_grid
        T_FE = T_grid
    elif basis_type == 102:
        P_FE, T_FE = msh.gen_mesh_P_T_1d(left, right, h, 102)
    else:
        raise ValueError("basis type is not 101 or 102")

    # assemble stiff matrix
    mat_size = np.array([N_FE+1, N_FE+1], dtype=np.int32)
    if basis_type == 101:
        num_trial_local_basis = 2
        num_test_local_basis = 2
    elif basis_type == 102:
        num_trial_local_basis = 3
        num_test_local_basis = 3
    A_mat = FE_mat.gen_stiff_mat_A_1d(func_c,
                                      N_grid, mat_size,
                                      P_grid, T_grid,
                                      T_FE, T_FE,
                                      num_trial_local_basis, num_test_local_basis,
                                      basis_type, 1,
                                      basis_type, 1)

    # assemble load vector b
    vec_size = N_FE+1
    b_vec = FE_mat.gen_load_vec_b_1d(func_f,
                                     N_grid, vec_size,
                                     P_grid, T_grid, T_FE, num_test_local_basis,
                                     basis_type, 0)
    # treat boundary condition
    bc_node_vec = gen_BC_nodes_1d(N_FE)
    b_vec = FE_mat.treat_Neumann_BC_1d(func_g_n, b_vec, bc_node_vec, P_FE)
    A_mat, b_vec = FE_mat.treat_Robin_BC_1d(func_g_r_neumann, func_g_r, A_mat, b_vec, bc_node_vec, P_FE)
    A_mat, b_vec = FE_mat.treat_Dirichlet_BC_1d(func_g_d, A_mat, b_vec, bc_node_vec, P_FE)
    x_sol = np.linalg.solve(A_mat, b_vec)
    return x_sol

def func_c(x):
    """ function c(x) in the l.h.s
    """
    return np.exp(x)


def func_f(x):
    """ function f(x) in the r.h.s
    """
    return -np.exp(x) * (np.cos(x) - 2. * np.sin(x) - x * np.cos(x) - x * np.sin(x))


def func_exact_u(x):
    return x*np.cos(x)

def func_u_derivative(x):
    return np.cos(x) - x*np.sin(x)


def gen_BC_nodes_1d(N_FE):
    """ generate boundary node vector
    * N_FE: element number in FE space
    * bc_node_vec[0, k]: type of kth boundary node
                         -1: Dirichlet boundary node
                         -2: Neumann boundary node
                         -3: Robin boundary node
      bc_node_vec[1, k]: global index of kth boundary node
      bc_node_vec[2, k]: normal direction of kth boundary node
    """
    bc_node_vec = np.zeros((3, 2), dtype=np.int32)
    # example1: all for Dirichlet nodes
    bc_node_vec[0, 0] = -1
    bc_node_vec[1, 0] = 1
    bc_node_vec[2, 0] = -1
    bc_node_vec[0, 1] = -1
    bc_node_vec[1, 1] = N_FE + 1
    bc_node_vec[2, 1] = 1

    # # example3: left: Dirichlet, right: Neumann
    # bc_node_vec[0, 0] = -1
    # bc_node_vec[1, 0] = 1
    # bc_node_vec[2, 0] = -1
    # bc_node_vec[0, 1] = -2
    # bc_node_vec[1, 1] = N_FE + 1
    # bc_node_vec[2, 1] = 1

    # example4: left: Dirichlet, right: Neumann
    # bc_node_vec[0, 0] = -3
    # bc_node_vec[1, 0] = 1
    # bc_node_vec[2, 0] = -1
    # bc_node_vec[0, 1] = -1
    # bc_node_vec[1, 1] = N_FE + 1
    # bc_node_vec[2, 1] = 1
    return bc_node_vec


def func_g_d(x):
    """ function g_d(x), the Dirichlet boundary condition function
    """
    tol = 1.e-10
    if abs(x-0) < tol:
        return func_exact_u(0)
    if abs(x-1) < tol:
        return func_exact_u(1)
    return 0



def func_g_n(x):
    """ function g_n(x), the Neumann boundary condition function
    """
    # example 3
    tol = 1.e-10
    if abs(x-1.) < tol:
        r = func_u_derivative(x)
        return r * func_c(x)
    return 0


def func_g_r(x):
    """ function g_r(x), the Robin boundary condition function -> to modify A
    """
    tol = 1.e-10
    # example 4
    if abs(x-0)<tol:
        q = 1.
        return q * func_c(x)
    return 0


def func_g_r_neumann(x):
    """ function g_r_neumann(x), Robin boundary condition function -> to modify b
    """
    tol = 1.e-10
    # example 4
    if abs(x-0) < tol:
        p=1.
        return p*func_c(x)
    return 0