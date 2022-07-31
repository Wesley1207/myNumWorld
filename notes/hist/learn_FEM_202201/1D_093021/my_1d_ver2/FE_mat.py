import numpy as np
import scipy.integrate as sci_integrate

"""
*** def gen_stiff_mat_A_1d(coeff_func,
                       N_grid, mat_size,  # decide A property
                       P_grid, T_grid,  # calc lower-upper integral bound
                       T_FE_trial, T_FE_test,  # calc index of A in every loop
                       num_trial_local_basis, num_test_local_basis,  # basis properties
                       trial_basis_type, trial_derivative_degree,
                       test_basis_type, test_derivative_degree)

*** def gen_load_vec_b_1d(coeff_func,
                      N_grid, vec_size,
                      P_grid, T_grid,
                      T_FE_test,
                      num_test_local_basis,
                      test_basis_type, test_derivative_degree)
                      
*** def gen_BC_nodes_1d(N_FE)

*** def treat_Dirichlet_BC_1d(Dirichlet_BC_func, A, b, bc_node_vec, P_FE)

*** def integral_test_1d(coeff_func,
                     test_vert_vec,
                     test_basis_type, test_basis_index, test_derivative_degree,
                     lower_bound, upper_bound)
                     
*** def integral_test_1d(coeff_func,
                     test_vert_vec,
                     test_basis_type, test_basis_index, test_derivative_degree,
                     lower_bound, upper_bound)

*** def local_basis_1d(x, vert_vec, basis_type, basis_index, derivative_degree)

"""

def gen_stiff_mat_A_1d(coeff_func,
                       N_grid, mat_size,  # decide A property
                       P_grid, T_grid,  # calc lower-upper integral bound
                       T_FE_trial, T_FE_test,  # calc index of A in every loop
                       num_trial_local_basis, num_test_local_basis,  # basis properties
                       trial_basis_type, trial_derivative_degree,
                       test_basis_type, test_derivative_degree):
    """ generate stiff matrix A
    * coeff_func: coeff_func : function c(x)
    * N_grid: element number of mesh
    * mat_size: (N_FE+1, N_FE+1)
    * P_grid: P matirx
    * T_grid: T matrix
    * T_FE_trial: \
    * T_FE_test:  \
    * num_trial_local_basis: \
    * num_test_local_basis: \
    * trial_basis_type: \
    * trial_derivative_degree: \
    * test_basis_type: \
    * test_derivative_degree: \
    """
    A = np.zeros((mat_size[0], mat_size[1]), dtype=np.double)
    for n in range(N_grid):
        # generate integral low-up bound
        vert_vec = P_grid[(T_grid[:, n]-1)]
        lower_bound = np.min(vert_vec)
        upper_bound = np.max(vert_vec)

        for alpha in range(num_trial_local_basis):
            for beta in range(num_test_local_basis):
                r = integral_trial_test_1d(coeff_func,
                                           vert_vec, trial_basis_type, alpha, trial_derivative_degree,
                                           vert_vec, test_basis_type, beta, test_derivative_degree,
                                           lower_bound, upper_bound)
                A[T_FE_test[beta, n]-1, T_FE_trial[alpha, n]-1] += r
    return A


def gen_load_vec_b_1d(coeff_func,
                      N_grid, vec_size,
                      P_grid, T_grid,
                      T_FE_test,
                      num_test_local_basis,
                      test_basis_type, test_derivative_degree):
    """ generate load vector b
    * coeff_func: function f(x)
    * N_grid: element number of mesh
    * vec_size: N_FE+1
    * P_grid:\
    * T_grid:\
    * T_FE_test: T in FE space
    * num_test_local_basis: \
    * test_basis_type: 101 or 102 or ...
    * test_derivative_degree: \
    """
    b = np.zeros(vec_size, dtype=np.double)

    for n in range(N_grid):
        vert_vec = np.array(P_grid[(T_grid[:, n] - 1)])
        lower_bound = np.min(vert_vec)
        upper_bound = np.max(vert_vec)

        for beta in range(num_test_local_basis):
            r = integral_test_1d(coeff_func,
                                 vert_vec,
                                 test_basis_type, beta, test_derivative_degree,
                                 lower_bound, upper_bound)
            b[T_FE_test[beta, n]-1] += r
    return b


def treat_Dirichlet_BC_1d(Dirichlet_BC_func, A, b, bc_node_vec, P_FE):
    """ treat Dirichlet boundary condition, return modified A and b
    * Dirichlet_BC_func: function g_d(x)
    * A: \
    * b: \
    *bc_node_vec: boundary node information matrix
      bc_node_vec[0, k]: type of kth boundary node
                         -1: Dirichlet boundary node
                         -2: Neumann boundary node
                         -3: Robin boundary node
      bc_node_vec[1, k]: global index of kth boundary node
      bc_node_vec[2, k]: normal direction of kth boundary node
    * P_FE: node coordinate in FE space
    :return A, b
    """
    # nbn: number of boundary nodes (in FE space)
    nbn = bc_node_vec.shape[1]
    for k in range(nbn):
        if bc_node_vec[0, k] == -1:
            i = bc_node_vec[1, k]
            A[i-1, :] = 0.
            A[i-1, i-1] = 1.
            b[i-1] = Dirichlet_BC_func(P_FE[i-1])
    return A, b


def treat_Neumann_BC_1d(Neumann_BC_func, b, bc_node_vec, P_FE):
    """ treat Neumann boundary condition, return modified b
    * Neumann_BC_func: function g_n(x)
    * b: \
    * bc_node_vec: boundary node information matrix
      bc_node_vec[0, k]: type of kth boundary node
                         -1: Dirichlet boundary node
                         -2: Neumann boundary node
                         -3: Robin boundary node
      bc_node_vec[1, k]: global index of kth boundary node
      bc_node_vec[2, k]: normal direction of kth boundary node
    * P_FE: node coordinate in FE space
    :return b
    """
    # nbn: number of boundary nodes (in FE space)
    nbn = bc_node_vec.shape[1]
    for k in range(nbn):
        if bc_node_vec[0, k] == -2:
            norm_dir = bc_node_vec[2, k]
            i = bc_node_vec[1, k]
            b[i-1] += norm_dir * Neumann_BC_func(P_FE[i-1])
    return b


def treat_Robin_BC_1d(Neumann_like_BC_func, Robin_BC_func, A, b, bc_node_vec, P_FE):
    """ treat Robin boundary condition, return modified A and b
    * Neumann_like_BC_func: function g_n(x) -> to modify b
    * Robin_BC_func: function g_r(x) -> to modify A
    * A: \
    * b: \
    * bc_node_vec: boundary node information matrix
      bc_node_vec[0, k]: type of kth boundary node
                         -1: Dirichlet boundary node
                         -2: Neumann boundary node
                         -3: Robin boundary node
      bc_node_vec[1, k]: global index of kth boundary node
      bc_node_vec[2, k]: normal direction of kth boundary node
    * P_FE: node coordinate in FE space
    :return A, b
    """
    # nbn: number of boundary nodes (in FE space)
    nbn = bc_node_vec.shape[1]
    for k in range(nbn):
        if bc_node_vec[0, k] == -3:
            norm_dir = bc_node_vec[2, k]
            i = bc_node_vec[1, k]  # global node number
            b[i-1] += norm_dir * Neumann_like_BC_func(P_FE[i-1])
            A[i-1, i-1] += norm_dir * Robin_BC_func(P_FE[i-1])

    return A, b


def FE_solution_1d(x, uh_local_vec, vert_vec, basis_type, derivative_degree):
    """ calc FE solution for any x
    * x: point x
    * uh_local_vec: value on nodes of FE solution
    * vert_vec: vertice coordinate
    """
    sol = 0
    for i in range(len(uh_local_vec)):
        sol += uh_local_vec[i] * local_basis_1d(x, vert_vec, basis_type, i, derivative_degree)
    return sol


def integral_trial_test_1d(coeff_func,
                           trial_vert_vec, trial_basis_type, trial_basis_index, trial_derivative_degree,
                           test_vert_vec, test_basis_type, test_basis_index, test_derivative_degree,
                           lower_bound, upper_bound):
    """ serve for integral in generation of matrix A
    * coeff_func : function c(x)
    * trial_vert_vec: vertice coordinates of trial function
    * trial_basis_type: 101, 102 ...
    * trial_basis_index: \
    * trial_derivative_degree: \
    * ...
    """
    result = sci_integrate.quad(lambda x:
                                coeff_func(x) *
                                local_basis_1d(x, trial_vert_vec, trial_basis_type,
                                               trial_basis_index, trial_derivative_degree) *
                                local_basis_1d(x, test_vert_vec, test_basis_type,
                                               test_basis_index, test_derivative_degree),
                                lower_bound, upper_bound)

    return result[0]


def integral_test_1d(coeff_func,
                     test_vert_vec,
                     test_basis_type, test_basis_index, test_derivative_degree,
                     lower_bound, upper_bound):
    """ serve for integral in generation of vector b
    * coeff_func: function f(x)
    * test_vert_vec: vertice coordinates of test function
    * test_basis_type: 101 or 102 or ...
    * test_basis_index: \
    * test_derivative_degree: \
    """
    result = sci_integrate.quad(lambda x:
                                coeff_func(x) *
                                local_basis_1d(x, test_vert_vec, test_basis_type,
                                               test_basis_index, test_derivative_degree),
                                lower_bound, upper_bound)
    return result[0]


def local_basis_1d(x, vert_vec, basis_type, basis_index, derivative_degree):
    """ local basis function for 1d case
    * x: coordinate of where we want to evaluate
    * vert_vec: grid vertices, not FE vertices
    * basis_type: 101: 1d linear FE
                  102: 1d quadratic FE
    * basis_index: which basis function we want to use
    * derivative_degree : as its name
    """
    result = 0
    if basis_type == 101:
        if derivative_degree == 0:
            if basis_index == 0:
                result = (vert_vec[1] - x) / (vert_vec[1]-vert_vec[0])
            elif basis_index == 1:
                result = (x - vert_vec[0]) / (vert_vec[1]-vert_vec[0])
            else:
                raise ValueError("wrong basis index")
        elif derivative_degree == 1:
            if basis_index == 0:
                result = -1./(vert_vec[1]-vert_vec[0])
            elif basis_index == 1:
                result = 1./(vert_vec[1]-vert_vec[0])
            else:
                raise ValueError("wrong basis index")
        else:
            raise ValueError("wrong derivative degree")
    elif basis_type == 102:
        # use affine mapping
        h = vert_vec[1] - vert_vec[0]
        x_hat = (x - vert_vec[0]) / h

        if derivative_degree == 0:
            if basis_index == 0:
                result = 2. * x_hat**2 - 3. * x_hat + 1.
            elif basis_index == 1:
                result = 2. * x_hat**2 - x_hat
            elif basis_index == 2:
                result = -4. * x_hat**2 + 4. * x_hat
            else:
                raise ValueError("wrong basis_index")
        elif derivative_degree == 1:
            if basis_index == 0:
                result = (4. * x_hat - 3.) * (1./h)
            elif basis_index == 1:
                result = (4. * x_hat - 1.) * (1./h)
            elif basis_index == 2:
                result = (-8. * x_hat + 4.) * (1./h)
            else:
                raise ValueError("wrong basis_index")
        elif derivative_degree == 2:
            if basis_index == 0:
                result = (4. * (1./h)) * (1./h)
            elif basis_index == 1:
                result = (4. * (1./h)) * (1./h)
            elif basis_index == 2:
                result = (-8. * (1./h)) * (1./h)
            else:
                raise ValueError("wrong basis_index")
        else:
            raise ValueError("wrong derivative degree")

    else:
        raise ValueError("basis_type is not 101 or 102")

    return result
