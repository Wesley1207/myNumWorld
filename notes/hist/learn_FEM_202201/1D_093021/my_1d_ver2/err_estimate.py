import numpy as np
import mesh as msh
import scipy.integrate as sci_integrate
import FE_mat


def err_est_1d(uh, exact_func, left, right, h_grid, basis_type, derivative_degree):
    """ calc FE x_sol error
    * uh: x_sol
    * exact_func: \
    * left: \
    * right: \
    * h_grid: \
    * basis_type: \
    * derivative_degree: \
    :return: err
    """
    h = h_grid
    N_grid = int((right - left) / h)
    P_grid, T_grid = msh.gen_mesh_P_T_1d(left, right, h, 101)
    if basis_type == 101:
        P_FE = P_grid
        T_FE = T_grid
    elif basis_type == 102:
        P_FE, T_FE = msh.gen_mesh_P_T_1d(left, right, h, 102)
    else:
        raise ValueError("wrong basis type")

    err = 0
    for n in range(N_grid):
        vert_vec = P_grid[(T_grid[:, n]-1)]
        uh_local_vec = uh[(T_FE[:, n]-1)]
        err += integral_err_1d(uh_local_vec, exact_func, basis_type, derivative_degree, vert_vec)
    err = np.sqrt(err)
    return err


def err_est_inf_norm_1d(uh, exact_func, left, right, h_grid, basis_type, derivative_degree, sample_point_num):
    """ calc FE sol |u-uh|_inf error
    """
    h = h_grid
    N_grid = int((right-left)/h)
    P_grid, T_grid = msh.gen_mesh_P_T_1d(left, right, h, 101)
    if basis_type == 101:
        P_FE = P_grid
        T_FE = T_grid
    elif basis_type ==102:
        P_FE, T_FE = msh.gen_mesh_P_T_1d(left, right, h, 102)
    else:
        raise ValueError("wrong basis type")

    err = 0
    for n in range(N_grid):
        vert_vec = P_grid[(T_grid[:, n] - 1)]
        lower_bound = np.min(vert_vec)
        upper_bound = np.max(vert_vec)
        uh_local_vec = uh[(T_FE[:, n]-1)]

        x_vec = np.linspace(lower_bound, upper_bound, sample_point_num)
        max_err_abs = 0
        for i in range(len(x_vec)):
            x = x_vec[i]
            temp = abs(
                       exact_func(x) -
                       FE_mat.FE_solution_1d(x, uh_local_vec, vert_vec, basis_type, derivative_degree)
                      )
            if temp > max_err_abs:
                max_err_abs = temp
        if max_err_abs > err:
            err = max_err_abs
    return err


def integral_err_1d(uh_local_vec, exact_func, basis_type, derivative_degree, vert_vec):
    """ serve for err_est function, calc integral part in error estimation
    * uh_local_vec: value on nodes of FE solution
    """
    lower_bound = np.min(vert_vec)
    upper_bound = np.max(vert_vec)
    result = sci_integrate.quad(lambda x: (
                                        exact_func(x) -
                                        FE_mat.FE_solution_1d(x, uh_local_vec, vert_vec, basis_type, derivative_degree)
                                        )**2,
                              lower_bound, upper_bound)
    return result[0]