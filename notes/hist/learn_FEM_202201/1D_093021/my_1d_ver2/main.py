"""
10.03.2021
"""
import numpy as np
import matplotlib.pyplot as plt

import solver
import mesh as msh
import err_estimate


if __name__ == '__main__':
    """ init """
    grid_start = 0
    grid_end = 1
    h = 1./32
    basis_type = 102
    """ deal with mesh """
    P_grid, T_grid = msh.gen_mesh_P_T_1d(grid_start, grid_end, h, 101)
    if basis_type == 101:
        P_FE = P_grid
        T_FE = T_grid
    elif basis_type == 102:
        P_FE, T_FE = msh.gen_mesh_P_T_1d(grid_start, grid_end, h, 102)
    else:
        raise ValueError("basis type is not 101 or 102")

    def x_analysis(Pb_mat):
        x_analy = np.array([])
        for i in range(len(Pb_mat)):
            x = Pb_mat[i]
            x_analy = np.append(x_analy, x*np.cos(x))
        return x_analy

    """ calc solution """
    x_sol = solver.poisson_solver_1d(grid_start, grid_end, h, basis_type)
    x_analy = x_analysis(P_FE)

    """ calc error """
    max_err = np.amax(np.abs(x_sol - x_analy))
    print('error=', max_err)

    # L^inf norm err
    L_inf_norm_err = err_estimate.err_est_inf_norm_1d(x_sol, solver.func_exact_u,
                                                      grid_start, grid_end, h, basis_type, 0, 10)
    print("L^inf norm err = ", L_inf_norm_err)
    # L^2 norm err
    L_2_norm_err = err_estimate.err_est_1d(x_sol, solver.func_exact_u, grid_start, grid_end,
                                        h, basis_type, 0)
    print("L^2 norm err = ", L_2_norm_err)
    # H^1 norm err
    H_1_norm_err = err_estimate.err_est_1d(x_sol, solver.func_u_derivative, grid_start, grid_end,
                                        h, basis_type, 1)
    print("H^1 norm err = ", H_1_norm_err)



    """ plot figures """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(P_FE, x_sol, linestyle='dashed',lw=3)
    ax1.plot(P_FE, x_analy)
#    ax21.set_xlabel(r'$B_{z2}/B_{z1}$')
#    ax21.set_ylabel(r'$\beta_{2}/\beta_{1}$')
    ax1.grid(True, linestyle='--')
    plt.show()
    pass
