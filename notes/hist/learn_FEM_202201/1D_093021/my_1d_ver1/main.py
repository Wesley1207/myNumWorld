import numpy as np
import mesh_generate as msh
import FE_mat_1d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """ init """
    grid_start = 0
    grid_end = 1
    h = 1./128
    grid_num = int((grid_end - grid_start) / h)

    def func_c(x):
        return np.exp(x)

    def func_f(x):
        return -np.exp(x)*(np.cos(x)-2*np.sin(x)-x*np.cos(x)-x*np.sin(x))

    def func_g(x):
        tol = 1.e-10
        if abs(x-0) < tol:
            return 0.
        if abs(x-1) < tol:
            return np.cos(1)
        return 0

    def x_analysis(Pb_mat):
        x_analy = np.array([])
        for i in range(len(Pb_mat)):
            x = Pb_mat[i]
            x_analy = np.append(x_analy, x*np.cos(x))
        return x_analy


    """ deal with mesh """
    mesh = msh.gen_mesh_1d(grid_start, grid_end, grid_num)   # get 1d mesh
    P = msh.gen_P_1d(mesh)  # get node coordinate matrix P
    T = msh.gen_T_1d(mesh)  # get element node number matrix T
    Pb = msh.gen_Pb_1d(mesh)  # FE coordinate matrix Pb
    Tb = msh.gen_Tb_1d(mesh)  # FE element node number matrix Tb

    A = FE_mat_1d.gen_stiff_mat_A_1d(Pb, Tb, func_c)
    b = FE_mat_1d.gen_load_vec_b_1d(Pb, Tb, func_f)
    bc_mat = FE_mat_1d.gen_bc_mat_1d(0, Tb)
    FE_mat_1d.add_Dirichlet_BC(A, b, bc_mat, Pb, func_g)
    x_calc = FE_mat_1d.solve_Ax_b(A, b)
    x_analy = x_analysis(Pb)

    max_err = np.amax(np.abs(x_calc - x_analy))
    print('error=', max_err)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(Pb, x_calc, linestyle='dashed',lw=3)
    ax1.plot(Pb, x_analy)
#    ax21.set_xlabel(r'$B_{z2}/B_{z1}$')
#    ax21.set_ylabel(r'$\beta_{2}/\beta_{1}$')
    ax1.grid(True, linestyle='--')
    plt.show()
    pass
