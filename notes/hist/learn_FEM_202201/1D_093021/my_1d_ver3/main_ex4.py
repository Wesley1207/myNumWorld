"""
ymma, 2022.01.26
example 3, ppt p135 (Robin BC)
"""
from myFE.mesh.mesh1d import UniformMesh1d
import matplotlib.pyplot as plt
import myFE.FE.solver1d as fe
import numpy as np


if __name__ == '__main__':
    # generate mesh, i.e. matrix P
    mesh1d = UniformMesh1d(xmin=0.0, xmax=1.0, nx=16)
    # mesh1d.printMatP()
    # mesh1d.printMatT()
    # fevar1d = fe.feVar1d(matP=mesh1d.getMatP(), basisType=102)
    # fevar1d.printMatPb()
    # fevar1d.printMatTb()
    mesh1d.plotmesh()
    matP = mesh1d.getMatP()
    basisType = 102
    # FE, linear basis
    # -d/dx (c(x) * d/dx u(x)) = f(x)
    # example 1, ppt p85
    # c(x) = e^x, f(x) = -e^x *(cos(x) -2*sin(x)-x*cos(x)-x*sin(x))
    # u'(0)+u(0)=1, u(1) = cos(1),  x \in [0,1]
    solver = fe.Solver1d(matP, basisType)
    def coeff(x):
        return np.exp(x)
    def rhsf(x):
        return -np.exp(x) *(np.cos(x) -2.*np.sin(x) -x*np.cos(x)-x*np.sin(x))
    def gfunc(x):
        tol = 1e-14
        if abs(x-1) < tol:
            return np.cos(1)
        else:
            raise ValueError("wrong x")
    def qfunc(x):
        return 1
    def pfunc(x):
        tol = 1e-14
        if abs(x-0) < tol:
            return 1
        else:
            raise ValueError("wrong x")
    matA = solver.getMatA(coeff, 1, 1)
    vecb = solver.getVecb(0, rhsf)
    # treat BC
    Tb = solver.getMatTb()
    bcnodes = np.zeros((3,2), dtype=np.int)
    bcnodes[0, 0] = -3  # robin type
    bcnodes[1, 0] = 1  # global node #1
    bcnodes[2, 0] = -1  # normal direction
    bcnodes[0, 1] = -1  # dirichlet type
    bcnodes[1, 1] = solver.Nb
    bcnodes[2, 1] = 1
    treatbc = fe.TreatBC1d(matP, basisType, bcnodes)
    matA, vecb = treatbc.dirichlet(gfunc, matA, vecb)
    matA, vecb = treatbc.robin(qfunc, pfunc, coeff, matA, vecb)

    xsol = solver.solAxeqb(matA, vecb)

    # err estimate
    def exactVal(x):
        return x*np.cos(x)
    Pb = solver.getMatPb()
    err = np.zeros(len(Pb))
    for i in range(len(Pb)):
        err[i] = abs(xsol[i] - exactVal(Pb[i]))
    print(" *** max abs err at all nodes: ")
    print("%.4e" % (np.amax(err)))

    # plot result
    plt.figure()
    plt.plot(Pb, xsol)
    # plot estimate result
    xnew = np.linspace(0,1, 1000)
    solnew = np.zeros(len(xnew))
    for i in range(len(xnew)):
        solnew[i] = solver.getSol(xnew[i], xsol)
    plt.plot(xnew, solnew, 'r:')
    
    plt.show()

