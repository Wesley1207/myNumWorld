"""
ymma, 2022.01.31
2D, dirichlet bc, corresponding to ch3, ppt p80,81
"""
from myFE.FE.solver2d_new import Solver2d
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
# import sys
# sys.path.append("./FE")


def run():
    h=1/2
    basisType=202

    @jit(nopython=True, cache=True)
    def coeff(x,y):
        return 1

    @jit(nopython=True, cache=True)
    def rhsf(x,y):
        return (-y*(1-y)*(1-x-x**2 *0.5)*np.exp(x+y) - 
                x*(1-x*0.5)*(-3*y-y**2)*np.exp(x+y)
               )

    @jit(nopython=True, cache=True)
    def gfunc(x,y):
        tol = 1e-14
        if abs(x-(-1))<tol:
            return (-1.5*y*(1-y)*np.exp(-1+y))
        elif abs(x-1)<tol:
            return (0.5*y*(1-y)*np.exp(1+y))
        elif abs(y-(-1))<tol:
            return (-2*x*(1-x*0.5)*np.exp(x-1))
        elif abs(y-1)<tol:
            return 0
        else:
            raise ValueError("(x,y) is not on the boundary")

    @jit(nopython=True, cache=True)
    def exactVal(x,y):
        return (x*y*(1-x*0.5)*(1-y)*np.exp(x+y))

    n = int(2/h)
    solver2d = Solver2d(-1,1,-1,1,nx=n,ny=n,basisType=basisType, numGaussPoint=9)
    # t1111 = time.time()
    matA = solver2d.getMatA(coeff, 1,0,1,0) 
    matA += solver2d.getMatA(coeff, 0,1,0,1)
    # t2222 = time.time()
    # print("time use matA = %f sec" % (t2222-t1111))
    vecb = solver2d.getVecb(rhsf,0,0)
    bcEdge, bcNode = solver2d.getMatBCedgeAndNode(-1,-1,-1,-1)
    # print(bcNode)
    matA, vecb = solver2d.treatDirichletBC(gfunc,bcNode,matA,vecb)
    sol = solver2d.solAxeqb(matA, vecb)
    
    Pb = solver2d.matPb
    err = np.zeros(Pb.shape[1])
    for i in range(Pb.shape[1]):
        err[i] = abs(sol[i]-exactVal(Pb[0,i], Pb[1,i]))
    print(" *** max abs err at all nodes: ")
    print("%.4e" % (np.amax(err)))



if __name__ == '__main__':
    # # generate mesh, i.e. matrix P
    # fevar = feVar2d(xmin=-1., xmax=1., ymin=-1., ymax=1.,
    #                 nx=10, ny=10, basisType=1
    #                 )
    # matP = fevar.getMatP()
    # matT = fevar.getMatT()
    # bcEdge, bcNode = fevar.getMatBCedgeAndNode()  
    # matPb = fevar.getMatPb() 
    # matTb = fevar.getMatTb()
    # x = np.linspace(-1,-0.8,10)
    # y = np.ones(len(x))*(-1)
    # for i in range(len(x)):
    #     vertMat = np.array( [ [-1, -0.8, -1],
    #                           [-1, -1,   -0.8], ] )
    #     r = fevar.locBasis(x[i], y[i], vertMat,1,0,0)
    #     print("x=%f, y=%f, result=%f" %(x[i], y[i], r))
    start_time = time.time()
    run()
    end_time = time.time()
    print("time use = %f sec" % (end_time-start_time))
    
    pass

 