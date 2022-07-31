"""
ymma, 2022.02.07
2D, Robin bc, corresponding to ch3, ppt p104
"""
from myFE.FE.solver2d_new import Solver2d
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit


def run():
    h=1/12
    basisType=202

    @jit(nopython=True, cache=True)
    def coeff(x,y):
        return 1

    @jit(nopython=True, cache=True)
    def rhsf(x,y):
        return (
                -2.*np.exp(x+y)
               )

    @jit(nopython=True, cache=True)
    def gfunc(x,y):
        return np.exp(x+y)

    @jit(nopython=True, cache=True)
    def pfunc(x,y):
        tol = 1e-14
        if abs(y-(-1))<tol:
            return (-np.exp(x-1.))
        else:
            raise ValueError("(x,y) is not on the boundary")
    
    @jit(nopython=True, cache=True)
    def crfunc(x,y):
        tol = 1e-14
        if abs(y-(-1))<tol:
            return 1.
        else:
            raise ValueError("(x,y) is not on the boundary")

    @jit(nopython=True, cache=True)
    def cqfunc(x,y):
        tol = 1e-14
        if abs(y-(-1))<tol:
            return 0.
        else:
            raise ValueError("(x,y) is not on the boundary")



    @jit(nopython=True, cache=True)
    def exactVal(x,y):
        return (np.exp(x+y))

    n = int(2/h)
    solver2d = Solver2d(-1,1,-1,1,nx=n,ny=n,basisType=basisType, numGaussPoint=9, numGaussPoint1d=4)
    # t1111 = time.time()
    matA = solver2d.getMatA(coeff, 1,0,1,0) 
    matA += solver2d.getMatA(coeff, 0,1,0,1)
    # t2222 = time.time()
    # print("time use matA = %f sec" % (t2222-t1111))
    vecb = solver2d.getVecb(rhsf,0,0)
    bcEdge, bcNode = solver2d.getMatBCedgeAndNode(-1,-1,-3,-1)
    # print(bcNode)
    pass
    bcNode[0,0]=-1
    vecb = solver2d.treatNeumannBC(pfunc, bcEdge, vecb, 0, 0)
    matA, vecb = solver2d.treatRobinBC(cqfunc, crfunc, bcEdge, matA, vecb, 0, 0, 0, 0, 0, 0)
    matA, vecb = solver2d.treatDirichletBC(gfunc,bcNode,matA,vecb)
    
    
    pass
    sol = solver2d.solAxeqb(matA, vecb)
    
    Pb = solver2d.matPb
    err = np.zeros(Pb.shape[1])
    for i in range(Pb.shape[1]):
        err[i] = abs(sol[i]-exactVal(Pb[0,i], Pb[1,i]))
    print(" *** max abs err at all nodes: ")
    print("%.4e" % (np.amax(err)))



if __name__ == '__main__':
    start_time = time.time()
    run()
    end_time = time.time()
    print("time use = %f sec" % (end_time-start_time))
    
    pass

 