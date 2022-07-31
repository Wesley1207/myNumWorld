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


def parabolic_solver():
    h = 1./4
    basisType = 201
    xmin = 0.
    xmax = 2.
    ymin = 0.
    ymax = 1. 
    tstart = 0.
    tend = 1.
    dt = h
    theta = 0.5
    
    Mm = int((tend - tstart)/dt)  # time element 
    nx = int((xmax-xmin)/h)
    ny = int((ymax-ymin)/h)
    solver2d = Solver2d(xmin,xmax, ymin, ymax,nx=nx,ny=ny,basisType=basisType, numGaussPoint=9)

    # assemble mat M
    @jit(nopython=True, cache=True)
    def coeff_matM(x, y):
        return 1.
    matM = solver2d.getMatA(coeff_matM, 0, 0, 0, 0)
    # assemble mat A
    @jit(nopython=True, cache=True)
    def coeff(x,y):
        return 2.
    matA = solver2d.getMatA(coeff, 1,0,1,0) 
    matA += solver2d.getMatA(coeff, 0,1,0,1)
    matAfix = matM*(1./dt) + theta*matA
    # assemble X0
    vecX0 = np.zeros(solver2d.Nb)
    for i in range(vecX0.shape[0]):
        x = solver2d.matPb[0,i]
        y = solver2d.matPb[1,i]
        vecX0[i] = np.exp(x+y)
    
    print(matM.shape)
    print(vecX0.shape)
    # time iteration
    @jit(nopython=True, cache=True)
    def rhsf(x,y,t):
        return (
                -3.*np.exp(x+y+t)
               )
    @jit(nopython=True, cache=True)
    def gfunc(x,y,t):
        return np.exp(x+y+t)

    for m in range(0, (Mm-1)+1):
        tnew = (m+1)*dt # t_{m+1}
        told = m*dt
        if m==0:
            vecXold = vecX0
        # assemble bnew
        vecbnew = solver2d.getVecb(rhsf,0,0,t=tnew)
        # assemble bold
        vecbold = solver2d.getVecb(rhsf,0,0,t=told)
        bnewTilde = theta*vecbnew + (1-theta)*vecbold + \
                    (1./dt)*matM.dot(vecXold)  -  \
                    (1-theta)*matA.dot(vecXold)
        # treat dirichlet BC
        bcEdge, bcNode = solver2d.getMatBCedgeAndNode(-1,-1,-1,-1)
        matAnew = matAfix
        matAnew, bnewTilde = solver2d.treatDirichletBC(gfunc,bcNode,matAnew,bnewTilde,tnew)
        # solve vecXnew
        vecXnew = solver2d.solAxeqb(matAnew, bnewTilde)
        print("t=%.2f" % tnew)
        vecXold = vecXnew

    # err estimate
    @jit(nopython=True, cache=True)
    def exactVal(x,y,t):
        return np.exp(x+y+t)
    Pb = solver2d.matPb
    err = np.zeros(Pb.shape[1])
    for i in range(Pb.shape[1]):
        err[i] = abs(vecXnew[i]-exactVal(Pb[0,i], Pb[1,i], tnew))
    print(" *** max abs err at all nodes: ")
    print("%.4e" % (np.amax(err)))




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
    start_time = time.time()
    parabolic_solver()
    end_time = time.time()
    print("time use = %f sec" % (end_time-start_time))
    
    pass
