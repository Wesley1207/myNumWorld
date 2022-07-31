"""
2022.01.31
ymma

Different from 1D, mesh information can not separate with the FE information. 
i.e., we can not generate Pb, Tb using P and T, for the reason that we must 
also know the (left, right), (bottom, top), nx and ny while generating Pb and Tb. 
So mesh information and "solver information" are coupled. 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sciIntegrate
from numba import jit
import quadpy
from .opt2d import solver2dopt as sopt
import scipy.sparse.linalg

class feVar2d:
    """
    generate P, T, Pb, Tb, boudnaryedge matrix, and boundary node matrix
    """
    def __init__(self,  xmin, xmax, ymin, ymax, nx, ny, basisType):
        """
        :param xmin: the left 
        :type xmin: double
        :param xmax: the right 
        :type xmax: double
        :param ymin: the bottom
        :type ymin: double
        :param ymax: the top
        :type ymax: double
        :param nx: the number of elements in x dir
        :type nx: int
        :param ny: the number of elements in y dir
        :type ny: int
        :param basisType: basis function type, 
            1-> linear basis function
            2-> quadratic basis function
        :type basisType: int
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.N = self.nx*self.ny*2
        self.nmx = self.nx+1   # mesh nodes nubmer in x dir
        self.nmy = self.ny+1   # mesh nodes nubmer in x dir
        self.bt = basisType  # basis type
        self.hx = (xmax - xmin) / nx
        self.hy = (ymax - ymin) / ny
        if self.bt == 201:
            self.nbx = self.nx+1  # basis node number in x dir
            self.nby = self.ny+1  # basis node number in y dir
            self.Nb = self.nbx*self.nby
            self.hbx = self.hx
            self.hby = self.hy
            self.alpha = 3  # Nlb of trial function, # of local basis
            self.beta = 3  # Nlb of test function, # of local basis
        elif self.bt == 202:
            self.nbx = self.nx*2 + 1
            self.nby = self.ny*2 + 1
            self.Nb = self.nbx*self.nby
            self.hbx = self.hx*0.5
            self.hby = self.hy*0.5
            self.alpha = 6  # Nlb of trial function, # of local basis
            self.beta = 6  # Nlb of test function, # of local basis
        else:
            raise ValueError("wrong basis type")
        self.matP = self.getMatP()
        self.matT = self.getMatT()
        self.matPb = self.getMatPb()
        self.matTb = self.getMatTb()

    def getMatP(self):
        """
        ref: ppt ch2 p12, p8
        """
        Nmx = self.nmx  # nodes nubmer in x dir
        Nmy = self.nmy  # nodes number in y dir
        P = np.zeros((2, Nmx * Nmy))
        for rn in range(1, Nmy+1):  # row node index, [1, Nxm]
            for cn in range(1, Nmx+1): # col node index, [1, Nym]
                i = rn + (cn-1)*Nmy
                idx = i-1
                x = self.hx * (cn - 1) + self.xmin
                y = self.hy * (rn - 1) + self.ymin
                P[0, idx] = x
                P[1, idx] = y 
        return P 
    
    def getMatT(self):
        """
        ref: ppt ch2 p13, p8
        """
        Nmx = self.nmx  # nodes nubmer in x dir
        Nmy = self.nmy  # nodes number in y dir
        T = np.zeros((3 ,2*self.nx*self.ny), dtype=np.int)
        for re in range(1, self.ny + 1):  # [1, ny], row element index
            for ce in range(1, self.nx + 1):  # [1, nx], column element index
                # 4 --- 3
                # |     |
                # |     |
                # 1 --- 2
                p1 = re + (ce-1)*Nmy
                p4 = p1 + 1
                p2 = re + ce*Nmy
                p3 = p2 + 1
                ne = re + (ce-1)*self.ny  # global index of rectangle element
                netri1 = (ne-1)*2 + 1  # left corner triangle element index
                netri2 = (ne-1)*2 + 2  # right corner triangle element index
                T[0, netri1-1] = p1
                T[1, netri1-1] = p2
                T[2, netri1-1] = p4
                T[0, netri2-1] = p4
                T[1, netri2-1] = p2
                T[2, netri2-1] = p3
        return T

    def getMatPb(self):
        """
        ref: ppt p59
        """
        if self.bt == 201:  # linear basis
            return self.getMatP()
        elif self.bt == 202:  # quadratic basis
            Pb = np.zeros((2, self.nbx*self.nby))
            for rn in range(1, self.nby+1):  # [1, Nby], row node index
                for cn in range(1, self.nbx+1):  # [1,Nbx], column node index
                    i = self.nby*(cn-1) + rn
                    idx = i-1
                    Pb[0, idx] = self.xmin+self.hbx*(cn-1)
                    Pb[1, idx] = self.ymin+self.hby*(rn-1)
            return Pb
        else:
            raise ValueError("wrong basis type")

    def getMatTb(self):
        """
        ref: ppt p60
        """
        if self.bt == 201:  # linear basis 
            return self.getMatT()
        elif self.bt == 202:  # quadratic basis
            Tb = np.zeros((6, 2*self.nx*self.ny), dtype=np.int)
            for re in range(1, self.ny+1):   # [1, ny], row rectangle element index
                for ce in range(1, self.nx+1):  # [1, nx], column rectangle element index
                    # 7 --- 6 --- 5
                    # |           |
                    # 8     9     4
                    # |           |
                    # 1 --- 2 --- 3
                    ne = re + (ce-1)*self.ny  # global index of rectangle element
                    netri1 = (ne-1)*2 + 1  # left corner triangle element index
                    netri2 = (ne-1)*2 + 2  # right corner triangle element index
                    p1 = (re*2-1) + (ce-1)*self.nby*2
                    p8 = p1+1
                    p7 = p1+2
                    p2 = p1 + self.nby
                    p9 = p2+1
                    p6 = p2+2
                    p3 = p1 + 2*self.nby
                    p4 = p3+1
                    p5 = p3+2
                    Tb[0,netri1-1] = p1; Tb[1,netri1-1] = p3; Tb[2,netri1-1] = p7
                    Tb[3,netri1-1] = p2; Tb[4,netri1-1] = p9; Tb[5,netri1-1] = p8
                    Tb[0,netri2-1] = p7; Tb[1,netri2-1] = p3; Tb[2,netri2-1] = p5
                    Tb[3,netri2-1] = p9; Tb[4,netri2-1] = p4; Tb[5,netri2-1] = p6                    
            return Tb
        else:
            raise ValueError("wrong basis type")
    
    def getMatBCedgeAndNode(self, ltype=-1, rtype=-1, btype=-1, ttype=-1):
        """
        :param ltype: left boundary type
        :type ltype: int
        :param rtype: right boundary type
        :type rtype: int
        :param btype: bottom boundary type
        :type btype: int
        :param ttype: top boundary type
        :type ttype: int

        ref: ch2 ppt p15, p16
        bcEdge: mesh boudary edge matrix, only relates to mesh
        bcEdge[0, k] : type of kth boundary edge
        bcEdge[1, k] : index of element contains kth boundary edge 
        bcEdge[2, k] : global node index of the first end of kth boudnary edge 
        bcEdge[3, k] : global node index of the second end of kth boudnary edge 
        boundary type:
            -1: dirichlet 
            -2: neumann
            -3: robin

        ref: ppt ch2 p40(1d basis), p62(2d basis)
        bcNode: boundary node number matrix, only relates to FE space
        bcNode[0, k]: type of kth boundary FE node
        bcNode[1, k]: node index of kth boundary FE node
        boundary type:
            -1: dirichlet 
            -2: neumann
            -3: robin
        """
        bcEdge = np.zeros((4, self.nx*2 + self.ny*2), dtype=int)
        bcNode = np.zeros((2, 2*(self.nbx+self.nby)-4), dtype=int)
        # calc bcEdge
        # bottom boundary edges
        for i in range(1, self.nx+1):  # [1, nx]
            idx = i - 1
            bcEdge[0, idx] = btype
            bcEdge[1, idx] = 2*self.ny*(i-1) + 1
            bcEdge[2, idx] = (i-1)*(self.ny+1) + 1
            bcEdge[3, idx] = i*(self.ny+1) + 1
        # right boundary edges
        for i in range(self.nx+1, self.nx+self.ny+1):  # [nx+1, nx+ny]
            idx = i - 1
            temp = i - self.nx  # the 1th,2th,3th... of the first 
            bcEdge[0, idx] = rtype
            bcEdge[1, idx] = 2*temp + self.ny*(self.nx-1)*2
            bcEdge[2, idx] = (self.ny+1)*self.nx + temp
            bcEdge[3, idx] = (self.ny+1)*self.nx + temp + 1
        # top boudnary edges
        for i in range(self.nx+self.ny+1, self.nx*2+self.ny+1):  # [nx+ny+1,nx*2+ny]
            idx = i-1
            temp = i-(self.nx+self.ny)
            bcEdge[0, idx] = ttype
            bcEdge[1, idx] = self.ny*2*(self.nx+1-temp)
            bcEdge[2, idx] = (self.ny+1)*(self.nx+2-temp)
            bcEdge[3, idx] = (self.ny+1)*(self.nx+1-temp) 
        # left boudary edges
        for i in range(self.nx*2+self.ny+1, 2*(self.nx+self.ny)+1):  #[nx*2+ny+1, 2*(nx+ny)]
            idx = i-1
            temp = i-(self.nx*2+self.ny)
            bcEdge[0, idx] = ltype
            bcEdge[1, idx] = self.ny*2-(2*temp-1)
            bcEdge[2, idx] = self.ny+1-(temp-1)
            bcEdge[3, idx] = self.ny-(temp-1)
        
        # calc bcNode
        # bottom boundary
        for i in range(1, self.nbx+1):  # [1, nbx]
            idx = i-1
            bcNode[0,idx]=btype
            bcNode[1,idx]=self.nby*(i-1)+1
        # right boundary
        for i in range(self.nbx, self.nbx+self.nby-1+1): # [nbx, nbx+nby-1]
            idx = i-1
            temp = i-self.nbx
            bcNode[0, idx] = rtype
            bcNode[1, idx] = (self.nbx-1)*self.nby + temp + 1
        # top boundary
        for i in range(self.nbx+self.nby-1, self.nbx*2+self.nby-2+1): # [nbx+nby-1, nbx*2+nby-2]
            idx=i-1
            temp = i-(self.nbx+self.nby-1)
            bcNode[0, idx] = ttype
            bcNode[1, idx] = self.nby*(self.nbx-temp)
        # left boundary
        for i in range(self.nbx*2+self.nby-2, self.nbx*2+self.nby*2-4+1): #[nbx*2+nby-2, nbx*2+nby*2-4]
            idx=i-1
            temp = i-(self.nbx*2+self.nby-2)
            bcNode[0,idx] = ltype
            bcNode[1,idx] = self.nby - temp
        return bcEdge, bcNode


    def locBasis(self, x, y, vertMat, bIdx, dIdx_x, dIdx_y):
        """
        local basis function
        :param x: x coordinate
        :type x: double
        :param y: y coordinate
        :type y: double
        :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
        :type vertMat: np.array
        :param bIdx: basis index, 1<= bIdx <= alpha or beta or N_{lb}
        :type bIdx: int
        :param dIdx_x: derivative degree in x dir
        :type dIdx_x: int
        :param dIdx_y: derivative degree in y dir
        :type dIdx_y: int
        """
        return sopt._locBasis(self.bt, x, y, vertMat, bIdx, dIdx_x, dIdx_y)  


    def refBasis(self, xHat, yHat, bIdx, dIdx_x, dIdx_y):
        """
        reference basis function
        ref: ppt p32, p41-p56
        :param xHat: reference x
        :type xHat: double
        :param yHat: reference y
        :type yHat: double
        :param bIdx: basis index, 1<= bIdx <= alpha or beta or N_{lb}
        :type bIdx: int
        :param dIdx_x: derivative degree in x dir
        :type dIdx_x: int
        :param dIdx_y: derivative degree in y dir
        :type dIdx_y: int
        """
        
        bt = self.bt
        return sopt._refBasis(bt, xHat,yHat, bIdx, dIdx_x, dIdx_y)
       

    ######################################################
    def printMatP(self):
        print('*** matrix P is ')
        print(self.getMatP())

    def printMatT(self):
        print('*** matrix T is ')
        print(self.getMatT())

    def plotmesh(self, showNodeNum=True):
        P = self.getMatP()
        x = P[0, :]
        y = P[1, :]
        plt.figure()
        plt.scatter(x, y)
        # plt.plot(y, x)
        if showNodeNum:
            for i in range(P.shape[1]):
                xcor = P[0 ,i]
                ycor = P[1, i]
                num = i + 1
                plt.text(xcor, ycor, '%d'%(num))
        plt.xlabel('X')
        plt.ylabel('Y')  


class Solver2d(feVar2d):
    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, basisType, numGaussPoint=9, numGaussPoint1d=4):
        super().__init__(xmin, xmax, ymin, ymax, nx, ny, basisType)
        self.cRefGaussMat, self.pRefGaussMat=sopt._genRefTriGauss(numGaussPoint)
        self.cRefGaussMat1d, self.pRefGaussMat1d = sopt._genRefGauss1d(numGaussPoint1d)
    def getMatA(self, cFunc, rd, sd, pd, qd):
        """
        ref: ppt ch3 p39
        :param cFunc: coefficient function, ref ppt ch3 p4
        :type cFunc: function
        :param rd: partial derivative degree of trial function on x
        :type rd: int 
        :param sd: partial derivative degree of trial function on y
        :type sd: int
        :param pd: partial derivative degree of test function on x
        :type pd: int
        :param qd: partial derivative degree of test function on y
        :type qd: int
        """
        return sopt._genMatA(cFunc, self.Nb, self.N, self.matP, self.matT, self.matTb,
                             self.bt, self.alpha, self.beta, rd, sd, pd, qd, 
                             self.cRefGaussMat, self.pRefGaussMat
                             )


    def getVecb(self, fFunc, pd, qd):
        """
        ref: ppt ch3 p51
        :param fFunc: function f(x,y) on the R.H.S. of PDE, ref to ppt ch3 p4
        :type fFunc: function
        :param pd: partial derivative degree of test function on x
        :type pd: int
        :param qd: partial derivative degree of test function on y
        :type qd: int
        """
        
        return sopt._genVecb(fFunc, self.Nb,self.N, self.matP, self.matT, self.matTb,
                            self.bt, self.beta, pd,qd, self.cRefGaussMat, self.pRefGaussMat
                            )

    def treatDirichletBC(self, gFunc, matbcNode, matA, vecb):
        """
        ref: ppt ch3 p57
        :param gFunc: like u(x,y) = g(x,y)
        :type gFunc: function
        :param matbcNode: boundary node number matrix, only relates to FE space
        :type matbcNode: np.array
        :param matA: matrix A
        :type matA: np.array
        :param vecb: vector b
        :type vecb: np.array

        ref: ppt ch2 p40(1d basis), p62(2d basis)
        bcNode: boundary node number matrix, only relates to FE space
        bcNode[0, k]: type of kth boundary FE node
        bcNode[1, k]: node index of kth boundary FE node
        boundary type:
            -1: dirichlet 
            -2: neumann
            -3: robin
        """
        nbn = matbcNode.shape[1]  # number of boudnary nodes (FE nodes)
        for k in range(1, nbn+1): # [1, nbn]
            kdx = k-1
            if matbcNode[0,kdx] == -1:
                i = matbcNode[1,kdx]
                idx = i-1
                matA[idx,:] = 0
                matA[idx,idx] = 1
                x = self.matPb[0,idx]
                y = self.matPb[1,idx]
                vecb[idx] = gFunc(x,y)
        return matA, vecb


    def treatNeumannBC(self, cpFunc, matbcEdge, vecb, ad, bd):
        """
        ref: ppt ch3 p100
        :param cpFunc: coefficient function times Neumann bc function
        :type cpFunc: function
        :param matbcEdge: boundary node edge matrix, only relates to mesh
        :type matbcEdge: np.ndarray
        :param vecb: vector b
        :type vecb: np.array
        :param ad: the derivative degree of the test FE basis function with respect to x
        :type ad: int
        :param bd: the derivative degree of the test FE basis function with respect to y
        :type bd: int
        """
        nbe = matbcEdge.shape[1]
        for k in range(1, nbe+1):  # [1, nbe]
            kdx=k-1
            if matbcEdge[0,kdx] == -2:
                meshElementIdx = matbcEdge[1, kdx]
                vertMat = self.matP[:, self.matT[:, meshElementIdx-1]-1]
                endPoint1 = self.matP[:, matbcEdge[2,kdx]-1]
                endPoint2 = self.matP[:, matbcEdge[3,kdx]-1]
                for beta in range(1, self.beta+1):  #[1, beta]
                    r = sopt._GaussQuadTriLineIntegralTest(cpFunc, self.cRefGaussMat1d, self.pRefGaussMat1d,
                                                           endPoint1, endPoint2, vertMat, self.bt, beta, ad, bd)
                    i = self.matTb[beta-1,meshElementIdx-1] 
                    idx=i-1
                    vecb[idx] += r
                    pass
        return vecb

    def treatRobinBC(self, cqFunc, crFunc, matbcEdge, matA, vecb,ad, bd, md, sd, dd, ld):
        """
        :param cqFunc: coefficient function times q function, ref ppt p106 
        :type cqFunc: function
        :param crFunc: coefficient function times r function, ref ppt p106 
        :type crFunc: function
        :param matbcEdge: boundary node edge matrix, only relates to mesh
        :type matbcEdge: np.ndarray
        :param matA: matrix A
        :type matA: np.ndarray
        :param vecb: vector b
        :type vecb: np.ndarray
        :param ad: the derivative degree of the test FE basis function with respect to x //vecb
        :type bd: int
        :param bd: the derivative degree of the test FE basis function with respect to y //vecb
        :type bd: int   
        :param md: the derivative degree of the trial FE basis function with respect to x //matA
        :type md: int  
        :param sd: the derivative degree of the trial FE basis function with respect to y //matA
        :type sd: int   
        :param dd: the derivative degree of the test FE basis function with respect to x //matA
        :type dd: int  
        :param ld: the derivative degree of the test FE basis function with respect to y //matA
        :type ld: int  
        """
        nbe = matbcEdge.shape[1]
        for k in range(1, nbe+1):
            kdx = k-1
            if matbcEdge[0,kdx] == -3:  # Robin BC
                meshElementIdx = matbcEdge[1, kdx]
                vertMat = self.matP[:, self.matT[:, meshElementIdx-1]-1]
                endPoint1 = self.matP[:, matbcEdge[2,kdx]-1]
                endPoint2 = self.matP[:, matbcEdge[3,kdx]-1]
                # modify matA
                for alpha in range(1, self.alpha+1):
                    for beta in range(1, self.beta+1):
                        r = sopt._GaussQuadTriLineIntegralTrialTest(crFunc, self.cRefGaussMat1d, self.pRefGaussMat1d,
                                                                    endPoint1, endPoint2, vertMat, self.bt, alpha, beta, 
                                                                    md, sd, dd, ld)
                        i = self.matTb[beta-1, meshElementIdx-1]  # test 
                        j = self.matTb[alpha-1, meshElementIdx-1] # trial
                        idx = i-1
                        jdx = j-1
                        matA[idx, jdx] += r

                # modify vecb
                for beta in range(1, self.beta+1):  
                        r = sopt._GaussQuadTriLineIntegralTest(cqFunc, self.cRefGaussMat1d, self.pRefGaussMat1d,
                                                           endPoint1, endPoint2, vertMat, self.bt, beta, ad, bd)
                        i = self.matTb[beta-1,meshElementIdx-1] 
                        idx=i-1
                        vecb[idx] += r
        return matA, vecb


    def solAxeqb(self, matA, vecb):
        matA=matA.tocsr()
        return scipy.sparse.linalg.spsolve(matA, vecb)
        # return np.linalg.solve(matA, vecb)
    
    def getSol(self, x, y, sol):
        """
        return solution u(x,y) result at any given (x,y)
        :param x: given point's x coordinate
        :type x: double 
        :param y: given point's y coordinate
        :type y: double 
        :param sol: FE solution
        :type sol: np.array
        """
        r = 0
        for n in range(1, self.N+1):  # [1,N]
            vertMat = self.matP[:, (self.matT[:,n-1]-1)]
            for alpha in range(1, self.alpha+1): #[1,alpha], sum trial basis function
                r += self.locBasis(x,y,vertMat,alpha,0,0)*sol[self.matTb[alpha-1,n-1]-1]
        return r
       

    def tridblquad(self, func, vertMat):
        """
        integrate on a triangle mesh 
        ref: http://connor-johnson.com/2014/03/09/integrals-over-arbitrary-triangular-regions-for-fem/
        we do the mapping that maps triangle to rectangle:
            (u,v)=(0,0) -> p1 (x1, y1)
            (u,v)=(0,1) -> p1 (x1, y1)
            (u,v)=(1,0) -> p2 (x2, y2)
            (u,v)=(1,1) -> p3 (x3, y3)
            x = (1-u)*x1 + u*((1-v)*x2+v*x3)
            y = (1-u)*y1 + u*((1-v)*y2+v*y3)
            dxdu = -x1 + (1-v)*x2+v*x3
            dxdv = -u*x2 + u*x3
            dydu = -y1+(1-v)*y2+v*y3
            dydv = -u*y2 + u*y3
            J = ((dxdu, dxdv), 
                 (dydu, dydv))
            detJ = dxdu * dydv - dxdv * dydu
            \int \int func(x,y) dxdy = \int \int func(x(u,v), y(u,v)) detJ dudv
        :param func: func(x,y), we need to integrate \int \int func(x,y) dxdy
        :type func: function
        :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  
                vertMat[1,0] = y1
                vertMat[0,1] = x2  
                vertMat[1,1] = y2
                vertMat[0,2] = x3  
                vertMat[1,2] = y3
        :type vertMat: np.array

        # triangle integral method1:
        # https://stackoverflow.com/questions/50712589/numerical-integration-over-a-mesh-in-python-with-existing-mesh
        # http://connor-johnson.com/2014/03/09/integrals-over-arbitrary-triangular-regions-for-fem/
        # https://tutorial.math.lamar.edu/Classes/CalcIII/ChangeOfVariables.aspx
            # method 2: (use quadpy)
            # https://pypi.org/project/quadpy/

        """
        x1 = vertMat[0,0]; y1 = vertMat[1,0]
        x2 = vertMat[0,1]; y2 = vertMat[1,1]
        x3 = vertMat[0,2]; y3 = vertMat[1,2]

        def dJ(u,v):
            dxdu = -x1 + (1-v)*x2+v*x3
            dxdv = -u*x2 + u*x3
            dydu = -y1+(1-v)*y2+v*y3
            dydv = -u*y2 + u*y3
            return np.abs(dxdu * dydv - dxdv * dydu)

        def newfunc(u, v):
            x = (1-u)*x1 + u*((1-v)*x2+v*x3)  # transform (u,v) to (x,y)
            y = (1-u)*y1 + u*((1-v)*y2+v*y3)
            r = func(x,y)  # \int \int func(x(u,v), y(u,v)) detJ dudv
            r = r*dJ(u,v)
            return r
        r, err = sciIntegrate.dblquad(newfunc, 0,1, lambda x: 0, lambda x:1)
        return r,err











