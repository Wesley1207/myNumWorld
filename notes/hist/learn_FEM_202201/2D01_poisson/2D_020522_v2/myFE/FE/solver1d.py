import numpy as np
import scipy.integrate as sciIntegrate

class feVar1d:
    def __init__(self, matP, basisType):
        """
        Initialize fem vars
        :param matP: P matrix
        :type matP: np.array, double
        :param matT: T matrix
        :type matT: np.array, int
        :param basisType:
            101: linear FE basis
            102: quadratic FE basis
        :type basisType: int
        """
        self.N = matP.shape[0] - 1  # mesh element number
        self.Nm = self.N+1  # mesh node number
        self.matP = matP
        self.matT = self.getMatT()
        self.bt = basisType # basis type
        if self.bt == 101:
            # self.Nb = self.N  # number of FE element
            # self.Nbm = self.Nb+1  # number of all basis functions
            self.Nb = self.N + 1 # number of all basis functions
            self.alpha = 2  # number of local trial basis
            self.beta = 2   # number of local test basis
        elif self.bt == 102:
            # self.Nb = self.N *2
            # self.Nbm = self.Nb+1
            self.Nb = self.N * 2 + 1 # number of all basis functions
            self.alpha = 3  # number of local trial basis
            self.beta = 3   # number of local test basis
        else:
            raise ValueError("wrong basis type")
    
    def getMatT(self):
        T = np.zeros((2, self.N), dtype=np.int)
        for i in range(1, self.N+1):  # [1, N]
            idx = i-1
            T[0, idx] = i
            T[1, idx] = i+1
        return T
    
    def getMatPb(self):
        if self.bt == 101:
            return self.matP
        elif self.bt == 102:
            Pb = np.zeros(self.Nb)
            # matP: 1 --- 2 --- 3 --- 4 --- 5
            # new:  1 -2- 3 -4- 5 -6- 7 -8- 9
            for i in range(1, self.Nm+1):  # [1, Nm]
                idx = i-1
                kdx = (2*i-1) - 1
                Pb[kdx] = self.matP[idx]  # update odd index 
            for i in range(1, self.Nb+1): # [1, Nbm]
                if i % 2 == 0:
                    idx = i-1
                    Pb[idx] = (Pb[idx-1] + Pb[idx+1]) * 0.5
            return Pb
        else:
            raise ValueError("wrong basis type")
    
    def getMatTb(self):
        if self.bt == 101:
            Tb = np.zeros((2, self.N), dtype=np.int)
            for i in range(1, self.N+1):  # [1, N]
                idx = i-1
                Tb[0, idx] = i
                Tb[1, idx] = i+1
            return Tb
        elif self.bt == 102:
            Tb = np.zeros((3, self.N), dtype=np.int)
            for i in range(1, self.N+1):  # [1, N]
                idx = i - 1
                Tb[0, idx] = 2*i - 1  # left node
                Tb[1, idx] = 2*i + 1  # right node
                Tb[2, idx] = 2*i  # middle node
            return Tb
        else:
            raise ValueError("wrong basis type")
    
    ######################################################
    def printMatPb(self):
        print("*** matrix Pb is")
        print(self.getMatPb())
    
    def printMatTb(self):
        print("*** matrix Tb is")
        print(self.getMatTb())


class Solver1d(feVar1d):
    def __init__(self, matP, basisType):
        super().__init__(matP, basisType)
    
    def getMatA(self, cFunc, rdIdx, sdIdx):
        """
        here we do not distinguish trial and test basis functions.
        (trial and test use the same Tb)
        Corresponds to ppt page 67, Algorithm 4. 
        :param cFunc: coefficient function c(x), ref to ppt p4
        :type cFunc: function
        :param rdIdx: trial basis function derivative degree
        :type rdIdx: int
        :param sdIdx: test basis function derivative degree
        :type sdIdx: int
        """
        Tb = super().getMatTb()
        A = np.zeros((self.Nb, self.Nb), dtype=np.double)
        for n in range(1, self.N+1):  # [1, N]
            a = np.min(self.matP[(self.matT[:, n-1]) - 1]) # lower bound of nth element
            b = np.max(self.matP[(self.matT[:, n-1]) - 1]) # upper bound of nth element
            for alpha in range(1, self.alpha+1):  # [1,alpha], trial
                for beta in range(1, self.beta+1):  # [1,beta], test
                    r = sciIntegrate.quad(lambda x: 
                                         cFunc(x) *
                                         self.basisPsi(x, a, b, alpha, rdIdx) *
                                         self.basisPsi(x, a, b, beta, sdIdx),
                                         a, b
                                         )
                    r = r[0]
                    i = Tb[beta-1, n-1]
                    j = Tb[alpha-1, n-1]
                    idx = i-1
                    jdx = j-1
                    A[idx, jdx] += r
        return A
    
    def getVecb(self, sdIdx, fFunc):
        """
        here we do not distinguish trial and test basis functions.
        (trial and test use the same Tb)
        Corresponds to ppt page 79, Algorithm 5. 
        :param sdIdx: test basis function derivative degree
        :type sdIdx: int
        :param fFunc: function f(x) on the R.H.S. of PDE, ref to ppt p4
        :type fFunc: function
        """
        Tb = super().getMatTb()
        bVec = np.zeros(self.Nb, dtype=np.double)
        for n in range(1, self.N+1):  # [1, N]
            a = np.min(self.matP[(self.matT[:, n-1]) - 1]) # lower bound of nth element
            b = np.max(self.matP[(self.matT[:, n-1]) - 1]) # upper bound of nth element
            for beta in range(1, self.beta+1):  # [1,beta], test
                r = sciIntegrate.quad(lambda x: 
                                         fFunc(x) *
                                         self.basisPsi(x, a, b, beta, sdIdx),
                                         a, b
                                         )
                r = r[0]
                i = Tb[beta-1, n-1]
                idx = i-1
                bVec[idx] += r
        return bVec

    def solAxeqb(self, matA, vecb):
        return np.linalg.solve(matA, vecb)
        
    def getSol(self, x, xsol):
        """
        return solution u(x) result at any given x
        :param x: given point
        :type x: double 
        :param xsol: FE solution
        :type xsol: np.array
        """
        sol = 0
        Tb = super().getMatTb()
        for n in range(1, self.N+1):  # [1, N]
            a = np.min(self.matP[(self.matT[:, n-1]) - 1]) # lower bound of nth element
            b = np.max(self.matP[(self.matT[:, n-1]) - 1]) # upper bound of nth element
            if x >= a and x <= b:
                for alpha in range(1, self.alpha+1):  # [1,alpha], loop trial
                    sol += self.basisPsi(x, a, b, alpha, 0) * xsol[Tb[alpha-1, n-1] - 1]
            else:
                continue
        return sol
    
    def _affine(self, x, a, b):
        """
        affine transformation: x -> xHat 
        :type x: double
        :param n: element number of mesh
        :type n: int
        psiHat \in [0,1], a = xn, b=x_{n+1}
            #   xHat = a < x < b, 
            #   0 < x-a < b - a
            #   0 < (x-a) / (b-a) < 1
            # xHat = (x-a) / (b-a)
        """
        if x >= a and x <= b:
            xHat = (x - a) / (b - a)
            dxHatdx = 1. / (b-a)
        else:
            xHat = 0
            dxHatdx = 0
        return xHat, dxHatdx

    def basisPsi(self, x, a, b, bIdx, dIdx):
        """
        local basis function Psi
        :type x: double
        :param a: lower bound of the element
        :type a: double
        :param b: upper bound of the element
        :type b: double
        :param bIdx: basis index, 1<= bIdx <= alpha or beta or N_{lb}
        :type bIdx: int
        :param dIdx: derivative degree
        :type dIdx: int
        """
        xHat, dxHatdx = self._affine(x, a, b)
        r = 0  # result

        if self.bt == 101:
            if dIdx == 0: 
                # psiHat1 = 1-xHat
                # psiHat2 = xHat
                # ppt page 102
                if bIdx == 1:
                    r = 1 - xHat
                elif bIdx == 2:
                    r = xHat
                else:
                    raise ValueError("wrong basis function index")
            elif dIdx == 1:
                # d psiHat(xHat) / dx = (dpsiHat / dxHat) * (dxHat/dx)
                if bIdx == 1:
                    r = -1.*dxHatdx
                elif bIdx == 2:
                    r = 1.*dxHatdx
                else:
                    raise ValueError("wrong basis function index")
            else:
                r = 0
        elif self.bt == 102:
            if dIdx == 0:
                # [A1,A3,A2]
                # psiHat1 = 2 * xHat**2 -3*xHat + 1
                # psiHat2 = 2* xHat**2 - xHat
                # psiHat3 = -4*xHat**2 + 4*xHat
                # ppt page 109
                if bIdx == 1:
                    r = 2. * xHat**2 -3 * xHat + 1
                elif bIdx == 2:
                    r = 2.* xHat**2 - xHat
                elif bIdx == 3:
                    r = -4.*xHat**2 + 4*xHat
                else:
                    raise ValueError("wrong basis function index")
            elif dIdx == 1:
                if bIdx == 1:
                    r = (4.*xHat - 3.) * dxHatdx
                elif bIdx == 2:
                    r = (4.*xHat - 1.) * dxHatdx
                elif bIdx == 3:
                    r = (-8.*xHat+ 4.) * dxHatdx
                else:
                    raise ValueError("wrong basis function index")
            elif dIdx == 2:
                if bIdx == 1:
                    r = 4. * dxHatdx**2
                elif bIdx == 2:
                    r = 4.* dxHatdx**2
                elif bIdx == 3:
                    r = -8. * dxHatdx**2
                else:
                    raise ValueError("wrong basis function index")
            else:
                r = 0
        else:
            raise ValueError("wrong basis type")
        return r


class TreatBC1d(feVar1d):
    def __init__(self, matP, basisType, bcnodes):
        """
        :param bcnodes: an array contains bc info, 
            bc_node_vec[0, k]: type of kth boundary node
                        -1: Dirichlet boundary node
                        -2: Neumann boundary node
                        -3: Robin boundary node
            bc_node_vec[1, k]: global index of kth boundary node(FE)
            bc_node_vec[2, k]: normal direction of kth boundary node(FE)
        :type bcnodes: array, shape:(3, 2)
        """
        super().__init__(matP, basisType)
        self.bcnodes = bcnodes
        self.nbn = 2  # number of boundary nodes
    
    def dirichlet(self, gFunc, matA, vecb):
        """
        ref: ppt p81
        :param gFunc: like u(a) = g(a), u(b) = g(b)
        :type gFunc: function
        """
        Pb = self.getMatPb()
        for k in range(1, self.nbn+1):  #[1,nbn]
            kdx = k - 1
            if self.bcnodes[0, kdx] == -1:
                i = self.bcnodes[1, kdx]
                idx = i - 1
                matA[idx, :] = 0
                matA[idx, idx] = 1
                vecb[idx] = gFunc(Pb[idx])
        return matA, vecb

    def neumann(self, rFunc, cFunc, vecb):
        """
        ref: ppt p128, p129
        :param rFunc: like u'(a)=r(a), u'(b) = r(b)
        :type rFunc: function
        :param cFunc: coefficient function c(x), ref to ppt p4
        :type cFunc: function
        # source code of the course is wrong? no coefficient function in the source code !
        """
        Pb = self.getMatPb()
        for k in range(1, self.nbn+1):  #[1,nbn]
            kdx = k-1
            if self.bcnodes[0, kdx] == -2:
                norm_dir = self.bcnodes[2, kdx]
                i = self.bcnodes[1, kdx]
                idx = i - 1
                vecb[idx] += rFunc(Pb[idx]) * cFunc(Pb[idx]) * norm_dir
                # vecb[idx] += rFunc(Pb[idx]) *  norm_dir
        return vecb

    def robin(self, qFunc, pFunc, cFunc, matA, vecb):
        """
        ref: ppt p132
        :param rFunc: like u'(a)+q(a)*u(a)=p(a), u'(b)+q(b)*u(b)=p(b)
        :type rFunc: function
        :param cFunc: coefficient function c(x), ref to ppt p4
        :type cFunc: function
        # source code of the course is wrong? no coefficient function in the source code !
        """
        Pb = self.getMatPb()
        for k in range(1, self.nbn+1):  #[1,nbn]
            kdx = k-1
            if self.bcnodes[0, kdx] == -3:
                norm_dir = self.bcnodes[2, kdx]
                i = self.bcnodes[1, kdx]
                idx = i - 1
                matA[idx, idx] += qFunc(Pb[idx]) * cFunc(Pb[idx]) * norm_dir
                vecb[idx] += pFunc(Pb[idx]) * cFunc(Pb[idx]) * norm_dir
        return matA, vecb


        

