import numpy as np
from numba import jit
import scipy.sparse as sparse


@jit(nopython=True, cache=True)
def _locBasis(bt, x, y, vertMat, bIdx, dIdx_x, dIdx_y):
    """
    local basis function
    :param bt: basis type
    :type bt: int
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
    # affine mapping from (x,y) to (xHat, yHat)
    x1 = vertMat[0,0]
    y1 = vertMat[1,0]
    x2 = vertMat[0,1]
    y2 = vertMat[1,1]
    x3 = vertMat[0,2]
    y3 = vertMat[1,2]
    JMat = np.array(
                        [[x2-x1, x3-x1],
                        [y2-y1, y3-y1]]
                    )  # ppt p43
    jacobin = abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    xHat = ((y3-y1)*(x-x1)-(x3-x1)*(y-y1)) / jacobin
    yHat = (-(y2-y1)*(x-x1) + (x2-x1)*(y-y1)) / jacobin

    dxHdx = (y3-y1) / jacobin
    dyHdx = (y1-y2) / jacobin
    dxHdy = (x1-x3) / jacobin
    dyHdy = (x2-x1) / jacobin

    if dIdx_x == 0 and dIdx_y==0:
        return _refBasis(bt, xHat, yHat, bIdx, 0, 0)
    elif dIdx_x == 0 and dIdx_y==1:
        return (_refBasis(bt, xHat, yHat, bIdx, 1, 0)*dxHdy +
                _refBasis(bt, xHat, yHat, bIdx, 0, 1)*dyHdy 
                )
    elif dIdx_x == 0 and dIdx_y==2:
        return ( _refBasis(bt, xHat, yHat, bIdx, 2, 0)*dxHdy**2 + 
                    _refBasis(bt, xHat, yHat, bIdx, 0, 2)*dyHdy**2 + 
                    _refBasis(bt, xHat, yHat, bIdx, 1, 1)*dxHdy*dyHdy*2
                )
    elif dIdx_x == 1 and dIdx_y==0:
        return ( _refBasis(bt, xHat, yHat, bIdx, 1, 0)*dxHdx +
                    _refBasis(bt, xHat, yHat, bIdx, 0, 1)*dyHdx
        )
    elif dIdx_x == 1 and dIdx_y==1:
        return ( _refBasis(bt, xHat, yHat, bIdx, 2, 0)*dxHdx*dxHdy+
                    _refBasis(bt, xHat, yHat, bIdx, 0, 2)*dyHdx*dyHdy+
                    _refBasis(bt, xHat, yHat, bIdx, 1, 1)*(dxHdy*dyHdx+dxHdx*dyHdy)
                )
    elif dIdx_x == 2 and dIdx_y==0:
        return ( _refBasis(bt, xHat, yHat, bIdx, 2, 0)*dxHdx**2 +
                    _refBasis(bt, xHat, yHat, bIdx, 0, 2)*dyHdx**2 + 
                    _refBasis(bt, xHat, yHat, bIdx, 1, 1)*dxHdx*dyHdx*2
                )
    else:
        return 0



@jit(nopython=True, cache=True)
def _refBasis(bt, xHat, yHat, bIdx, dIdx_x, dIdx_y):
    """
    reference basis function
    ref: ppt p32, p41-p56
    :param bt: basis type
    :type bt: int
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
    x = xHat
    y = yHat
    bt = bt

    if x<0 or x>1 or y<0 or y>1 or y>(-x+1):  # point is not in triangle
        return 0.

    if bt == 201:  # linear basis function
        # 3 ref points: (0,0)->bIdx=1, (1,0)->bIdx=2, (0,1)->bIdx=3
        # ppt p32
        if dIdx_x==0 and dIdx_y==0:
            if bIdx == 1:
                return (-x-y+1.)
            elif bIdx == 2:
                return x
            elif bIdx == 3:
                return y
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==0 and dIdx_y==1:
            if bIdx == 1:
                return -1.
            elif bIdx == 2:
                return 0.
            elif bIdx == 3:
                return 1.
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==1 and dIdx_y==0:
            if bIdx == 1:
                return -1.
            elif bIdx == 2:
                return 1.
            elif bIdx == 3:
                return 0.
            else:
                raise ValueError("wrong local basis index")
        else:
            return 0
    elif bt == 202:  # quadratic basis function
        # 6 ref points: (0,  0)->bIdx=1, (1,  0  )->bIdx=2, (0,1  )->bIdx=2,
        #               (0.5,0)->bIdx=4, (0.5,0.5)->bIdx=5, (0,0.5)->bIdx=6
        # ppt p55
        if dIdx_x==0 and dIdx_y==0:
            if bIdx==1:
                return (2*x**2 + 2*y**2 + 4*x*y - 3*y -3*x +1)
            elif bIdx==2:
                return (2.*x**2 - x)
            elif bIdx==3:
                return (2.*y**2 - y)
            elif bIdx==4:
                return (-4.*x**2 - 4.*x*y +4*x)
            elif bIdx==5:
                return (4.*x*y)
            elif bIdx==6:
                return (-4.*y**2-4*x*y+4*y)
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==0 and dIdx_y==1:
            if bIdx==1:
                return (4.*y + 4*x - 3)
            elif bIdx==2:
                return 0
            elif bIdx==3:
                return (4.*y-1)
            elif bIdx==4:
                return (-4.*x)
            elif bIdx==5:
                return (4.*x)
            elif bIdx==6:
                return (-8.*y-4*x+4)
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==0 and dIdx_y==2:
            if bIdx==1:
                return 4.
            elif bIdx==2:
                return 0.
            elif bIdx==3:
                return 4.
            elif bIdx==4:
                return 0.
            elif bIdx==5:
                return 0.
            elif bIdx==6:
                return -8.
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==1 and dIdx_y==0:
            if bIdx==1:
                return (4.*x + 4*y - 3)
            elif bIdx==2:
                return (4.*x - 1)
            elif bIdx==3:
                return 0.
            elif bIdx==4:
                return (-8*x -4*y + 4)
            elif bIdx==5:
                return (4.*y)
            elif bIdx==6:
                return (-4.*y)
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==1 and dIdx_y==1:
            if bIdx==1:
                return 4.
            elif bIdx==2:
                return 0.
            elif bIdx==3:
                return 0.
            elif bIdx==4:
                return -4.
            elif bIdx==5:
                return 4.
            elif bIdx==6:
                return -4.
            else:
                raise ValueError("wrong local basis index")
        elif dIdx_x==2 and dIdx_y==0:
            if bIdx==1:
                return 4.
            elif bIdx==2:
                return 4.
            elif bIdx==3:
                return 0.
            elif bIdx==4:
                return -8.
            elif bIdx==5:
                return 0.
            elif bIdx==6:
                return 0.
            else:
                raise ValueError("wrong local basis index")
    else: 
        raise ValueError("wrong basis type")



def _genMatA(cFunc, Nb, N, matP, matT, matTb, basisType, Alpha, Beta, rd, sd, pd, qd, cRefTriGaussMat, pRefTriGaussMat):
    """
    ref: ppt ch3 p39
    :param cFunc: coefficient function, ref ppt ch3 p4
    :type cFunc: function
    :param Nb: number of basis function
    :type Nb: int
    :param N: number of mesh element (triangle mesh: N=nx*ny*2)
    :type N: int
    :param matT: mesh T metrix
    :type matT: np.array
    :param matTb: FE space T metrix
    :type matTb: np.array
    :param basisType: basis type
    :type basisType: int
    :param Alpha: total number of trial function
    :type Alpha: int
    :param Beta: total number of test function
    :type Beta: int
    :param rd: partial derivative degree of trial function on x
    :type rd: int 
    :param sd: partial derivative degree of trial function on y
    :type sd: int
    :param pd: partial derivative degree of test function on x
    :type pd: int
    :param qd: partial derivative degree of test function on y
    :type qd: int
    :param cRefTriGaussMat: the Gauss coefficients on the reference triangle
    :type cRefTriGaussMat: np.array
    :param pRefTriGaussMat: the Gauss points on the reference triangle
    :type pRefTriGaussMat: np.array
    """
    # matA = np.zeros((Nb,Nb))
    matA = sparse.lil_matrix((Nb,Nb),dtype=np.double)
    for n in range(1, N+1): # [1,N]
        vertMat = matP[:, (matT[:,n-1]-1)]
        cLocTriGaussMat, pLocTriGaussMat = _genLocTriGauss(cRefTriGaussMat, pRefTriGaussMat, vertMat)
        for alpha in range(1, Alpha+1): # [1, alpha], trial local basis
            for beta in range(1, Beta+1):  # [1, beta], test local basis
                r = _GaussQuadTriVolIntegralTrialTest(cFunc, cLocTriGaussMat, pLocTriGaussMat,
                                                     vertMat, basisType, alpha, beta,
                                                     rd, sd, pd, qd
                                                     )
                i = matTb[beta-1, n-1]  # test 
                j = matTb[alpha-1, n-1] # trial
                idx = i-1
                jdx = j-1
                matA[idx, jdx] += r
    return matA


@jit(nopython=True, cache=True)
def _genVecb(fFunc, Nb, N, matP, matT, matTb, basisType, Beta, pd, qd, cRefTriGaussMat, pRefTriGaussMat):
    """
    ref: ppt ch3 p51
    :param fFunc: function f(x,y) on the R.H.S. of PDE, ref to ppt ch3 p4
    :type fFunc: function
    :param Nb: number of basis function
    :type Nb: int
    :param N: number of mesh element (triangle mesh: N=nx*ny*2)
    :type N: int
    :param matT: mesh T metrix
    :type matT: np.array
    :param matTb: FE space T metrix
    :type matTb: np.array
    :param basisType: basis type
    :type basisType: int
    :param Beta: total number of test function
    :type Beta: int
    :param pd: partial derivative degree of test function on x
    :type pd: int
    :param qd: partial derivative degree of test function on y
    :type qd: int
    :param cRefTriGaussMat: the Gauss coefficients on the reference triangle
    :type cRefTriGaussMat: np.array
    :param pRefTriGaussMat: the Gauss points on the reference triangle
    :type pRefTriGaussMat: np.array
    """
    vecb = np.zeros(Nb, dtype=np.double)
    for n in range(1, N+1):  # [1,N]
        vertMat = matP[:, (matT[:,n-1]-1)]
        cLocTriGaussMat, pLocTriGaussMat = _genLocTriGauss(cRefTriGaussMat, pRefTriGaussMat, vertMat)
        for beta in range(1, Beta+1):
            r = _GaussQuadTriVolIntegralTest(fFunc, cLocTriGaussMat,pLocTriGaussMat,vertMat,
                                            basisType, beta, pd, qd
                                            )
            i = matTb[beta-1, n-1]
            idx = i-1
            vecb[idx] += r
    return vecb


@jit(nopython=True, cache=True)
def _GaussQuadTriVolIntegralTrialTest(cFunc, cLocTriGaussMat, pLocTriGaussMat, vertMat,
                                      basisType,alpha, beta, rd, sd, pd, qd):
    """
    :param cFunc: coefficient function, ref ppt ch3 p4
    :type cFunc: function
    :param cLocTriGaussMat: local triangle Gauss coefficient matrix
    :type cLocTriGaussMat: np.array
    :param pLocTriGaussMat: local triangle Gauss point matrix
    :type pLocTriGaussMat: np.array
    :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
    :type vertMat: np.array
    :param basisType: basis type
    :type basisType: int
    :param alpha: alpha's local trial function
    :type alpha: int
    :param beta: beta's local test function
    :type beta: int
    :param rd: partial derivative degree of trial function on x
    :type rd: int 
    :param sd: partial derivative degree of trial function on y
    :type sd: int
    :param pd: partial derivative degree of test function on x
    :type pd: int
    :param qd: partial derivative degree of test function on y
    :type qd: int
    """
    gpn = cLocTriGaussMat.shape[0]  # number of the Gauss points of the Gauss quadrature we are using
    r = 0.
    for i in range(1, gpn+1):
        idx = i-1
        x = pLocTriGaussMat[idx,0]
        y = pLocTriGaussMat[idx,1]
        r += cLocTriGaussMat[idx]*cFunc(x,y)*\
            _locBasis(basisType,x,y,vertMat,alpha,rd,sd)*\
            _locBasis(basisType,x,y,vertMat,beta,pd,qd)
    return r


@jit(nopython=True, cache=True)
def _GaussQuadTriVolIntegralTest(fFunc, cLocTriGaussMat, pLocTriGaussMat, vertMat,
                                      basisType, beta, pd, qd):
    """
    ref: ppt ch3 p51
    :param fFunc: function f(x,y) on the R.H.S. of PDE, ref to ppt ch3 p4
    :type fFunc: function
    :param cLocTriGaussMat: local triangle Gauss coefficient matrix
    :type cLocTriGaussMat: np.array
    :param pLocTriGaussMat: local triangle Gauss point matrix
    :type pLocTriGaussMat: np.array
    :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
    :type vertMat: np.array
    :param basisType: basis type
    :type basisType: int
    :param beta: beta's local test function
    :type beta: int
    :param pd: partial derivative degree of test function on x
    :type pd: int
    :param qd: partial derivative degree of test function on y
    :type qd: int
    """
    gpn = cLocTriGaussMat.shape[0]  # number of the Gauss points of the Gauss quadrature we are using
    r = 0.
    for i in range(1, gpn+1):
        idx = i-1
        x = pLocTriGaussMat[idx,0]
        y = pLocTriGaussMat[idx,1]
        r += cLocTriGaussMat[idx]*fFunc(x,y)*\
            _locBasis(basisType,x,y,vertMat,beta,pd,qd)
    return r



@jit(nopython=True, cache=True)
def _genLocTriGauss(cRefTriGaussMat, pRefTriGaussMat, vertMat):
    """
    Generate the Gauss coefficients and Gauss points on the 
    local triangle by using affine tranformation
    :param cRefTriGaussMat: the Gauss coefficients on the reference triangle
    :type cRefTriGaussMat: np.array
    :param pRefTriGaussMat: the Gauss points on the reference triangle
    :type pRefTriGaussMat: np.array
    :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
    :type vertMat: np.array
    """
    # affine mapping from (x,y) to (xHat, yHat)
    x1 = vertMat[0,0]
    y1 = vertMat[1,0]
    x2 = vertMat[0,1]
    y2 = vertMat[1,1]
    x3 = vertMat[0,2]
    y3 = vertMat[1,2]
    jacobin = abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    cLocTriGaussMat = jacobin * cRefTriGaussMat
    pLocTriGaussMat = np.zeros(pRefTriGaussMat.shape)
    pLocTriGaussMat[:,0] = x1+(x2-x1)*pRefTriGaussMat[:,0]+(x3-x1)*pRefTriGaussMat[:,1];
    pLocTriGaussMat[:,1] = y1+(y2-y1)*pRefTriGaussMat[:,0]+(y3-y1)*pRefTriGaussMat[:,1];
    return cLocTriGaussMat, pLocTriGaussMat



@jit(nopython=True, cache=True)
def _genRefTriGauss(numGaussPoint):
    """
    Generate the Gauss coefficients and Gauss points on the reference 
    triangle whose vertices are (0,0),(1,0),(0,1)
    :param numGaussPoint: the number of Gauss points in the formula. The Gauss formula depends on it.
    :type numGaussPoint: int
    """
    if numGaussPoint==3:
        coeffGaussMat = np.array([1/6,1/6,1/6])
        pointGaussMat = np.array([
                                 [1/2,0],
                                 [1/2,1/2],
                                 [0,1/2]
                                 ]
                                 )
    elif numGaussPoint==4:
        coeffGaussMat = np.array([(1.-1/np.sqrt(3))/8, (1.-1/np.sqrt(3))/8, 
                                  (1.+1/np.sqrt(3))/8,(1.+1/np.sqrt(3))/8])
        pointGaussMat = np.array([
                                 [(1/np.sqrt(3)+1)/2,(1-1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
                                 [(1/np.sqrt(3)+1)/2,(1-1/np.sqrt(3))*(1-1/np.sqrt(3))/4],
                                 [(-1/np.sqrt(3)+1)/2,(1+1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
                                 [(-1/np.sqrt(3)+1)/2,(1+1/np.sqrt(3))*(1-1/np.sqrt(3))/4]
                                 ]
                                 )
    elif numGaussPoint==9:
        coeffGaussMat = np.array( [64/81*(1-0)/8, 100/324*(1-np.sqrt(3/5))/8,
                                  100/324*(1-np.sqrt(3/5))/8, 100/324*(1+np.sqrt(3/5))/8,
                                  100/324*(1+np.sqrt(3/5))/8, 40/81*(1-0)/8,
                                  40/81*(1-0)/8, 40/81*(1-np.sqrt(3/5))/8,
                                  40/81*(1+np.sqrt(3/5))/8]
                                )
        pointGaussMat = np.array([
                                 [(1+0)/2, (1-0)*(1+0)/4],
                                 [(1+np.sqrt(3/5))/2, (1-np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                 [(1+np.sqrt(3/5))/2, (1-np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                 [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                 [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                 [(1+0)/2,(1-0)*(1+np.sqrt(3/5))/4],
                                 [(1+0)/2,(1-0)*(1-np.sqrt(3/5))/4],
                                 [(1+np.sqrt(3/5))/2,(1-np.sqrt(3/5))*(1+0)/4],
                                 [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1+0)/4]
                                 ]
                                )
    else:
        raise ValueError("wrong number of Gauss point")
    return coeffGaussMat, pointGaussMat


@jit(nopython=True, cache=True)
def _GaussQuadTriLineIntegralTrialTest(cFunc, cRefGaussMat1d, pRefGaussMat1d, 
                                      endPoint1, endPoint2, vertMat, basisType, alpha, beta, md, sd, dd, ld):
    """
    Gauss quadrature of line integral, for Neumann like BC
    ref: ppt ch3 p125
    :param cFunc: coefficient function, ref ppt ch3 p4
    :param cRefGaussMat1d: 1d reference Gauss coefficient matrix
    :type cRefGaussMat1d: np.ndarray
    :param pRefGaussMat1d: 1d reference Gauss point matrix
    :type pRefGaussMat1d: np.ndarray
    :param endPoint1: the coordinates of the end points of the edge on which we are computing the line integral
                      np.array([x_cor, y_cor])
    :type endPoint1: np.ndarray
    :param endPoint2: the coordinates of the end points of the edge on which we are computing the line integral
                      np.array([x_cor, y_cor])
    :type endPoint2: np.ndarray
    :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
    :type vertMat: np.array
    :param basisType: basis type
    :type basisType: int
    :param alpha: alpha's local trial function
    :type alpha: int
    :param beta: beta's local test function
    :type beta: int
    :param md: the derivative degree of the trial FE basis function with respect to x
    :type md: int
    :param sd: the derivative degree of the trial FE basis function with respect to y
    :type sd: int
    :param dd: the derivative degree of the test FE basis function with respect to y
    :type dd: int
    :param ld: the derivative degree of the test FE basis function with respect to y
    :type ld: int
    """                                      
    gpn = cRefGaussMat1d.shape[0]  # number of the Gauss points of the Gauss quadrature we are using
    r=0.
    tol = 1e-12
    x1 = endPoint1[0]
    y1 = endPoint1[1]
    x2 = endPoint2[0]
    y2 = endPoint2[1]
    if abs(x1 - x2) < tol:  # the edge is vertical
        lowerBound = min(y1, y2)
        upperBound = max(y1, y2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = x1
            y = pLocGaussMat1d[idx]
            r += cLocGaussMat1d[idx]*cFunc(x, y)*\
                 _locBasis(basisType, x, y, vertMat, alpha, md, sd)*\
                 _locBasis(basisType, x, y, vertMat, beta, dd, ld)
    elif abs(y1 - y2) < tol:  # the edge is horizontal
        lowerBound = min(x1, x2)
        upperBound = max(x1, x2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = pLocGaussMat1d[idx]
            y = y1
            r += cLocGaussMat1d[idx]*cFunc(x, y)*\
                 _locBasis(basisType, x, y, vertMat, alpha, md, sd)*\
                 _locBasis(basisType, x, y, vertMat, beta, dd, ld)
    else:  # the slope of the edge is in (0,infinity).
        # line integral, ref: https://www.youtube.com/watch?v=_60sKaoRmhU
        lowerBound = min(x1, x2)
        upperBound = max(x1, x2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        slope = (y2-y1)/(x2-x1)
        jacobin = np.sqrt(1+slope*slope)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = pLocGaussMat1d[idx]
            y = slope * (x-endPoint1[0]) + endPoint1[1]
            r+= cLocGaussMat1d[idx]*jacobin*cFunc(x,y)*\
                _locBasis(basisType, x, y, vertMat, alpha, md, sd)*\
                 _locBasis(basisType, x, y, vertMat, beta, dd, ld)
    return r

@jit(nopython=True, cache=True)
def _GaussQuadTriLineIntegralTest(cpFunc, cRefGaussMat1d, pRefGaussMat1d, 
                                  endPoint1, endPoint2, vertMat, basisType, beta, ad, bd):
    """
    Gauss quadrature of line integral, for Neumann like BC
    ref: ppt ch3 p99
    :param cpFunc: coefficient function, ppt ch3 p107
    :type cpFunc: function
    :param cRefGaussMat1d: 1d reference Gauss coefficient matrix
    :type cRefGaussMat1d: np.ndarray
    :param pRefGaussMat1d: 1d reference Gauss point matrix
    :type pRefGaussMat1d: np.ndarray
    :param endPoint1: the coordinates of the end points of the edge on which we are computing the line integral
                      np.array([x_cor, y_cor])
    :type endPoint1: np.ndarray
    :param endPoint2: the coordinates of the end points of the edge on which we are computing the line integral
                      np.array([x_cor, y_cor])
    :type endPoint2: np.ndarray
    :param vertMat: vertice coordinate, 2*3 matrix
                vertMat[0,0] = x1  # relate to (0,1) in reference frame
                vertMat[1,0] = y1
                vertMat[0,1] = x2  # relate to (1,0) in reference frame
                vertMat[1,1] = y2
                vertMat[0,2] = x3  # relate to (0,1) in reference frame
                vertMat[1,2] = y3
    :type vertMat: np.array
    :param basisType: basis type
    :type basisType: int
    :param beta: beta's local test function
    :type beta: int
    :param ad: the derivative degree of the test FE basis function with respect to x
    :type ad: int
    :param bd: the derivative degree of the test FE basis function with respect to y
    :type bd: int
    """
    gpn = cRefGaussMat1d.shape[0]  # number of the Gauss points of the Gauss quadrature we are using
    r=0.
    tol = 1e-12
    x1 = endPoint1[0]
    y1 = endPoint1[1]
    x2 = endPoint2[0]
    y2 = endPoint2[1]
    if abs(x1 - x2) < tol:  # the edge is vertical
        lowerBound = min(y1, y2)
        upperBound = max(y1, y2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = x1
            y = pLocGaussMat1d[idx]
            r += cLocGaussMat1d[idx]*cpFunc(x, y)*\
                 _locBasis(basisType, x, y, vertMat, beta, ad, bd)
    elif abs(y1 - y2) < tol:  # the edge is horizontal
        lowerBound = min(x1, x2)
        upperBound = max(x1, x2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = pLocGaussMat1d[idx]
            y = y1
            r += cLocGaussMat1d[idx]*cpFunc(x, y)*\
                 _locBasis(basisType, x, y, vertMat, beta, ad, bd)
    else:  # the slope of the edge is in (0,infinity).
        # line integral, ref: https://www.youtube.com/watch?v=_60sKaoRmhU
        lowerBound = min(x1, x2)
        upperBound = max(x1, x2)
        cLocGaussMat1d, pLocGaussMat1d = _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound)
        slope = (y2-y1)/(x2-x1)
        jacobin = np.sqrt(1+slope*slope)
        for i in range(1, gpn+1):  # [1,gpn]
            idx = i-1
            x = pLocGaussMat1d[idx]
            y = slope * (x-endPoint1[0]) + endPoint1[1]
            r+= cLocGaussMat1d[idx]*jacobin*cpFunc(x,y)*\
                _locBasis(basisType, x, y, vertMat, beta, ad, bd)
    return r





@jit(nopython=True, cache=True)
def _genLocGauss1d(cRefGaussMat1d, pRefGaussMat1d, lowerBound, upperBound):
    """
    Generate the Gauss coefficients and Gauss points on an arbitrary interval 
    [lower_bound,upper_bound] by using affine tranformation.
    :param cRefGaussMat1d: Gauss coefficients on the reference interval [-1,1].
    :type cRefGaussMat1d: np.ndarray
    :param pRefGaussMat1d: Gauss points on the reference interval [-1,1].
    :type pRefGaussMat1d: np.ndarray
    :param lowerBound: lower bound
    :type lowerBound: double
    :param upperBound: upper bound
    :type upperBound: double
    """
    coeffGaussMat = (upperBound-lowerBound)*cRefGaussMat1d*0.5
    pointGaussMat = (upperBound-lowerBound)*pRefGaussMat1d*0.5 + (upperBound+lowerBound)*0.5
    return coeffGaussMat, pointGaussMat 




@jit(nopython=True, cache=True)
def _genRefGauss1d(numGaussPoint):
    """
    Generate the Gauss coefficients and Gauss points on the reference interval [-1,1].
    :param numGaussPoint: the number of Gauss points in the formula. The Gauss formula depends on it.
    :type numGaussPoint: int
    """
    if numGaussPoint == 2:
        coeffGaussMat = np.array([1.,1.])
        pointGaussMat = np.array([-1./np.sqrt(3), 1./np.sqrt(3)])
    elif numGaussPoint == 4:
        coeffGaussMat = np.array([0.3478548451,0.3478548451,0.6521451549,0.6521451549])
        pointGaussMat = np.array([0.8611363116,-0.8611363116,0.3399810436,-0.3399810436])
    elif numGaussPoint == 8:
        coeffGaussMat = np.array([0.1012285363,0.1012285363,0.2223810345,0.2223810345,
                                  0.3137066459,0.3137066459,0.3626837834,0.3626837834])
        pointGaussMat = np.array([0.9602898565,-0.9602898565,0.7966664774,-0.7966664774,
                                  0.5255324099,-0.5255324099,0.1834346425,-0.1834346425])
    else:
        raise ValueError("wrong number of Gauss point")
    return coeffGaussMat, pointGaussMat















