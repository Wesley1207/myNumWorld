
import numpy as np

# decimal to fractions
# https://stackoverflow.com/questions/42209365/numpy-convert-decimals-to-fractions
import fractions
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

def fd_coeff(k, x_i, x):
    """
    use n points to evaluate kth derivative of u(x_{0})
    :param k: (int) kth derivative of u(x)
    :param x_i: (np.arrray, shape=(n,)) array of [..., x_{-1}, x_{0}, x_{1}, ...]
        n should greater than or equal to  k+1
    :param x: (float) target point
    """
    n = x_i.shape[0]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ii = i+1
            A[i, j] = 1/np.math.factorial(ii-1) * (x_i[j] - x)**(ii-1)
    b = np.zeros(n)
    b[k+1-1] = 1
    return np.linalg.solve(A, b)


if __name__ == '__main__':
    x_i = np.array([-3., -2, -1, 0, 1, 2, 3])
    x = 0
    sol = fd_coeff(4, x_i, x)
    print(sol)
    
    