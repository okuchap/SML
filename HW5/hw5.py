import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def centering_sphering(X):
    '''
    X: d x n matrix
    '''
    n = X.shape[1]
    H = np.eye(n) - np.ones((n,n))/n
    XH = np.dot(X, H)
    temp = sqrtm(np.linalg.inv(np.dot(XH, XH.T)/n))
    X_tilde = np.dot(temp, XH)
    return X_tilde


def approx_newton(X, Nlim=50):
    '''
    X should be normalized.
    X: d x n matrix
    '''
    n = X.shape[1]
    b = np.array([1,0])
    threshold = 1e-08
    diff = np.inf
    n_loop = 1
    
    while n_loop < Nlim:
        #print(b)
        b_prev = b
        sum = 0
        for i in range(n):
            sum += X[:, i] * (np.dot(b, X[:, i]) ** 3)
        b = 3 * b - sum/n
        b = b / np.linalg.norm(b)
        diff = np.linalg.norm(b - b_prev)
        if (diff < threshold):
            break
        else:
            n_loop += 1
    
    if n_loop == Nlim:
        print('may not be converged')
    
    return b


def line(b, X):
    x_min = np.min(X[0])
    x_max = np.max(X[0])
    x = np.linspace(x_min, x_max, 1000)
    return [x, (b[1]/b[0])*x]


def plot(x1, line=None):
    x = x1[0]
    y = x1[1]
    plt.plot(x, y, 'ro', ms=3, label='class1')

    if not (line is None):
            plt.plot(line[0], line[1], 'k-', ms=5)
            
    #plt.xlim(np.min(x)-1, np.max(x)+1)
    #plt.ylim(np.min(y)-1, np.max(y)+1)
    
    plt.show()

# simulation
N = 1000

## original sigmal
s_gauss = np.random.randn(N)*2 + 3
s_uniform = np.random.rand(N) * 3 - 2
S = np.array([s_gauss, s_uniform])

## transformation matrix
M = np.array([[1,3],[5,1]])

## observed sigmal
X = np.dot(M, S)

X_tilde = centering_sphering(X)
b = approx_newton(X_tilde)

plot(X_tilde, line(b, X_tilde))
