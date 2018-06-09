import numpy as np
from numpy.random import randn, rand
import matplotlib.pyplot as plt


def gen(n=600, alpha=0.1):
# data generation
    n1 = np.sum(rand(n, 1) < alpha)
    n2 = n - n1
    x1 = np.concatenate([randn(1, n1) + 2, 3 * randn(1, n1)])
    x2 = np.concatenate([randn(1, n2) - 2, 3 * randn(1, n2)])
    
    return x1, x2


def plot(x1, x2, line=None):
    x, y = x1
    plt.plot(x, y, 'ro', ms=3, label='class1')
    x, y = x2
    plt.plot(x, y, 'bx', ms=3, label='class2')
    
    if not (line is None):
            plt.plot(line[0], line[1], 'k-', ms=5)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    plt.show()


def fisher(x1, x2):
    x1 = x1.T
    x2 = x2.T
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    n = n1 + n2
    
    # dimension
    DIM = x1.shape[1]
    
    
    # average
    mean1 = np.mean(x1, axis=0)
    mean1 = mean1.reshape(DIM, 1)
    mean2 = np.mean(x2, axis=0)
    mean2 = mean2.reshape(DIM, 1)
    
    # covariance
    sample_cov_1 = np.zeros((DIM, DIM))
    sample_cov_2 = np.zeros((DIM, DIM))
    for x_i in x1:
        x_i = x_i.reshape(DIM, 1)
        sample_cov_1 += np.dot((x_i - mean1), (x_i - mean1).T)
        sample_cov_1 = sample_cov_1 / n1
    for x_i in x2:
        x_i = x_i.reshape(DIM, 1)
        sample_cov_2 += np.dot((x_i - mean2), (x_i - mean2).T)
        sample_cov_2 = sample_cov_2 / n2
    sample_cov = (n1/n) * sample_cov_1 + (n2/n) * sample_cov_2
    
    sample_cov_inv = np.linalg.inv(sample_cov + 0.000001 * np.eye(DIM))
    
    a = np.dot(sample_cov_inv, (mean1 - mean2))
    if (n1 - n2 > 1e-10):
        b = -0.5 * (np.dot(mean1.T, np.dot(sample_cov_inv, mean1)) - np.dot(mean2.T, np.dot(sample_cov_inv, mean2))) + np.log(n1/n2)
    else:
        b = -0.5 * (np.dot(mean1.T, np.dot(sample_cov_inv, mean1)) - np.dot(mean2.T, np.dot(sample_cov_inv, mean2)))
    b = b.reshape(1)

    return a, b



def line_est(a, b, x):
    if abs(a[1]) > 1e-10:
        c = - a[0]/a[1]
        d = - b/a[1]
    return [x, c*x+d]


if __name__ == '__main__':
    x1, x2 = gen(n=600, alpha=0.1)
    a, b = fisher(x1, x2)
    x = np.linspace(-10, 10, 3000)
    l = line_est(a, b, x)
    plot(x1, x2, l)