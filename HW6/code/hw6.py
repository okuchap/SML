import numpy as np
import scipy as sp
from numpy.random import randn, rand
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def myrand(n=5000):
    x = np.zeros(n)
    u = rand(n)
    flag = (0 <= u) * (u < 1/8)
    x[flag] = np.sqrt(8*u[flag])
    flag = (1/8 <= u) * (u < 1/4)
    x[flag] = 2 - np.sqrt(2 - 8*u[flag])
    flag = (1/4 <= u) * (u < 1/2)
    x[flag] = 1 + 4*u[flag]
    flag = (1/2 <= u) * (u < 3/4)
    x[flag] = 3 + np.sqrt(4*u[flag] - 2)
    flag = (3/4 <= u) * (u <= 1)
    x[flag] = 5 - np.sqrt(4 - 4*u[flag])
    return x


def gkernel_est(sample, x=np.linspace(0, 5, 501), bandwidth=0.1):
    pxh = np.zeros_like(x)
    n = sample.shape[0]
    for i in range(n):
        pxh = pxh + norm.pdf(x, loc=sample[i], scale=bandwidth)
    return x, pxh


def cross_validation(sample, n_split=5, params=[0.01, 0.1, 0.5]):
    n_params = len(params)
    likelihoods = np.zeros(n_params)
    group = np.split(sample, n_split)
    for j in range(n_params):
        for i in range(n_split):
            if i==0:
                sample_temp = np.hstack(group[i+1:][0])
            elif i==n_split-1:
                sample_temp = np.hstack(group[0:i])
            else:
                sample_temp = np.hstack([np.hstack(group[0:i]), group[i+1:][0]])
            _, pxh = gkernel_est(sample_temp, group[i], bandwidth=params[j])
            likelihoods[j] += np.sum(np.log(pxh))
    opt_param = params[np.argmax(likelihoods)]
    #print(likelihoods)
    return opt_param


if __name__ == '__main__':
    np.random.seed(1)
    sample = myrand()
    opt_b = cross_validation(sample=sample)
    x, pxh = gkernel_est(sample=sample, bandwidth=opt_b)

    # plot original samples
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    sns.distplot(sample, kde=False, rug=False, bins=25)
    None

    # plot estimated distribution
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(x, pxh)
    plt.show()