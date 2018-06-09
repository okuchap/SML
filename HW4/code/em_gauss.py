import numpy as np
import scipy as sp
from numpy.random import randn, rand
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def em_gauss_sim(n_sample=1000, n_component=5, iter_lim=20, seed=7):
    # set seed
    np.random.seed(seed)

    x = randn(n_sample) + (rand(n_sample) > 0.3) * 4 - 2

    L = - np.inf

    # initial values
    weights = np.ones(n_component) / n_component
    means = np.linspace(np.min(x), np.max(x), n_component)
    covs = np.ones(n_component)/10
    eta = np.zeros([n_sample, n_component])

    # main loop
    n_iter = 0
    while n_iter < iter_lim:
        # E step
        for i in range(n_sample):
            for l in range(n_component):
                eta[i][l] = (weights[l] * norm.pdf(x=x[i], loc=means[l], scale=np.sqrt(covs[l]))) / np.dot(weights, norm.pdf(x=x[i], loc=means, scale=np.sqrt(covs)))

        # M step
        means_prev = means.copy()
        weights_prev = weights.copy()
        covs_prev = covs.copy()
        for l in range(n_component):
            weights[l] = np.sum(eta[:,l]) / n_sample
            covs[l] = np.dot(eta[:,l], (x - means_prev[l])**2) / (1 * np.sum(eta[:,l]))
            means[l] = np.dot(eta[:,l], x) / np.sum(eta[:, l])

        # stop condition
        if np.sum((weights - weights_prev)**2) < 0.00001:
            break

        n_iter += 1

    # for debugging
    print(n_iter)

    sim_x = np.linspace(np.min(x), np.max(x), 1000)
    y = np.zeros_like(sim_x)
    for l in range(n_component):
        tmp = weights[l] * norm.pdf(x=sim_x, loc=means[l], scale=np.sqrt(covs[l]))
        y += tmp

    return x, sim_x, y

