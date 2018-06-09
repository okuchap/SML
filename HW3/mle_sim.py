import numpy as np
import scipy as sp
from numpy.random import binomial
import matplotlib.pyplot as plt
import seaborn as sns

def mle_sim(N_MLE=100, N_BER=100, theta_true=0.3):
    mle_list = binomial(n=N_BER, p=theta_true, size = N_MLE)/N_BER

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('N_MLE$ = {0}$, N_BER$ = {1}$'.format(N_MLE, N_BER))
    ax.set_ylabel('freq')
    sns.distplot(mle_list, kde=False, rug=False, bins=25, axlabel="MLE")
    None