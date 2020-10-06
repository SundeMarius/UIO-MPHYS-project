import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.special as sps

def KL_div(P, Q, base_two=False):
    """
    :param P: normalised numpy.histogram object (bins sum to 1)
    :param Q: normalised numpy.histogram object (bins sum to 1)
    :param base_two: Set True if you want rel.entr in #bits (default is #nats)
    
    :return : The Kullback-Leibler divergence (avg. relative entropy) between the two input distributions 
    """
    #Check if inputs are normalised
    P_hist, P_bins = P
    Q_hist, Q_bins = Q
    P_sum = np.sum(P_hist * np.diff(P_bins))
    Q_sum = np.sum(Q_hist * np.diff(Q_bins))
    if not np.isclose(P_sum, 1.):
        raise Exception("Error: P histogram is not normalised (sum=%1.4f)"%P_sum)
    if not np.isclose(Q_sum, 1.):
        raise Exception("Error: Q histogram is not normalised (sum=%1.4f)"%Q_sum)
    
    #Calculate KL-divergence and return the value
    rel_entropy_array = sps.rel_entr(P_hist, Q_hist)

    d_kl = np.sum(rel_entropy_array) 
    if base_two:
        return d_kl/np.log(2.)
    else:
        return d_kl

def sample_normal(mu, sigma, n_samples, n_bins=15, density=False):
    """
    :param mu: mean of gaussian (scalar or array in mul.dim)
    :param sigma: stddev of gaussian (scalar or array in mul.dim)
    :return: count_array, bins (edge values) 
    """
    s = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    count, bins = np.histogram(s, bins=n_bins, density=density)
    return count, bins

