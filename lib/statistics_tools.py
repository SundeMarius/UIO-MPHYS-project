import numpy as np
from scipy.special import rel_entr

# HISTOGRAM TOOLS
def combine_two_bins(count, bin_edges, i):
    """
    Combine the i'th bin with the smallest neighbor

    :param count: np.array of histogram counts
    :param bin_edges:  np.array of histogram bin right edges
    :param i: index of the bin of interest.

    :return : A new histogram where bin "i" has been resized.
    """
    
    #Make copies of count and bins, turn them to lists for convenience
    new_count = list(count[:])
    new_bins = list(bin_edges[:])
    N = len(new_count)

    bin_left = new_count[i-1]
    width_left = new_bins[i-1] - new_bins[i-2]

    bin_right = new_count[(i+1)%N]
    width_right = new_bins[(i+1)%N] - new_bins[i]

    width = new_bins[i] - new_bins[i-1]
    # Combine bin i with the smallest of the neighbor bins (then scale it with the average count value between those two).
    if bin_right >= bin_left or bin_right == 0:
        new_bins.pop(i-1)
        new_count[i-1] *= width_left/(width + width_left)
    else:
        new_bins.pop((i+1)%N)
        new_count[(i+1)%N] *= width_right/(width + width_right)
    
    # Update the new count list and return
    new_count.pop(i%N)

    return np.array(new_count), np.array(new_bins)


# STATISTICAL DISCRIMINANTS
# Kullback-Leibler divergence
def KL_div(P, Q, base_two=False):
    """
    :param P: numpy.histogram object
    :param Q: numpy.histogram object
	:param base_two: Set True if you want rel.entr in #bits (default: False gives #nats)
    
    :return : The Kullback-Leibler divergence (avg. relative entropy) between the two input distributions 
    (divergence/information gain going from Q to P)
    """
    P_count, P_bins = P
    Q_count, Q_bins = Q

    # Check if Q-histogram contains empty bins. Resize bins in P and Q if so.
    while 0 in Q_count:
        i = list(Q_count).index(0)
        Q_count, Q_bins = combine_two_bins(Q_count, Q_bins, i)
        
        #Do the same change in P-histogram
        P_count, P_bins = combine_two_bins(P_count, P_bins, i)
    
    # Check P-histogram as well     
    while 0 in P_count:
        i = list(P_count).index(0)
        P_count, P_bins = combine_two_bins(P_count, P_bins, i)
        
        #Do the same change in Q-histogram
        Q_count, Q_bins = combine_two_bins(Q_count, Q_bins, i)

    #Normalise the inputs to be safe
    P_sum = np.sum(P_count * np.diff(P_bins))
    Q_sum = np.sum(Q_count * np.diff(Q_bins))

    Q_count = Q_count/Q_sum
    P_count = P_count/P_sum

    #Calculate KL-divergence and return the value
    rel_entropy_array = rel_entr(P_count, Q_count)
    kl_d = np.sum(rel_entropy_array * np.diff(P_bins)) 
    
    if base_two:
        return kl_d/np.log(2.)
    else:
        return kl_d

# Earth movers distance
def emd(Q, P):
    pass


# SAMPLING TOOLS
def sample_normal(mu, sigma, n_samples, n_bins, x_min=0, x_max=0, density=False):
    """
    :param mu: mean of gaussian (scalar or array in mul.dim)
    :param sigma: stddev of gaussian (scalar or array in mul.dim)
    :param n_samples: number of samples
    :param n_bins: number of bins
	:param x_min: the lower limit of the sample range
	:param x_max: the upper limit of the sample range
	:param density: to normalize the sample output or not (default: False)

    :return: count_array, bins (right edge values) 
    """
    s = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    if x_min == 0 and x_max == 0:
        return np.histogram(s, bins=n_bins, density=density)
    else:
        return np.histogram(s, bins=n_bins, density=density, range=(x_min, x_max))
