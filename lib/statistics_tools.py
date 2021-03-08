import numpy as np
from scipy.special import rel_entr


# HISTOGRAM TOOLS
def combine_two_bins(count, bin_edges, i):
    """
    Combine the i'th bin with the smallest neighbor

    :param count: np.array of histogram counts
    :param bin_edges:  np.array of histogram bin right edges
    :param i: index of the bin of interest (equivalent with index of count)

    :return : A new histogram where bin "i" has been resized.
    """

    # Make copies of count and bins, turn them to lists for convenience
    new_count = count[:]
    new_bins = bin_edges[:]
    N = len(new_count)

    # Check if user has picked the first or last bin
    if not (0 < i < N-1):
        return count, bin_edges

    bin_widths = np.diff(new_bins)

    bin_left = new_count[i-1]
    width_left = abs(bin_widths[i-1])

    bin_right = new_count[i+1]
    width_right = abs(bin_widths[i+1])

    width = abs(bin_widths[i])
    # Combine bin i with the smallest of the neighbor bins
    # Then scale it with the average count value between those two.
    if bin_right >= bin_left:
        new_bins = np.delete(new_bins, i)
        new_count[i] = (bin_left*width_left + new_count[i]*width)/(width + width_left)
        new_count = np.delete(new_count, i-1)
    else:
        new_bins = np.delete(new_bins, i+1)
        new_count[i] = (bin_right*width_right + new_count[i]*width)/(width + width_right)
        new_count = np.delete(new_count, i+1)

    return new_count, new_bins


# STATISTICAL DISCRIMINANTS
# Kullback-Leibler divergence
def KL_div(P, Q, base_two=False):
    """
    :param P: numpy.histogram object
    :param Q: numpy.histogram object
    :param base_two: Set True if you want rel.entr in #bits (default: False gives #nats)
    :return : The Kullback-Leibler divergence.
    "Avg. relative entropy" between the two input distributions
    (divergence/information gain going from Q to P).
    """
    P_count, P_bins = P
    Q_count, Q_bins = Q

    # Check if Q-histogram contains empty bins. Resize bins in P and Q if so.
    while 0 in Q_count:

        i = np.where(Q_count == 0)[0][0]

        if i == 0:
            Q_count, Q_bins = Q_count[1:], Q_bins[1:]
            P_count, P_bins = P_count[1:], P_bins[1:]
            continue
        if i == len(Q_count)-1:
            Q_count, Q_bins = Q_count[:-1], Q_bins[:-1]
            P_count, P_bins = P_count[:-1], P_bins[:-1]
            continue

        Q_count, Q_bins = combine_two_bins(Q_count, Q_bins, i)
        P_count, P_bins = combine_two_bins(P_count, P_bins, i)

    # Check P-histogram as well
    while 0 in P_count:

        i = np.where(P_count == 0)[0][0]

        if i == 0:
            P_count, P_bins = P_count[1:], P_bins[1:]
            Q_count, Q_bins = Q_count[1:], Q_bins[1:]
            continue
        if i == len(P_count)-1:
            P_count, P_bins = P_count[:-1], P_bins[:-1]
            Q_count, Q_bins = Q_count[:-1], Q_bins[:-1]
            continue

        P_count, P_bins = combine_two_bins(P_count, P_bins, i)
        Q_count, Q_bins = combine_two_bins(Q_count, Q_bins, i)

    # Normalise histograms
    P_count = P_count/np.sum(P_count * np.diff(P_bins))
    Q_count = Q_count/np.sum(Q_count * np.diff(Q_bins))

    # Calculate KL-divergence
    rel_entropy_array = rel_entr(P_count, Q_count)
    data = rel_entropy_array * np.diff(P_bins)
    kl_d = np.sum(data)
    kl_d_cumulative = np.cumsum(data)

    if base_two:
        output = kl_d/np.log(2.), kl_d_cumulative/np.log(2), P_bins
    else:
        output = kl_d, kl_d_cumulative, P_bins

    return output


def KL_div_legend_title(kl_divergence, unit='nats'):

    return r'$D_{KL}$$\left(\mathrm{LO+NLO}\mid'\
           r'\mathrm{LO}\right)$: %1.2e ' % kl_divergence + unit


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


# PLOT TOOLS
def calculate_relative_difference(series1, series2):

    # Get stddev of bin counts (assuming Poisson process)
    ones = np.ones(series1.shape[0])
    std1 = np.maximum(np.sqrt(series1), ones)
    std2 = np.maximum(np.sqrt(series2), ones)

    # Normalised binned sets (relative frequencies + relative uncert.)
    c1 = np.sum(series1)
    c2 = np.sum(series2)

    x1 = series1 / c1
    dx1 = std1 / c1

    x2 = series2 / c2
    dx2 = std2 / c2

    # Calculate rel. difference and the uncertainty in the bin counts
    rel_diff = x2/x1 - 1.

    # From error propagating "rel_diff" wrt. x1 and x2 (uncert. dx1 and dx2 resp.):
    rel_diff_uncertainty = dx1/x1 * np.sqrt((x2/x1)**2 + (dx2/dx1)**2)

    return rel_diff, rel_diff_uncertainty
