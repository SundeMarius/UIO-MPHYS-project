#!/usr/bin/python3
import lib.statistics_tools as st
import lib.utilities as util
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "..")


def test_4momentum():
    e = 5
    px = 2
    py = 1
    pz = 0.4
    p = util.FourMomentum(e, px, py, pz)

    p_norm = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    p_pT = np.sqrt(px**2 + py**2)
    assert p_norm == p.norm()
    assert p_pT == p.transverse_momentum()
    a = p - util.FourMomentum(4, 3, 2, 1)
    p.print(unit_e="GeV", unit_p="GeV")
    a.print()


def test_KL_div():
    mu1 = 0
    mu2 = 4
    sig1 = 0.60
    sig2 = 0.65
    N_sampl = 1000_000
    n_bins = 150
    dkl_analytical = ((sig1/sig2)**2 + (mu1 - mu2)**2 / sig2 **
                      2 - 1 + np.log(sig2**2/sig1**2))/2.

    print("# samples: %d" % N_sampl)
    print("# bins: %d" % n_bins)
    print("D_KL analytic: %f\n" % dkl_analytical)

    avg_dkl = 0
    N = 10
    for i in range(N):
        p = st.sample_normal(mu1, sig1, N_sampl, n_bins)
        q = st.sample_normal(mu2, sig2, N_sampl, n_bins)
        dkl_numeric = st.KL_div(p, q)[0]
        avg_dkl += dkl_numeric
        print(dkl_numeric)

    avg_dkl /= N
    print("Average D_KL: %f" % avg_dkl)


def test_plot():

    counts, bins = st.sample_normal(0, 1, 50, 20)

    print(counts, bins)

    while 0 in counts:
        i = np.where(counts == 0)[0][0]
        if i == 0:
            counts, bins = counts[1:], bins[1:]
            continue
        elif i == len(counts) - 1:
            counts, bins = counts[:-1], bins[:-1]
            continue
        print(counts)
        counts, bins = st.combine_two_bins(counts, bins, i)
        print(counts)

    print(counts, bins)
    # Normalise histogram
    counts = counts/np.sum(counts * np.diff(bins))


spacer = 20*'-'

test_4momentum()
print(spacer)
test_KL_div()
print(spacer)
test_plot()
print("Tests done.")
