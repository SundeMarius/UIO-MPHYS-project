#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

import lib.utilities as util
import lib.statistics_tools as st


def test_4momentum():
    e = 23
    px = 7
    py = 3
    pz = 17
    p = util.FourMomentum(e, px, py, pz)

    p_norm = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    p_pT = np.sqrt(px**2 + py**2)
    assert p_norm == p.norm()
    assert p_pT == p.transverse_momentum()

    p.print(unit_e="GeV", unit_p="GeV")


def test_KL_div():
    mu1 = 0
    mu2 = 0
    sig1 = 1
    sig2 = 2
    N_sampl = 100_000
    n_bins = 130
    dkl_analytical = ((sig1/sig2)**2 + (mu1 - mu2)**2 / sig2**2 - 1 + np.log(sig2**2/sig1**2))/2.

    print("# samples: %d"%N_sampl)
    print("# bins: %d"% n_bins)
    print("D_KL analytic: %f\n"%dkl_analytical)

    avg_dkl = 0
    N = 5
    for i in range(N):
        p = st.sample_normal(mu1, sig1, N_sampl, n_bins, -15, 15, density=True)
        q = st.sample_normal(mu2, sig2, N_sampl, n_bins, -15, 15, density=True)
        dkl_numeric = st.KL_div(p, q)
        avg_dkl += dkl_numeric
        print(dkl_numeric)
    
    avg_dkl /= N
    print("Average D_KL: %f"%avg_dkl)


def test_plot():
    x = np.linspace(0, 2*np.pi)
    y = np.exp(-x)

    counts, bins = st.sample_normal(0, 1, 1_000_000, 180, density=True)

    while 0 in counts:
        i = list(counts).index(0.)
        counts, bins = st.combine_two_bins(counts, bins, i)
    
    #Create plot object
    xlabel = "GeV"
    ylabel = "Counts"
    title = "This is an awesome test plot"

    plot = util.Plot(xlabel, ylabel, title)

    # Add test data
    plot.add_series(x, y, label=r"$e^{-x}$")
    plot.add_histogram(counts, bins, label=r"$\mathrm{N(\mu, \sigma)}$")

    plot.plot_all()
    plot.display()


spacer = 20*'-'

test_4momentum()
print(spacer)
test_KL_div()
print(spacer)
test_plot()
print("Tests done.")
