#!/usr/bin/python3
import sys
import os
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import pylhe as lhe
import lib.utilities as util
import lib.statistics_tools as st
from lib.mssm_particles import jet_hard

# Prepare plot object to visualise result
x_label=r"pT of leading jet [GeV]"
y_label=r"Density $[\mathrm{GeV}^{⁻1}]$"
title=r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
labels = ["LO", "NLO"]
plot = util.Plot(x_label, y_label, title)
plot.set_xlim(0., 20.)

# Enter paths to LHE files
path_LO = "/home/mariusss/University/master_project/data/pp_epemn1n1_LO.lhe"
path_LO_j = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO_1.lhe"
path_NLO = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO_1.lhe"
path_NLO_j = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO_1.lhe"
xsec_LO = 1.591440e-02 #pb
xsec_LO_j = 3.946654e-02 #pb
xsec_NLO = 2.187095e-02 #pb
xsec_NLO_j = 1.833450e-02 #pb

filenames = [[path_LO, path_LO_j], 
             [path_NLO, path_NLO_j]]
xsecs = [[xsec_LO, xsec_LO_j], 
         [xsec_NLO, xsec_NLO_j]]

n_files = len(filenames)

# To avoid redoing expensive calculations, prepare storage files
result_filenames = []
for f, filename in enumerate(filenames):
    file_basename, ext = os.path.splitext(filename[1])
    result_filename = file_basename + "_leading_jet_PT" + ".dat"
    result_filenames.append(result_filename)

# Create list to store histograms
histograms = []
binning = np.linspace(0., 20., 40)

# Iterate through available files
for f, filename in enumerate(filenames):

    res_file = result_filenames[f]
    xsec = xsecs[f]

    # Check if storage file exists, use that file if so
    if os.path.isfile(res_file):
        print("Reading from %s (%d/%d)" %(res_file, f+1, n_files))
        data = np.loadtxt(res_file)
    else:
        print("Reading from %s (%d/%d)" %(filename, f+1, n_files))

        # Open LHE-file and iterate through
        events, num_events = util.combine_LHE_files(filename[0], filename[1],
        xsec[0], xsec[1], pt_cut=20)
        print("Running through %d events..."%num_events)

        data = np.zeros(num_events)

        cnt = 0
        prog = 0
        print_freq = num_events//10
        percent = int(num_events/print_freq)
        for e in events:
            # Pick out the jets from the event (already pt_cut checked)
            jets = util.get_jets(e)

            # Do analysis (calculate transverse momentum of leading jet)
            jet_pTs = [p.transverse_momentum() for p in jets]

            if jet_pTs:
                data[cnt] = max(jet_pTs)

            cnt += 1
            if not cnt % progress_print_freq:
                prog += 1
                print("%d%s of events processed." %(percent*prog, "%"))

        #Save result to the storage file
        header = "pT of leading jet"
        np.savetxt(res_file, data, fmt='%e', header=header)
        print("Stored calculations in %s"%res_file)

    # Create histogram (binning is fixed)
    counts, bins = np.histogram(data, bins=binning)

    # Normalise histogram, then store it
    counts = counts/np.sum(counts*np.diff(bins))
    histograms.append([counts, bins])

    # Add histogram to plot
    plot.add_histogram(counts, bins, label=labels[f], alpha=.45)

# Calculate KL-divergence between LO and NLO distributions
LO_hist = histograms[0]  # tuple (count, bin_edges), Q-distribution representing the model
NLO_hist = histograms[1]  # tuple (count, bin_edges), P-distribution representing the data

kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)

plot.add_series(np.array([]), np.array([]), label=r"KL-div(LO$\rightarrow$NLO): %1.2e bits"%kl_div, lw=0)

# Plot data
plot.plot_all()
print("Done!")
plt.show()
