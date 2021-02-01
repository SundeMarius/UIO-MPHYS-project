#!/usr/bin/python3
import sys
sys.path.insert(0, "/home/mariusss/University/master_project/scripts")

import numpy as np
import matplotlib.pyplot as plt
import os
import lib.utilities as util
import lib.statistics_tools as st
from lib.mssm_particles import invisible_particles

# Prepare plot object to visualise result
x_label = r"$p_T^{miss}$ [GeV]"
y_label = r"Events"
title = r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
plot = util.Plot(x_label, y_label, title)

labels_pt10 = [r"LO, min jet $p_T=10$GeV",
               r"NLO, min jet $p_T=10$GeV"]
labels_pt1 = [r"LO, min jet $p_T=1$GeV",
              r"NLO, min jet $p_T=1$GeV"]

labels = [labels_pt10, labels_pt1]

# Enter paths to LHE files
path_LO = "/home/mariusss/University/master_project/data/pp_epemn1n1_LO.lhe"

path_LO_j_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO_10.lhe"

path_NLO_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO_10.lhe"
path_NLO_j_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO_10.lhe"

path_LO_j_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO_1.lhe"

path_NLO_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO_1.lhe"
path_NLO_j_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO_1.lhe"

# Cross sections for each of the LHE files
xsec_LO = 1.591e-02  # pb

xsec_LO_j_pt10 = 1.507e-02  # pb

xsec_NLO_pt10 = 2.137e-02  # pb
xsec_NLO_j_pt10 = 1.840e-02  # pb

xsec_LO_j_pt1 = 3.959e-02  # pb

xsec_NLO_pt1 = 2.138e-02  # pb
xsec_NLO_j_pt1 = 1.844e-02  # pb

filenames_pt10 = [[path_LO, path_LO_j_pt10],
                  [path_NLO_pt10, path_NLO_j_pt10]]
filenames_pt1 = [[path_LO, path_LO_j_pt1],
                 [path_NLO_pt1, path_NLO_j_pt1]]

xsecs_pt10 = [[xsec_LO, xsec_LO_j_pt10],
              [xsec_NLO_pt10, xsec_NLO_j_pt10]]
xsecs_pt1 = [[xsec_LO, xsec_LO_j_pt1],
             [xsec_NLO_pt1, xsec_NLO_j_pt1]]

filenames = [filenames_pt10, filenames_pt1]
xsecs = [xsecs_pt10, xsecs_pt1]

n_files = len(filenames)

# To avoid redoing expensive calculations, prepare storage files
result_filenames = []
for p, pt in enumerate(filenames):
    result_files = []
    # Iterate through available files
    for f, filename in enumerate(pt):
        file_basename, ext = os.path.splitext(filename[1])
        result_filename = file_basename + "_missingPT_combdata" + ".dat"
        result_files.append(result_filename)

    result_filenames.append(result_files)


# Cut/filter parameters
pt_cut = 20.
pt_miss_low = 300.
pt_miss_high = 1000.

plot.set_xlim(pt_miss_low, pt_miss_high)


# Create list to store histograms, and finxed binning
histograms = []
binning = np.linspace(0., 1.e3, 101)


# Wrap all relevant cuts into one function
def pass_cuts(event):
    cut_1 = util.check_jet_pt(event, pt_cut)
    return cut_1 


for p, pt in enumerate(filenames):
    xsec = xsecs[p]
    lab = labels[p]
    # Iterate through available files
    for f, filename in enumerate(pt):
        res_file = result_filenames[p][f]
        # Check if storage file exists, use that file if so
        if os.path.isfile(res_file):
            print("Reading from %s (%d/%d)" % (res_file, f+1, n_files))
            data = np.loadtxt(res_file)
        else:
            print("Reading from %s (%d/%d)" % (filename, f+1, n_files))

            # Open LHE-file and iterate through
            events, num_events = util.combine_LHE_files(
                filename[0], filename[1], xsec[f][0], xsec[f][1], pt_cut)
            print("Running analysis and cuts through %d events..." % num_events)

            data = []

            cnt = 0
            prog = 0
            print_freq = num_events//10
            percent = int(num_events/print_freq)
            for e in events:
                # Print status
                cnt += 1
                if not cnt % print_freq:
                    prog += 1
                    print("%d%s" % (prog*percent, "%"))

                # Apply general cuts/filters
                if not pass_cuts(e): 
                    continue

                # Get list of final state particles for easy access
                fs_particles = util.get_final_state_particles(e)
                pT = np.array([0., 0.])
                for particle in fs_particles:
                    if particle.id in invisible_particles:
                        invisible_momentum = util.FourMomentum.from_LHEparticle(
                            particle)

                        pT += invisible_momentum.transverse_momentum(vector_out=True)

                # Calculate missing transverse energy
                missing_pt = np.linalg.norm(pT)
                if pt_miss_low < missing_pt < pt_miss_high:
                    data.append(missing_pt)

            # Save result to the storage file
            np.savetxt(res_file, data, fmt='%e',
                       header="Missing transverse momentum (MET)")
            print("Stored calculations in %s" % res_file)
            print(util.sep)

        # Create histogram (use fixed manual set binning)
        counts, bins = np.histogram(data, bins=binning)

        histograms.append([counts, bins])

        # Add histogram to plot
        plot.add_histogram(counts, bins, label=lab[f], alpha=1.)

# Calculate KL-divergence between LO and NLO distributions
# Q-distribution representing the model
LO_hist_pt10 = histograms[0]
# P-distribution representing the data
NLO_hist_pt10 = histograms[1]
# Q-distribution representing the model
LO_hist_pt1 = histograms[2]
# P-distribution representing the data
NLO_hist_pt1 = histograms[3]

kl_div_pt10 = st.KL_div(NLO_hist_pt10, LO_hist_pt10, base_two=True)
kl_div_pt1 = st.KL_div(NLO_hist_pt1, LO_hist_pt1, base_two=True)

plot.add_label(r"KL-div(LO$\rightarrow$NLO, $p_T=10$GeV): %1.2e bits"%kl_div_pt10)
plot.add_label(r"KL-div(LO$\rightarrow$NLO, $p_T=1$GeV): %1.2e bits" % kl_div_pt1)

# Plot data
plot.plot_all()
