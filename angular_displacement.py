#!/usr/bin/python3
"""
Assuming the lhe file has been uncompressed after completing the event generation
"""
import numpy as np
import matplotlib.pyplot as plt
import pylhe as lhe
import sys
import os
import lib.utilities as util
import lib.statistics_tools as st

# Prepare plot object to visualise result
x_label=r"Angle between 1000022 and 1000022 [deg]"
y_label=r"Density $[\mathrm{deg}^{⁻1}]$"
title=r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
labels = ["LO", "NLO"]
plot = util.Plot(x_label, y_label, title)

# Enter paths to LHE files
path_LO_1 = "/home/mariusss/University/master_project/data/pp_epemn1n1_LO.lhe"
path_LO_2 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO.lhe"
path_NLO_1 = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO.lhe"
path_NLO_2 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO.lhe"
xsec_LO_1 = 1.591440e-02 #pb
xsec_LO_2 = 9.746654e-03 #pb
xsec_NLO_1 = 2.140095e-02 #pb
xsec_NLO_2 = 1.833450e-02 #pb

filenames = [[path_LO_1, path_LO_2], [path_NLO_1, path_NLO_2]]
xsecs = [[xsec_LO_1, xsec_LO_2], [xsec_NLO_1, xsec_NLO_2]]

n_files = len(filenames)

# To avoid redoing expensive calculations, prepare storage files
result_filenames = []
for f, filename in enumerate(filenames):
    file_basename, ext = os.path.splitext(filename[0])
    result_filename = file_basename + "_angular_diff_1000022_combinedset" + ".storage"
    result_filenames.append(result_filename)

# Create list to store histograms
histograms = []

# Iterate through available files
for f, filename in enumerate(filenames):

    res_file = result_filenames[f]
    xsec = xsecs[f]

    # Check if storage file exists, use that file if so
    if os.path.isfile(res_file):
        print("Reading from %s (%d/%d)" %(res_file, f+1, n_files))
        data = np.loadtxt(res_file)
    else:
        # Open 1by1 LHE-file with *pylhe* and get the final state particles for all events.
        print("Reading from %s (%d/%d)" %(filename, f+1, n_files))

        # Open LHE-file and iterate through
        events, num_events = util.combine_LHE_files(filename[0], filename[1], xsec[0], xsec[1], pt_cut=20)
        print("Running through events (%e)..."%num_events)

        progress_print_freq = num_events*0.1

        data = np.zeros(num_events)

        cnt = 0
        for e in events:
            # Create dictionary of final state particles for easy access
            fs_particles = util.get_final_state_particles(e)

            # Extract particles of interest (as FourMomentum objects)
            p1, p2 = util.FourMomentum.from_LHEparticles(fs_particles[1000022])

            three_p1 = p1.three_momentum()
            three_p2 = p2.three_momentum()

            cs = np.dot(three_p1, three_p2)/(np.linalg.norm(three_p1)*np.linalg.norm(three_p2))

            data[cnt] = np.degrees(np.arccos(cs))

            cnt += 1
            if not cnt % progress_print_freq:
                print("%d%s of events processed." %(cnt*100//num_events, "%"))

        #Save result to the storage file
        header = "Angle between 1000022 and 1000022"
        np.savetxt(res_file, data, fmt='%e', header=header)
        print("Stored calculations in %s"%res_file)
        print(util.sep)

    # Create histogram
    counts, bins = np.histogram(data, bins=450)
    if f == 0:
        first_bins = bins

    # Normalise histogram, then store it (using the binning from first iteration to ensure identical binning)
    counts = counts/np.sum(counts*np.diff(first_bins))
    histograms.append([counts, first_bins])

    #Add histogram to plot
    plot.add_histogram(counts, first_bins, label=labels[f], alpha=.5)

# Calculate KL-divergence between LO and NLO distributions
LO_hist = histograms[0]  # tuple (count, bin_edges), Q-distribution representing the model
NLO_hist = histograms[1]  # tuple (count, bin_edges), P-distribution representing the data

kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)

plot.add_series(np.array([]), np.array([]), label=r"KL-div(LO$\rightarrow$NLO): %1.2e bits"%kl_div, lw=0)

# Plot data
plot.plot_all()
print("Done!")
plt.show()