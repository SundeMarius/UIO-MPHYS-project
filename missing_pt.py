#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os
import lib.utilities as util
import lib.statistics_tools as st
from lib.mssm_particles import invisible_particles

# Prepare plot object to visualise result
x_label = r"$p_T^{miss}$ [GeV]"
y_label = r"Density $[\mathrm{GeV}^{⁻1}]$"
title = r"$pp\rightarrow$ -11 1000022 11 1000022 j electroweak, $\sqrt{s} = 13$TeV"
plot = util.Plot(x_label, y_label, title)

labels_pt10 = [r"LO, min jet $p_T=10$GeV", r"NLO, min jet $p_T=10$GeV"]
labels_pt1 = [r"LO, min jet $p_T=1$GeV", r"NLO, min jet $p_T=1$GeV"]

labels = [labels_pt10, labels_pt1]

# Enter paths to LHE files
path_LO_1 = "/home/mariusss/University/master_project/data/pp_epemn1n1_LO.lhe"

path_LO_2_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO_10.lhe"
path_NLO_1_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO_10.lhe"
path_NLO_2_pt10 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO_10.lhe"

path_LO_2_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_LO_1.lhe"
path_NLO_1_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1_NLO_1.lhe"
path_NLO_2_pt1 = "/home/mariusss/University/master_project/data/pp_epemn1n1j_NLO_1.lhe"

# Cross sections for each of the LHE files
xsec_LO_1 = 1.591e-02  # pb

xsec_LO_2_pt10 = 1.507e-02  # pb
xsec_NLO_1_pt10 = 2.137e-02  # pb
xsec_NLO_2_pt10 = 1.840e-02  # pb

xsec_LO_2_pt1 = 3.959e-02  # pb
xsec_NLO_1_pt1 = 2.138e-02  # pb
xsec_NLO_2_pt1 = 1.844e-02  # pb

filenames_pt10 = [[path_LO_1, path_LO_2_pt10],
                  [path_NLO_1_pt10, path_NLO_2_pt10]]
filenames_pt1 = [[path_LO_1, path_LO_2_pt1], [path_NLO_1_pt1, path_NLO_2_pt1]]

xsecs_pt10 = [[xsec_LO_1, xsec_LO_2_pt10], [xsec_NLO_1_pt10, xsec_NLO_2_pt10]]
xsecs_pt1 = [[xsec_LO_1, xsec_LO_2_pt1], [xsec_NLO_1_pt1, xsec_NLO_2_pt1]]

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

# Create list to store histograms
histograms = []

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
            # Open 1by1 LHE-file with *pylhe* and get the final state particles
            # for all events.
            print("Reading from %s (%d/%d)" % (filename, f+1, n_files))

            # Open LHE-file and iterate through
            events, num_events = util.combine_LHE_files(
                filename[0], filename[1], xsec[f][0], xsec[f][1], pt_cut=20)
            print("Running through events (%e)..." % num_events)

            progress_print_freq = num_events//10

            data = np.zeros(num_events)

            cnt = 0
            prog = 0
            for e in events:
                # Create dictionary of final state particles for easy access
                fs_particles = util.get_final_state_particles(e)

                # Extract momenta of visible particles (as FourMomentum objects)
                event_ids = fs_particles.keys()
                pT_tot = 0
                for id in event_ids:
                    if id in invisible_particles:
                        momentum = util.FourMomentum.from_LHEparticles(
                            fs_particles[id])
                        # Do analysis
                        # (calculate missing transverse energy)
                        for particle in momentum:
                            pT_tot += particle.transverse_momentum(
                                vector_out=True)

                missing_pt = np.linalg.norm(pT_tot)
                data[cnt] = missing_pt

                cnt += 1
                if not cnt % progress_print_freq:
                    prog += 1
                    print("%d%s of events processed." % (10*prog, "%"))

            # Save result to the storage file
            np.savetxt(res_file, data, fmt='%e',
                       header="Missing transverse momentum (scalar) (MET)")
            print("Stored calculations in %s" % res_file)
            print(util.sep)

        # Create histogram
        counts, bins = np.histogram(data, bins=450)
        if f == 0:
            first_bins = bins

        # Normalise histogram, then store it
        # (using the binning from first iteration to ensure identical binning)
        counts = counts/np.sum(counts*np.diff(first_bins))
        histograms.append([counts, first_bins])

        # Add histogram to plot
        plot.add_histogram(counts, first_bins, label=lab[f], alpha=0.40)

# Calculate KL-divergence between LO and NLO distributions
# tuple (count, bin_edges), Q-distribution representing the model
LO_hist_pt10 = histograms[0]
# tuple (count, bin_edges), P-distribution representing the data
NLO_hist_pt10 = histograms[1]

# tuple (count, bin_edges), Q-distribution representing the model
LO_hist_pt1 = histograms[2]
# tuple (count, bin_edges), P-distribution representing the data
NLO_hist_pt1 = histograms[3]

kl_div_pt10 = st.KL_div(NLO_hist_pt10, LO_hist_pt10, base_two=True)
kl_div_pt1 = st.KL_div(NLO_hist_pt1, LO_hist_pt1, base_two=True)

plot.add_series(np.array([]), np.array(
    []), label=r"KL-div(LO$\rightarrow$NLO, $p_T=10$GeV): %1.2e bits" % kl_div_pt10, lw=0)
plot.add_series(np.array([]), np.array(
    []), label=r"KL-div(LO$\rightarrow$NLO, $p_T=1$GeV): %1.2e bits" % kl_div_pt1, lw=0)

# Plot data
plot.plot_all()
print("Done!")
plt.show()
