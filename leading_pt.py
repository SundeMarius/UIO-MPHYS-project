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

if len(sys.argv) < 2:
    print("Program requires one argument: 'run_file.csv'")
    print("Info: each line in 'run_file.csv' should have the following structure: ")
    print("'lhe-file path', 'Particle ID 1', 'Particle ID 2', ...")
    exit(1)

# Prepare plot object to visualise result
x_label=r"pT of leading electron [GeV]"
y_label=r"Density $[\mathrm{GeV}^{⁻1}]$"
title=r"$q\bar{q}\rightarrow$ -11 1000022 11 1000022 electroweak s-channel, $\sqrt{s} = 13$TeV"
labels = ["LO", "NLO"]

plot = util.Plot(x_label, y_label, title)

# Get lhe file and PDG particle ID's from user made csv run_file, store each line in a list
run_file = sys.argv[1]
# Collect "file, particle1 ID, particle2 ID, particle3 ID, ..." into a matrix
argument_groups = np.loadtxt(run_file, delimiter=', ', dtype=str)

filenames = argument_groups[:,0]
particle_ids = list(argument_groups[:,1:].astype(int))

n_files = len(filenames)

# To avoid redoing expensive calculations, prepare storage files
result_filenames = []
for f, filename in enumerate(filenames):
    file_basename, ext = os.path.splitext(filename)
    result_filename = file_basename + "_leading_electron_pT" + ".storage"
    result_filenames.append(result_filename)

# Create list to store histograms
histograms = []

# Iterate through available files
for f, filename in enumerate(filenames):

    res_file = result_filenames[f]

    # Check if storage file exists, use that file if so
    if os.path.isfile(res_file):
        print("Reading from %s (%d/%d)" %(res_file, f+1, n_files))
        data = np.loadtxt(res_file)
    else:
        # Open 1by1 LHE-file with *pylhe* and get the final state particles for all events.
        print("Reading from %s (%d/%d)" %(filename, f+1, n_files))

        # Open LHE-file and iterate through
        events = lhe.readLHE(filename)
        num_events = lhe.readNumEvents(filename)

        print_progress_freq = int(num_events*0.1)

        data = np.zeros(num_events)

        cnt = 0
        for e in events:
            # Create dictionary of particles for easy access ## TIP: this can be made into a function
            event_particles = { id : [] for id in [p.id for p in e.particles] }
            for p in e.particles:
                event_particles[p.id].append(p)

            # Extract particles of interest (as FourMomentum objects)
            event_momenta = []
            for id in particle_ids[f]:
                momentum = util.FourMomentum.from_LHEparticles(event_particles[id])
                event_momenta.append(momentum[0])

            # Do analysis (calculate transverse momentum of leading particle)
            particle_pTs = [p.transverse_momentum() for p in event_momenta]

            data[cnt] = max(particle_pTs)

            cnt += 1

            if not cnt % print_progress_freq:
                print("%d%s of events processed." %(cnt*100//num_events, "%"))

        #Save result to the storage file
        header = "pT of leading electron"
        np.savetxt(res_file, data, fmt='%e', header=header)
        print("Stored calculations in %s"%res_file)

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
