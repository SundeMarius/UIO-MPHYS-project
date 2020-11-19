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
x_label=r"Maximal $p_T$ among { -11 1000022 11 1000022 } [GeV]"
y_label="Events"
title=r"$q\bar{q}\rightarrow$ -11 1000022 11 1000022 electroweak s-channel, $\sqrt{s} = 13$TeV"
labels = ["LO", "NLO"]

plot = util.Plot(x_label, y_label, title) 

# Get lhe file and particle ID's from user made run_file, store each row in a matrix
run_file = sys.argv[1]
argument_groups = np.loadtxt(run_file, delimiter=', ', dtype=str) # collect "file, particle1 ID, particle2 ID, particle3 ID, ..." into a matrix

filenames = argument_groups[:,0]
particle_ids = argument_groups[:,1:].astype(int)

n_files = len(filenames) 

# Get number of final state particles from each event file
numb_final_particles = [len(group)-1 for group in argument_groups]

# To avoid redoing expensive calculations, prepare storage files
result_filenames = []
for filename in filenames:
    file_basename, ext = os.path.splitext(filename)
    result_filename = file_basename + "_pTs" + ".storage"
    result_filenames.append(result_filename)

# Create list to store histograms
histograms = []

# Iterate through available files
for f, filename in enumerate(filenames):

    res_file = result_filenames[f]

    n_particles = numb_final_particles[f]
    
    # Check if storage file exists, use that file if so
    if os.path.isfile(res_file):
        print("Reading from %s (%d/%d)" %(res_file, f+1, n_files))
        pTs = np.loadtxt(res_file)    
    else:
        # Open 1by1 LHE-file with *pylhe* and get the chosen final state particles for all events.
        print("Reading from %s (%d/%d)" %(filename, f+1, n_files))

        events = util.get_final_state_events(filename, particle_ids[f])
        num_events = len(events)

        print_progress_freq = int(num_events*0.1)

        pTs = np.zeros(num_events)

        #Loop over events
        for i in range(num_events):
            # Get transverse momenta pT of the chosen final state particles
            event_pts = []
            for p in events[i]:
                for part in p:
                    event_pts.append(util.FourMomentum.from_LHEparticle(part).transverse_momentum())
            #Get the maximal pT from event, store in array
            pTs[i] = np.max(event_pts)
            if not (i+1)%print_progress_freq and i != 0:
                print("%d%s of events processed." %(i*100//num_events + 1, "%"))

        #Save result to the storage file
        np.savetxt(res_file, pTs, fmt='%e', header="Transverse momentum pT of the particle with the highest at every event")
        print("Stored calculations in %s"%res_file)

    # Create normalized histogram
    counts, bins = np.histogram(pTs, bins=100, density=True)
    histograms.append([counts, bins])
    
    # Add histogram to plot
    plot.add_histogram(counts, bins, label=labels[f], alpha=.5)

# Calculate KL-divergence between LO and NLO distributions
LO_hist = histograms[0]  # tuple (count, bin_edges)
NLO_hist = histograms[1]  # tuple (count, bin_edges)

kl_div = st.KL_div(NLO_hist, LO_hist, base_two=True)

plot.add_series(np.array([]), np.array([]), label=r"KL(LO$\rightarrow$NLO) $= %1.2f$ bits"%kl_div, lw=0)

# Plot data
plot.plot_all()
plt.show()
