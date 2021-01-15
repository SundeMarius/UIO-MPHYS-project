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
from lib.mssm_particles import jet_hard, charged_leptons, neutralinos

if len(sys.argv) < 2:
    print("Program requires one argument: 'run_file.csv'")
    print("Info: each line in 'run_file.csv' should have the following structure: ")
    print("'lhe-file path'")
    exit(1)

# Prepare plot object to visualise result
x_label=r"$pT_{jet} [GeV]$"
y_label=r"Density $[\mathrm{GeV^{-1}}]$"
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
    result_filename = file_basename + "_leading_jet_pt_combinedset" + ".storage"
    result_filenames.append(result_filename)

# Create list to store histograms
histograms = []

# Check if storage file exists, use that file if so
if os.path.isfile(result_filename):
    print("Reading from %s" %result_filename)
    data = np.loadtxt(result_filename)
else:
    # Open 1by1 LHE-file with *pylhe* and get the final state particles for all events.
    print("Reading from %s" %filename)

    # Open LHE-file and iterate through
    events, num_events = util.combine_LHE_files(filename[0], filename[1], xsec[0], xsec[1])
    print("Running through events (%e)..."%num_events)

    progress_print_freq = num_events*0.1

    data = np.zeros(num_events)

    cnt = 0
    for e in events:
        event_ids = [abs(int(p.id)) for p in e.particles]
        if 21 not in event_ids:
            continue
        jet_momenta = []
        invisible_momenta = []
        lepton_momenta = []
        for particle in e.particles:
            id = int(abs(particle.id))
            pT = util.FourMomentum.from_LHEparticle(particle).transverse_momentum(vector_out=True)
            if id in charged_leptons:
                lepton_momenta.append(pT)
            elif id in jet_hard:
                jet_momenta.append(pT)
            elif id in neutralinos:
                invisible_momenta.append(pT)

        pT_jet = np.linalg.norm(sum(jet_momenta))
        #pT_invisible = np.linalg.norm(sum(invisible_momenta))
        #pT_lepton = np.linalg.norm(sum(lepton_momenta))

        #jet_pT_fraction = pT_jet/(pT_jet + pT_invisible + pT_lepton)

        data.append(pT_jet)

        cnt += 1
        if not cnt % print_progress_freq:
            print("%d%s of events processed." %(cnt*100//num_events, "%"))

    #Save result to the storage file
    header = "jet scalar pT"
    np.savetxt(result_filename, data, fmt='%e', header=header)
    print("Stored calculations in %s"%result_filename)

# Create histogram
counts, bins = np.histogram(data, bins=350)

counts = counts/(np.sum(counts*np.diff(bins)))

histograms.append([counts, bins])

#Add histogram to plot
plot.add_histogram(counts, bins, label=labels[0], alpha=.5)

# Plot data
plot.plot_all()
print("Done!")
plt.show()
