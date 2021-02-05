#!/usr/bin/python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lib.utilities as util
from lib.mssm_particles import invisible_particles, charged_leptons, jet_hard


# Enter paths to LHE data directories 
path_LO = "/home/mariusss/University/master_project/data/LO"
path_LO_NLO = "/home/mariusss/University/master_project/data/LO+NLO"

# Enter path to desired destination of the processed output files (.dat)
path_to_output = "/home/mariusss/University/master_project/data"

# Containers for LHE files
LO_files = []
LO_NLO_files = []

# Fill up containers with all the available files in the LHE data directories (recursively)
for (dirpath, dirnames, filenames) in os.walk(path_LO):
    LO_files += [os.path.join(dirpath, file) for file in filenames if '.lhe' in file]
for (dirpath, dirnames, filenames) in os.walk(path_LO_NLO):
    LO_NLO_files += [os.path.join(dirpath, file) for file in filenames if '.lhe' in file]

# Separate files with/without explicit jet
# NOTE: This works as long as each process has its own directory (for instance "dislepton/" & "dislepton_j/")
half_way_LO = len(LO_files)//2
half_way_LO_NLO = len(LO_NLO_files)//2
LO_1, LO_2 = LO_files[:half_way_LO], LO_files[half_way_LO:]
LO_NLO_1, LO_NLO_2 = LO_NLO_files[:half_way_LO_NLO], LO_NLO_files[half_way_LO_NLO:]

# Create filenames to store output
output_filename_LO = "LO_output.dat"
output_filename_LO_NLO = "LO+NLO_output.dat"
output_filename_LO = os.path.join(path_to_output, output_filename_LO)
output_filename_LO_NLO = os.path.join(path_to_output, output_filename_LO_NLO)


# Cut/filter parameters
pt_cut = 20.

# Wrap all relevant cuts into one function
def pass_cuts(event):
    cut_1 = util.check_jet_pt(event, pt_cut)
    return cut_1 

# Iterate through available files
datasets_at_order = [LO_files, 
                     LO_NLO_files]
datasets_jet_split = [[LO_1, LO_2], 
                      [LO_NLO_1, LO_NLO_2]]
outputs = [output_filename_LO, 
           output_filename_LO_NLO]


for f, files in enumerate(datasets_at_order):
    
    output = outputs[f]
    dataset = datasets_jet_split[f]
    
    data = {'missing_pt_GeV': [], 'leading_lepton_pt_GeV': [], 'leading_jet_pt_GeV': []}

    i = 0
    denom = min(len(dataset[0]), len(dataset[1])) 

    # While there are pair of files to combine
    while len(dataset[0]) and len(dataset[1]):

        # Check if output file exists
        if os.path.isfile(output):
            ans = input("'%s' already exists. Overwrite? (y,n): "%output)

            if ans.lower() != 'y':
                break

            os.remove(output)

        # Open LHE-files and combine two and two
        file_1 = dataset[0].pop()
        file_2 = dataset[1].pop()
        i += 1
        print("Combining datasets (%d/%d)..."%(i, denom))
        events, num_events = util.combine_LHE_files(file_1, file_2)
        print("Running analysis and cuts through %d events..." %num_events)

        for e in events:

            # Apply general cuts/filters
            if not pass_cuts(e): 
                continue

            # Get list of final state particles for easy access
            fs_particles = util.get_final_state_particles(e)

            # Declare event variables of interest
            pT = np.array([0., 0.])
            leading_lepton_pt = 0.
            leading_jet_pt = 0.

            for particle in fs_particles:

                if particle.id in invisible_particles:
                    p_invisible = util.FourMomentum.from_LHEparticle(particle)
                    pT += p_invisible.transverse_momentum(vector_out=True)

                if particle.id in charged_leptons:
                    p_lepton = util.FourMomentum.from_LHEparticle(particle)
                    pt_lepton = p_lepton.transverse_momentum()
                    
                    # Update leading lepton pt
                    if pt_lepton > leading_lepton_pt:
                        leading_lepton_pt = pt_lepton

                if particle.id in jet_hard:
                    p_jet = util.FourMomentum.from_LHEparticle(particle)
                    pt_jet = p_jet.transverse_momentum()
                    
                    # Update leading jet pt
                    if pt_jet > leading_jet_pt:
                        leading_jet_pt = pt_jet

            # Calculations of variables
            missing_pt = np.linalg.norm(pT)

            # Append results to data structure
            data['missing_pt_GeV'].append(missing_pt)
            data['leading_lepton_pt_GeV'].append(leading_lepton_pt)
            data['leading_jet_pt_GeV'].append(leading_jet_pt)

    # Write to output file
    print("Storing calculations in %s..." % output)
    df = pd.DataFrame(data)
    df.index.name = 'event'
    df.to_csv(output)
    print("Combination and analysis successful.")
    print(util.sep)
