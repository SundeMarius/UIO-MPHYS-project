#!/usr/bin/python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lib.utilities as util
import lib.config_parser as parse
from lib.mssm_particles import invisible_particles, charged_leptons, jet_hard

# Parse config file with paths/cuts
config = parse.read_config_file(sys.argv[1])

# Enter paths to LHE data directories 
# path_LO = "/home/mariusss/University/master_project/data/LO/200_50"
# path_LO_NLO = "/home/mariusss/University/master_project/data/LO+NLO/200_50"
path_LO = config['LO_path']
path_LO_NLO = config['LO_NLO_path']
# Enter path to desired destination of the processed output files (.csv)
# path_to_output = "/home/mariusss/University/master_project/data"
path_to_output = config['output_path']


#parameter_point = path_LO.split(sep='/')[-1]
# Create filenames to store output (optional filenames from argument list)
#if len(sys.argv) == 3:
#    path_to_output, output_filename_LO, output_filename_LO_NLO =\
#    sys.argv[1],\
#    sys.argv[2], sys.argv[3]
#else:
#    print("Invalid input -- using defaults.")
#    print("Usage: %s 'output-path' 'filename_LO.csv' 'filename_LO+NLO.csv'"%sys.argv[0])
#    output_filename_LO, output_filename_LO_NLO = "LO_output_%s.csv"%parameter_point,\
#    "LO+NLO_output_%s.csv"%parameter_point
output_filename_LO = config['LO_file']
output_filename_LO_NLO = config['LO_NLO_file']

output_filename_LO = os.path.join(path_to_output, output_filename_LO)
output_filename_LO_NLO = os.path.join(path_to_output, output_filename_LO_NLO)


# Containers for LHE files
LO_files = []
LO_NLO_files = []

sub_dirs_LO = [x.path for x in os.scandir(path_LO) if x.is_dir()]
sub_dirs_LO_NLO = [x.path for x in os.scandir(path_LO_NLO) if x.is_dir()]
# Fill up containers with all the available files in the LHE data directories (recursively)
for sub_dir in sub_dirs_LO:
    LHE_files = []
    for dirpath, dirnames, filenames in os.walk(sub_dir):
        LHE_files += [os.path.join(dirpath, file) for file in filenames if '.lhe' in file]
    LO_files.append(LHE_files)
for sub_dir in sub_dirs_LO_NLO:
    LHE_files = []
    for dirpath, dirnames, filenames in os.walk(sub_dir):
        LHE_files += [os.path.join(dirpath, file) for file in filenames if '.lhe' in file]
    LO_NLO_files.append(LHE_files)

# Separate files with/without explicit jet
# NOTE: This works as long as each process has its own directory (for instance "dislepton/" & "dislepton+jet/")
# NOTE: So there is one list in LO_files for each process (in this instance two lists bcs. two processes)
LO_1, LO_2 = LO_files
LO_NLO_1, LO_NLO_2 = LO_NLO_files


# Cut/filter parameters
# jet_pt_cut = 20. # GeV
jet_pt_cut = float(config['jet_pt_cut']) # GeV
# missing_pt_cut = 40. # GeV
missing_pt_cut = float(config['missing_pt_cut']) # GeV

# Wrap all relevant cuts into one function
def pass_cuts(event):
    cut_1 = not util.has_physical_jets(event, jet_pt_cut)
    cut_2 = util.get_missing_pt(event) > missing_pt_cut
    return cut_1 and cut_2

# Iterate through available files
datasets_at_order = [LO_files, 
                     LO_NLO_files]
datasets_jet_split = [[LO_1, LO_2], 
                      [LO_NLO_1, LO_NLO_2]]
outputs = [output_filename_LO, 
           output_filename_LO_NLO]

created_datasets = []

for f, files in enumerate(datasets_at_order):
    
    output = outputs[f]
    dataset = datasets_jet_split[f]
    
    data = \
    {
        'missing_pt_GeV': [], 
        'leading_lepton_pt_GeV': [], 
        'leading_jet_pt_GeV': []
    }

    i = 0
    ans = 'y'
    denom = min(len(dataset[0]), len(dataset[1])) 

    # While there are pair of files to combine
    while len(dataset[0]) and len(dataset[1]):

        # Check if output file exists
        if os.path.isfile(output):
            ans = input("'%s' already exists. Overwrite? (y,n): "%output).lower()

            if ans == 'y':
                os.remove(output)
            else:
                break

        # Open LHE-files and combine two and two
        file_1 = dataset[0].pop()
        file_2 = dataset[1].pop()
        i += 1
        print("Combining datasets (%d/%d)..."%(i, denom))
        events, num_events = util.combine_LHE_files(file_1, file_2)
        print("Running analysis and cuts through %d events..." %num_events, end='')

        for e in events:

            # Apply general cuts/filters
            if not pass_cuts(e): 
                continue

            # Get list of final state particles
            fs_particles = util.get_final_state_particles(e)

            # Declare event variables of interest
            pT = np.array([0.e0, 0.e0])
            leading_lepton_pt = 0.e0
            leading_jet_pt = 0.e0

            for particle in fs_particles:
                
                # Missing pt
                if particle.id in invisible_particles:
                    p_invisible = util.FourMomentum.from_LHEparticle(particle)
                    pT += p_invisible.transverse_momentum(vector_out=True)
                
                # Leading Lepton pt
                elif particle.id in charged_leptons:
                    p_lepton = util.FourMomentum.from_LHEparticle(particle)
                    pt_lepton = p_lepton.transverse_momentum()
                    # Update leading lepton pt
                    if pt_lepton > leading_lepton_pt:
                        leading_lepton_pt = pt_lepton

                # Leading Jet pt
                elif particle.id in jet_hard:
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
        
        print(" Done.")

    if ans == 'y':
        # Write to output file
        print("Storing calculations in %s..." % output, end='')
        df = pd.DataFrame(data)
        df.index.name = 'event'
        df.to_csv(output)
        created_datasets.append(output)
        print(" Done.")
        print("Combination and analysis successful.")
        print(util.sep)

if created_datasets:
    print(*created_datasets, sep=', ', end=' ')
    print("created successfully.")
