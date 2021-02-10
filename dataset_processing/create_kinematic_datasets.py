#!/usr/bin/python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lib.utilities as util
import lib.config_parser as parse

sep = 60*'-'

# Parse config file with paths/cuts
config = parse.read_config_file(sys.argv[1])

# Paths to LHE data directories 
path_LO = config['LO_path']
path_LO_NLO = config['LO_NLO_path']
# Path to desired destination of the processed output files (.csv)
path_to_output = config['output_path']

# Name of output files
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


kin_vars_keys = \
[
    'missing_pt_GeV', 
    'leading_lepton_pt_GeV', 
    'leading_jet_pt_GeV'
]

data = { var : [] for var in kin_vars_keys }


# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut']) # GeV
missing_pt_cut = float(config['missing_pt_cut']) # GeV

# Wrap all relevant cuts into one function
def pass_cuts(event):
    # No physical jets
    cut_1 = not util.has_physical_jets(event, jet_pt_cut)
    # Lower bound on missing pt
    cut_2 = util.get_missing_pt(event) > missing_pt_cut
    # "Two leptons SF-OS" is automatically satisfied for these simulated processes
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

    i = 0
    ans = 'y'
    number_of_file_pairs = min(len(dataset[0]), len(dataset[1])) 
    write_header = True
 
    # While there are pair of files to combine.
    while i < number_of_file_pairs:

        # Check if output file exists
        if write_header and os.path.isfile(output):

            ans = input("'%s' already exists. Overwrite? (y,N): "%output).lower()

            if ans == 'y':
                os.remove(output)
            else:
                break

        # Open LHE-files from each process, combine
        i += 1
        file_1 = dataset[0].pop()
        file_2 = dataset[1].pop()
        print("Combining datasets (%d/%d)..."%(i, number_of_file_pairs))
        events, num_events = util.combine_LHE_files(file_1, file_2)
        print(f"Running analysis and cuts through {num_events:,} events...", end='')

        for e in events:

            # Apply general cuts/filters
            if not pass_cuts(e): 
                continue

            # Declare event variables of interest
            pT = np.zeros(2)
            leading_lepton_pt = 0.
            leading_jet_pt = 0.

            for particle in util.get_final_state_particles(e):
 
                p = util.FourMomentum.from_LHEparticle(particle)
 
                # Missing pt
                if util.is_invisible(particle):
                    pT += p.transverse_momentum(vector_out=True)
    
                # Leading Lepton pt
                elif util.is_charged_lepton(particle):
                    pt = p.transverse_momentum()
                    leading_lepton_pt = max(pt, leading_lepton_pt)

                # Leading Jet pt
                elif util.is_jet(particle):
                    pt = p.transverse_momentum()
                    leading_jet_pt = max(pt, leading_jet_pt)

            # Calculations of variables
            missing_pt = np.linalg.norm(pT)

            # Append results to data structure
            data[kin_vars_keys[0]].append(missing_pt)
            data[kin_vars_keys[1]].append(leading_lepton_pt)
            data[kin_vars_keys[2]].append(leading_jet_pt)

        print(" Done.")

        # Write to output file
        print("Dumping processed data to %s..." %output, end='')
        df = pd.DataFrame(data)
        df.to_csv(output, header=write_header, index=False, mode='a')
 
        # Empty data lists for next iteration
        for variabel in data.values():
            variabel.clear()

        print(" Done.")

        write_header = False

    if ans == 'y':
        created_datasets.append(output)
        print("Combination and analysis successful.")
        print(sep)

if created_datasets:
    print(*created_datasets, sep=', ', end=' ')
    print("created successfully.")
