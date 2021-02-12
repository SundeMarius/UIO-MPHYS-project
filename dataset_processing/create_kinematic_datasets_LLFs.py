#!/usr/bin/python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lib.utilities as util
import lib.config_parser as parse

from collections import OrderedDict

sep = 60*'-'

# Parse config file with paths/cuts
config = parse.read_config_file(sys.argv[1])

# Paths to LHE data directories 
path_LO = config['LO_path']
path_LO_NLO = config['LO_NLO_path']
# Path to desired destination of the processed output files (.csv)
path_to_output = config['output_path']

# Name of output files
output_filename = os.path.join(path_to_output, config['output_file'])


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



# Low feature variables (preparation)
particle_pdgs = ['11', '-11', '1000022_1', '1000022_2']
attributes = ['e', 'px', 'py', 'pz']

kinematic_variables = [[p+'_'+v for v in attributes] for p in particle_pdgs]

# First column is 0 or 1 (from LO or LO+NLO)
data_components = { var : [] for var in particle_pdgs }
data = {'target': [], **data_components}



# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut']) # GeV
missing_pt_cut = float(config['missing_pt_cut']) # GeV

# Wrap all relevant cuts into one function
def pass_cuts(event):

    # No physical jets
    cut_1 = not util.has_physical_jets(event, jet_pt_cut)

    # Lower bound on missing pt
    cut_2 = True if not missing_pt_cut else util.get_missing_pt(event) > missing_pt_cut

    # "Two leptons SF-OC" is automatically satisfied for these simulated processes

    return cut_1 and cut_2


# Iterate through available files
datasets_at_order = [LO_files, 
                     LO_NLO_files]
datasets_jet_split = [[LO_1, LO_2], 
                      [LO_NLO_1, LO_NLO_2]]

created_datasets = []

for f, files in enumerate(datasets_at_order):
    
    output = output_filename
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

            # Collect daughter products from sleptons
            #  1000011 -->  11 1000022
            # -1000011 --> -11 1000022 
            slep_daugthers = []
            aslep_daugthers = []

            particles = [p for p in e.particles]
            
            # This will return [11, 1000022]
            slep_daugthers = util.get_daughters(1000011, particles)
            slep_daugthers.sort(key=lambda x: x.id)
            # This will return [-11, 1000022]
            aslep_daugthers = util.get_daughters(-1000011, particles)
            aslep_daugthers.sort(key=lambda x: x.id)

            # Get the four momenta components (np.arrays)
            em, n1_1 = util.FourMomentum.from_LHEparticles(slep_daugthers)
            ep, n1_2 = util.FourMomentum.from_LHEparticles(aslep_daugthers)

            data['11'].append(em.components()) 
            data['-11'].append(ep.components()) 
            data['1000022_1'].append(n1_1.components())
            data['1000022_2'].append(n1_2.components())
            data['target'].append(f)

 
        print(" Done.")

        # Write to output file
        print("Dumping processed data to %s..." %output, end='')
        
        dfs = [pd.DataFrame(data['target'], columns=['target'])]

        # Add rest of the columns 
        for d, pdg in enumerate(particle_pdgs):
            df = pd.DataFrame(data[pdg])
            df = pd.DataFrame(df.values.tolist(), columns=kinematic_variables[d])
            dfs.append(df)
        
        # Create final dataframe to write to csv -- 'target' as first column
        df = pd.concat(dfs, axis=1)
        
        df.to_csv(
            output, 
            header=write_header, 
            index=False, 
            mode='a',
            float_format='%.6e'
        )
 
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
