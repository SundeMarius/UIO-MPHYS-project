#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import lib.utilities as util
import lib.config_parser as parse


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
# Fill up containers with all the available files in the LHE data directories
# NOTE: This works as long as each process has its own directory
# NOTE: (for instance "dislepton/" & "dislepton+jet/")
for sub_dir in sub_dirs_LO:
    LHE_files = []
    for dirpath, dirnames, filenames in os.walk(sub_dir):
        LHE_files += [os.path.join(dirpath, file)
                      for file in filenames if '.lhe' in file]
    LO_files.append(LHE_files)
for sub_dir in sub_dirs_LO_NLO:
    LHE_files = []
    for dirpath, dirnames, filenames in os.walk(sub_dir):
        LHE_files += [os.path.join(dirpath, file)
                      for file in filenames if '.lhe' in file]
    LO_NLO_files.append(LHE_files)


# Low feature variables (preparation)
particle_pdgs = ['11', '-11', '1000022_1', '1000022_2']
attributes = ['e', 'px', 'py', 'pz']

kinematic_variables = sum([[p+'_'+v for v in attributes]
                           for p in particle_pdgs], [])

data = {'target': [], 'features': []}


# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut_GeV'])


# Wrap all relevant cuts into one function
def pass_cuts(event):

    # No physical jets
    cut_1 = not util.has_physical_jets(event, jet_pt_cut)

    return cut_1


# Iterate through available files
write_header = True

for f, files in enumerate([LO_files, LO_NLO_files]):

    i = 0
    ans = 'y'
    number_of_file_pairs = min(len(files[0]), len(files[1]))

    # While there are pair of files to combine.
    while i < number_of_file_pairs:

        # Check if output file exists
        if write_header and os.path.isfile(output_filename):

            ans = input("'%s' already exists. Overwrite? (y,N): " %
                        output_filename).lower()

            if ans == 'y':
                os.remove(output_filename)
            else:
                sys.exit(1)

        # Open LHE-files from each process, combine
        i += 1
        file_1 = files[0].pop()
        file_2 = files[1].pop()
        print("Combining datasets (%d/%d)..." % (i, number_of_file_pairs))
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

            # This will yield [11, 1000022]
            slep_daugthers = util.get_daughters(1000011, e.particles)
            slep_daugthers.sort(key=lambda x: x.id)
            # This will yield [-11, 1000022]
            aslep_daugthers = util.get_daughters(-1000011, e.particles)
            aslep_daugthers.sort(key=lambda x: x.id)

            # Get the four momenta components (np.arrays)
            em, n1_1 = util.FourMomentum.from_LHEparticles(slep_daugthers)
            ep, n1_2 = util.FourMomentum.from_LHEparticles(aslep_daugthers)

            em, n1_1 = em.components(), n1_1.components()
            ep, n1_2 = ep.components(), n1_2.components()

            feature_row = np.concatenate((em, ep, n1_1, n1_2), axis=None)

            data['features'].append(feature_row)
            data['target'].append(f)

        print(" Done.")

        # Write to output file
        print("Dumping processed data to %s..." % output_filename, end='')

        # Create final dataframe to write to csv -- 'target' as first column
        df0 = pd.DataFrame(data['target'], columns=['target'])
        df = pd.DataFrame(data['features'], columns=kinematic_variables)

        df = pd.concat([df0, df], axis=1)

        df.to_csv(
            output_filename,
            header=write_header,
            index=False,
            mode='a',
            float_format='%.6e'
        )

        data['target'].clear()
        data['features'].clear()
        write_header = False

        print(" Done.")

    if ans == 'y':
        print("Combination and analysis successful.")
        print(60*'-')

if ans == 'y':
    print(output_filename, " created successfully.")
