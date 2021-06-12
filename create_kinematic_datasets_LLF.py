#!/usr/bin/python3
import os
import sys

import pandas as pd

import lib.config_parser as parse
import lib.utilities as util

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

feature_data = {var:[] for var in kinematic_variables}
data = {'target': [], **feature_data}


# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut_GeV'])


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
                sys.exit()

        # Open LHE-files from each process, combine
        i += 1
        file_1 = files[0].pop()
        file_2 = files[1].pop()
        print("Combining datasets (%d/%d)..." % (i, number_of_file_pairs))
        events = util.combine_LHE_files(file_1, file_2)

        for e in events:

            event_particles = e.particles

            # Apply base cut
            if util.has_physical_jets(event_particles, jet_pt_cut):
                continue

            # Collect daughter products from sleptons
            #  1000011 -->  11 1000022
            # -1000011 --> -11 1000022

            # This will yield [11, 1000022]
            slep_daugthers = util.get_daughters(1000011, event_particles)
            slep_daugthers.sort(key=lambda x: x.id)
            # This will yield [-11, 1000022]
            aslep_daugthers = util.get_daughters(-1000011, event_particles)
            aslep_daugthers.sort(key=lambda x: x.id)

            em, n1_1 = util.FourMomentum.from_LHEparticles(slep_daugthers)
            ep, n1_2 = util.FourMomentum.from_LHEparticles(aslep_daugthers)

            # NOTE: elements must be in same order as data dictionary
            feature_row = [em.e, em.px, em.py, em.pz, 
                           ep.e, ep.px, ep.py, ep.pz,
                           n1_1.e, n1_1.px, n1_1.py, n1_1.pz,
                           n1_2.e, n1_2.px, n1_2.py, n1_2.pz,
                           ]

            data['target'].append(f)

            # Append results to data structure
            for key, dat in zip(kinematic_variables, feature_row):
                data[key].append(dat)

        print("Analysis done.")

        # Write to output file
        df = pd.DataFrame(data)
        df.to_csv(
            output_filename,
            header=write_header,
            float_format='%.6e',
            index=False,
            mode='a'
        )

        # Empty data lists for next iteration
        for variabel in data.values():
            variabel.clear()

        # Write to output file
        print("Processed data dumped to %s." % output_filename)

        write_header = False

    if ans == 'y':
        print("Combination and analysis successful.")
        print(60*'-')

if ans == 'y':
    print(output_filename, " created successfully.")
