#!/usr/bin/python3
import os
import sys
import pandas as pd
import numpy as np
import lib.utilities as util
import lib.config_parser as parse


# Parse config file with paths/cuts
config = parse.read_config_file(sys.argv[1])

# Paths to LHE data directories
path_LO = config['LO_path']
path_LO_NLO = config['LO_NLO_path']

# Path to desired destination of the processed output files (.csv)
path_to_output = config['output_path']
# Name of output file
output_filename = os.path.join(path_to_output, config['output_file'])


# Containers for LHE files
LO_files = []
LO_NLO_files = []

sub_dirs_LO = [x.path for x in os.scandir(path_LO) if x.is_dir()]
sub_dirs_LO_NLO = [x.path for x in os.scandir(path_LO_NLO) if x.is_dir()]
# Fill up containers with all the available files in the LHE data directories
# NOTE: This works as long as each process has its own directory
# NOTE: For instance "dislepton/" & "dislepton+jet/"
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


kin_vars_keys = [
    'mll',
    'mt',
    'ht',
    'met/ht',
    'delta-phi',
    'delta-R',
]

data = {'target': [], **{var: [] for var in kin_vars_keys}}


# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut'])  # GeV
missing_pt_cut = float(config['missing_pt_cut'])  # GeV


# Wrap all relevant cuts into one function
def pass_cuts(event):

    # No physical jets
    cut_1 = not util.has_physical_jets(event, jet_pt_cut)

    # Lower bound on missing pt
    cut_2 = util.get_missing_pt(event) > missing_pt_cut

    return cut_1 and cut_2


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
        events, num_events = util.combine_LHE_files(file_1, file_2)
        print(("Running analysis and cuts through "
               f"{num_events:,} events..."), end='')

        for e in events:

            # Check cuts
            if not pass_cuts(e):
                continue

            p_invs = util.FourMomentum()
            p_leptons = []
            for particle in util.get_final_state_particles(e):

                p = util.FourMomentum.from_LHEparticle(particle)

                # Collect leptons (e+ & e-)
                if util.is_charged_lepton(particle):
                    p_leptons.append(p)
                    continue

                # Missing pt
                if util.is_invisible(particle):
                    p_invs = p_invs + p
                    continue

            # Calculations
            p_l1, p_l2 = p_leptons
            p_lep_tot = p_l1 + p_l2

            pt_l1, phi_l1, eta_l1 = p_l1.collider_coordinates()
            pt_l2, phi_l2, eta_l2 = p_l2.collider_coordinates()

            pt_leps, phi_leps = p_lep_tot.collider_coordinates()[:2]
            pt_invs, phi_invs = p_invs.collider_coordinates()[:2]
            dphi = phi_l2 - phi_l1
            deta = eta_l2 - eta_l1

            mll = p_lep_tot.norm()
            delta_R = np.sqrt(dphi*dphi + deta*deta)
            delta_phi = phi_invs - phi_leps
            mt = np.sqrt(2*pt_l1*pt_l2*(1. - np.cos(delta_phi)))
            ht = pt_l1 + pt_l2
            met_ht_ratio = pt_invs/ht

            # NOTE: elements must be in same order as data dictionary
            kin_vars = [
                mll,
                mt,
                ht,
                met_ht_ratio,
                delta_phi,
                delta_R,
            ]

            data['target'].append(f)
            # Append results to data structure
            for key, dat in zip(kin_vars_keys, kin_vars):
                data[key].append(dat)

        print(" Done.")

        # Write to output file
        print("Dumping processed data to %s..." % output_filename, end='')
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

        print(" Done.")
        write_header = False

    if ans == 'y':
        print("Combination and analysis successful.")
        print(60*'-')

if ans == 'y':
    print(output_filename, " created successfully.")
