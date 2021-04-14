#!/usr/bin/python3
import os
import sys

import pandas as pd

from math import sqrt, cos
from mt2 import mt2

import lib.config_parser as parse
import lib.utilities as util

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
    'mt2',
    'ht',
    'met',
    'met/ht',
    'delta-phi',
    'delta-R',
]

data = {'target': [], **{var: [] for var in kin_vars_keys}}


# Cut/filter parameters
jet_pt_cut = float(config['jet_pt_cut'])  # GeV
missing_pt_cut = float(config['missing_pt_cut'])  # GeV
mt2_cut = float(config['mt2_cut'])  # GeV
mll_cut = float(config['mll_cut'])  # GeV


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

            if util.has_physical_jets(event_particles, jet_pt_cut):
                continue

            # Get pairs of decayed particles (lepton and neutralino)
            pair_1 = util.get_daughters(1000011, event_particles)
            pair_1.sort(key=lambda x: x.id)
            pair_2 = util.get_daughters(-1000011, event_particles)
            pair_2.sort(key=lambda x: x.id)
            p_l1, p_invs1 = util.FourMomentum.from_LHEparticles(pair_1)
            p_l2, p_invs2 = util.FourMomentum.from_LHEparticles(pair_2)

            p_lep_tot = p_l1 + p_l2
            pt_leps, phi_leps, _ = p_lep_tot.collider_coordinates()

            pt_l1, phi_l1, eta_l1 = p_l1.collider_coordinates()
            pt_l2, phi_l2, eta_l2 = p_l2.collider_coordinates()

            m_invs = p_invs1.norm()
            p_invs = p_invs1 + p_invs2
            pt_invs, phi_invs, _ = p_invs.collider_coordinates()

            dphi = phi_l2 - phi_l1
            deta = eta_l2 - eta_l1

            # Calculations
            mll = p_lep_tot.norm()
            delta_R = sqrt(dphi**2 + deta**2)
            delta_phi = phi_invs - phi_leps
            mt = sqrt(2*pt_l1*pt_l2*(1. - cos(delta_phi)))
            mt_2 = mt2(
                0, p_l1.px, p_l1.py,
                0, p_l2.px, p_l2.py,
                p_invs.px, p_invs.py,
                m_invs, m_invs
            )
            ht = pt_l1 + pt_l2
            met_ht_ratio = pt_invs/ht

            # Apply HLF cuts
            if (
                mll < mll_cut or
                mt_2 < mt2_cut or
                pt_invs < missing_pt_cut
            ):
                continue

            # NOTE: elements must be in same order as data dictionary
            kin_vars = [
                mll,
                mt,
                mt_2,
                ht,
                pt_invs,
                met_ht_ratio,
                delta_phi,
                delta_R,
            ]

            data['target'].append(f)
            # Append results to data structure
            for key, dat in zip(kin_vars_keys, kin_vars):
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

        write_header = False

        print("Processed data dumped to %s." % output_filename)

    if ans == 'y':
        print("Combination and analysis successful.")
        print(60*'-')

if ans == 'y':
    print(output_filename, " created successfully.")
