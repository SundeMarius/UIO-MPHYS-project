"""
Import this file to have access to different particle PDG ID's

Version: WIP
"""
quarks = [i for i in range(1,7)]
squarks = [1000000+i for i in range(1,7)]

charged_leptons = [11, 13, 15]
neutral_leptons = [12, 14, 16]
charged_sleptons = [1000011, 1000013, 1000015]
neutral_sleptons = [1000012, 1000014, 1000016]

gauge_bosons = [21, 22, 23] # gluon, gamma and Z
neutralinos = [1000022, 1000023, 1000025, 1000035]
charginos = [1000024, 1000037]

jet_hard = [21] + quarks
visible_particles = [22] + charged_leptons + charged_sleptons + charginos
invisible_particles = neutralinos + neutral_leptons + neutral_sleptons
