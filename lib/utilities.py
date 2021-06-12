from math import sqrt, atan2, log, isclose

import lib.mssm_particles as mssm


class FourMomentum:
    def __init__(self, e=0., px=0., py=0., pz=0.):
        """
        NOTE: Working in natural units (hbar = c = 1) by default.

        :param e : Energy
        :param px: x-comp 3-momentum
        :param py: y-comp 3-momentum
        :param pz: z-comp 3-momentum
        """
        self.e = e
        self.px = px
        self.py = py
        self.pz = pz

    def __mul__(self, p):
        """
        Define multiplication between two FourMomentum objects p1 and p2
        (contraction p1_mu * p2^mu)
        """
        return self.e*p.e - self.px*p.px - self.py*p.py - self.pz*p.pz

    def __rmul__(self, scalar):
        """
        Define scalar multiplication on a FourMomentum object p (scaling)
        """
        e = scalar*self.e
        px = scalar*self.px
        py = scalar*self.py
        pz = scalar*self.pz
        return FourMomentum(e, px, py, pz)

    def __add__(self, p):
        """
        Define addition between two FourMomentum objects p1 and p2
        (p1 + p2)
        """
        e = self.e + p.e
        px = self.px + p.px
        py = self.py + p.py
        pz = self.pz + p.pz
        return FourMomentum(e, px, py, pz)

    def __sub__(self, p):
        """
        Define subtraction between two FourMomentum objects p1 and p2
        (p1 - p2)
        """
        e = self.e - p.e
        px = self.px - p.px
        py = self.py - p.py
        pz = self.pz - p.pz
        return FourMomentum(e, px, py, pz)

    def norm(self):
        """
        Return the norm of a four vector p: sqrt(p * p) (= particle mass)
        """
        return sqrt(self*self)

    def as_tuple(self):
        """
        Return components as a Python tuple
        """
        return self.e, self.px, self.py, self.pz

    def transverse_momentum(self):
        """
        Get transverse momentum pT
        (xy-plane is conventionally the transverse plane, beam axis along z)

        Return components or the magnitude.
        """
        return sqrt(self.px**2 + self.py**2)

    def collider_coordinates(self):
        """
        Calculates pt, phi and eta (pseudo-rapidity) and return them
        as a tuple.

        Collider representation of three momentum.

        tan^2 (theta/2) = (p - p_z)/(p + p_z)
        """
        p = sqrt(self.px**2 + self.py**2 + self.pz**2)

        pt = sqrt(self.px**2 + self.py**2)
        phi = atan2(self.py, self.px)
        eta = .5*log((p + self.pz)/(p - self.pz))

        return pt, phi, eta

    def print(self, unit_e='GeV', unit_p='GeV'):
        """
        A method to print the FourMomentum in a nice way (with units of choice)
        """
        print("%s :  %.6e %s" % ('e', self.e, unit_e))
        print("%s:  %.6e %s" % ('px', self.px, unit_p))
        print("%s:  %.6e %s" % ('py', self.py, unit_p))
        print("%s:  %.6e %s" % ('pz', self.pz, unit_p))

    @ staticmethod
    def from_LHEparticle(lhe_particle):
        """
        Construct a FourMomentum object from a pyhle-particle object
        (see py-module "Pylhe" by H. Lukas.)
        """
        e = lhe_particle.e
        px = lhe_particle.px
        py = lhe_particle.py
        pz = lhe_particle.pz
        return FourMomentum(e, px, py, pz)

    @ staticmethod
    def from_LHEparticles(lhe_particles):
        """
        Construct a list of FourMomentum objects from a list of lhe-particles
        (see py-module "Pylhe" by H. Lukas.)
        """
        return [FourMomentum(p.e, p.px, p.py, p.pz) for p in lhe_particles]


# General tools for HEP
def invariant_mass(particle_momenta):
    """
    :param particle_momenta: list of FourMomentum objects

    :return: invariant mass of particle 1, particle 2, particle 3, ...
    [ M = sqrt((e1+e2+...)**2 - (p1+p2+...)**2) ]
    """
    total_momentum = FourMomentum()  # Null vector
    for p in particle_momenta:
        total_momentum = total_momentum + p
    return total_momentum.norm()


def is_onshell(p, m, rtol=1e-2):
    """
    :param p: FourMomentum object
    :param m: mass of particle
    :param tol: tolerance to be on-shell
    :return: True if onshell, False otherwise
    """
    return isclose(p.norm(), m, rtol=rtol)


# LHE tools
def is_final_state(particle):

    return particle.status == 1


def get_final_state_particles(event_particles):

    return [p for p in event_particles if is_final_state(p)]


def get_daughters(parent_pdg, event_particles):

    pdgs = [p.id for p in event_particles]

    # The line number in the LHE event specifies the parent
    parent_ln = pdgs.index(parent_pdg) + 1

    return [p for p in event_particles if p.mother1 == parent_ln]


def combine_LHE_files(file_1, file_2):
    """

    :param file_1: path to the first lhe-file
    :param file_2: path to the second lhe-file

    return: List of events from file_1 and file_2 in proportion to their
    cross sections, resp.
    """
    import pylhe as lhe
    from random import random
    
    events_1 = lhe.readLHE(file_1)
    events_1_init = lhe.readLHEInit(file_1)
    events_1_info = events_1_init['procInfo'][0]
    xsec_1 = events_1_info["xSection"]
    n1 = lhe.readNumEvents(file_1)
    print(f'Dataset 1 initialised ({n1:,} events), cross section: {xsec_1:e} pb.')

    events_2 = lhe.readLHE(file_2)
    events_2_init = lhe.readLHEInit(file_2)
    events_2_info = events_2_init['procInfo'][0]
    xsec_2 = events_2_info['xSection']
    n2 = lhe.readNumEvents(file_2)
    print(f'Dataset 2 initialised ({n2:,} events), cross section: {xsec_2:e} pb.')

    # Sampling probability for dataset 1
    p1 = xsec_1/(xsec_1 + xsec_2)

    # Main loop (drawing samples while both sets are non empty)
    while True:

        # yield an event from 1 or 2 depending on random number between (0,1)
        try:
            if random() < p1:
                yield next(events_1)
            else:
                yield next(events_2)
        except StopIteration:
            return


# Particle tools
def is_jet(particle):

    return is_final_state(particle) and (abs(particle.id) in mssm.jet_hard)


def get_jets(event_particles):

    return (p for p in event_particles if is_jet(p))


def has_physical_jets(event_particles, pt_cut):

    jets = FourMomentum.from_LHEparticles(get_jets(event_particles))

    return any([True for j in jets if j.transverse_momentum() > pt_cut])


def is_charged_lepton(particle):

    return abs(particle.id) in mssm.charged_leptons


def is_invisible(particle):

    return abs(particle.id) in mssm.invisible_particles
