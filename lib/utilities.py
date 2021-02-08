import numpy as np
import pylhe as lhe
import lib.mssm_particles as mssm

# Global constants
sep = '-'*45


class FourMomentum:
    def __init__(self, e=0, px=0, py=0, pz=0):
        """
        NOTE: Working in natural units (hbar = c = 1)

        :param e: Energy
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
        p1 = self.three_momentum()
        p2 = p.three_momentum()
        return self.e * p.e - np.dot(p1, p2)

    def __rmul__(self, scalar):
        self.e *= scalar
        self.px *= scalar
        self.py *= scalar
        self.pz *= scalar
        return self

    def __add__(self, p):
        """
        Define addition between two FourMomentum objects p1 and p2 (p1 + p2)
        """
        new_p = FourMomentum()
        new_p.e = self.e + p.e
        new_p.px = self.px + p.px
        new_p.py = self.py + p.py
        new_p.pz = self.pz + p.pz
        return new_p

    def __sub__(self, p):
        """
        Define subtraction between two FourMomentum objects p1 and p2 (p1 - p2)
        """
        return self + (-1.0 * p)

    def norm(self):
        """
        The Minkowski norm of a four vector p: sqrt(p * p)
        """
        return np.sqrt(self * self)

    def three_momentum(self):
        """
        :return: Spatial part of the four momentum (3-momentum)
        as np.array [px, py, pz]
        """
        return np.array([self.px, self.py, self.pz])

    def transverse_momentum(self, vector_out=False):
        """
        Get transverse momentum pT
        (transverse plane orthogonal to z-axis is conventional)

        Can return components or the magnitude.
        """
        p = np.array([self.px, self.py])

        if vector_out:
            return p
        else:
            return np.linalg.norm(p)

    def print(self, unit_e='GeV', unit_p='GeV'):
        """
        A method to print the FourMomentum in a nice way (with units of choice)
        """
        for mu in ['e', 'px', 'py', 'pz']:
            attribute = getattr(self, mu)
            if mu == 'e':
                print(" %s: %e %s" % (mu, attribute, unit_e))
            else:
                print("%s: %e %s" % (mu, attribute, unit_p))

    @staticmethod
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

    @staticmethod
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
    :param p: FourMomentum object (of some particle)
    :param m: mass of particle
    :param tol: tolerance to be on-shell
    :return: True if onshell, False otherwise
    """
    return np.isclose(p.norm(), m, rtol=rtol)


# Jet Tools
def is_jet(particle):
    """
    :param particle: LHEparticle object
    :return: True if particle is a parton jet candidate, False otherwise
    """
    return (particle.status == 1) and (abs(particle.id) in mssm.jet_hard)


def has_physical_jets(event, pt_cut):
    jets = get_jets(event)

    if len(jets) > 0:
        leading_jet_pt = max([p.transverse_momentum() for p in jets])

        # if leading pt is above the cut, return True (jet is physical)
        if leading_jet_pt > pt_cut:
            return True

    return False
    

def get_jets(event):
    jets = []
    for p in event.particles:
        if is_jet(p):
            jets.append(p)

    return FourMomentum.from_LHEparticles(jets)


def get_final_state_particles(event):
    fs_particles = []

    for p in event.particles:
        # if particle is final state, add to list
        if p.status == 1:
            fs_particles.append(p)

    return fs_particles



# Common kinematic variables
def get_missing_pt(event):
    fs_particles = get_final_state_particles(event)
    pT = np.array([0., 0.])
    for p in fs_particles:
        if p.id in mssm.invisible_particles:
            p_invs = FourMomentum.from_LHEparticle(p)
            pT += p_invs.transverse_momentum(vector_out=True)

    return np.linalg.norm(pT)



# LHE file tools
def combine_LHE_files(file_1, file_2, xsec_1=0, xsec_2=0):
    """
    :param file_1: path to the first lhe-file
    :param file_2: path to the second lhe-file
    :param xsec_1: total xsec for process in file_1
    :param xsec_2: total xsec for process in file_2

    return: List of events from file_1 and file_2 in proportion to their
    cross sections, resp.
    """
    events_1 = [e for e in lhe.readLHE(file_1)]
    n1 = len(events_1)
    if not xsec_1:
        events_1_init = lhe.readLHEInit(file_1)
        events_1_info = events_1_init['procInfo'][0]
        xsec_1 = events_1_info["xSection"]
    print("Dataset 1 initialised (%e events), cross section: %e pb."%(n1, xsec_1))

    events_2 = [e for e in lhe.readLHE(file_2)]
    n2 = len(events_2)
    if not xsec_2:
        events_2_init = lhe.readLHEInit(file_2)
        events_2_info = events_2_init['procInfo'][0]
        xsec_2 = events_2_info['xSection']
    print("Dataset 2 initialised (%e events), cross section: %e pb."%(n2, xsec_2))

    # Sampling probability for set 1
    p1 = xsec_1/(xsec_1 + xsec_2)

    events = []
    # Main loop (drawing samples while both sets are non empty)
    while n1 and n2:
        # pick a number 0<x<1 randomly
        x = np.random.uniform(0., 1.)

        # pick an event depending on outcome, remove element after selection
        if x < p1:
            e = events_1.pop()
            n1 -= 1
        else:
            e = events_2.pop()
            n2 -= 1

        events.append(e)

    # return combined dataset, and number of events (tuple)
    return events, len(events)
