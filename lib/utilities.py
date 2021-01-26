import numpy as np
import scipy as sp
import pylhe as lhe
import matplotlib.pyplot as plt
import lib.mssm_particles as mssm

#Global constants
sep = '-'*50
pTcut = 20. #GeV


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
        Define multiplication between two FourMomentum objects p1 and p2 (contraction p1_mu * p2^mu)
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
        :return: Spatial part of the four momentum (3-momentum) as np.array [px, py, pz]
        """
        return np.array([self.px, self.py, self.pz])

    def transverse_momentum(self, vector_out=False):
        """
        Get transverse momentum pT (transverse plane orthogonal to z-axis is conventional)

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
        for mu in ['e','px','py','pz']:
            attribute = getattr(self, mu)
            if mu == 'e':
                print(" %s: %e %s" %(mu, attribute, unit_e))
            else:
                print("%s: %e %s" %(mu, attribute, unit_p))

    @staticmethod
    def from_LHEparticle(lhe_particle):
        """
        Construct a FourMomentum object from a pyhle-particle object (see py-module "Pylhe" by H. Lukas.)
        """
        e = lhe_particle.e
        px = lhe_particle.px
        py = lhe_particle.py
        pz = lhe_particle.pz
        return FourMomentum(e, px, py, pz)

    @staticmethod
    def from_LHEparticles(lhe_particles):
        """
        Construct a list of FourMomentum objects from a list of lhe-particles (see py-module "Pylhe" by H. Lukas.)
        """
        return [FourMomentum(p.e, p.px, p.py, p.pz) for p in lhe_particles]



# General tools for HEP
def invariant_mass(particle_momenta):
    """
    :param particle_momenta: list of FourMomentum objects

    :return: invariant mass of particle 1, particle 2, particle 3, ... [ M = sqrt((e1+e2+...)**2 - (p1+p2+...)**2) ]
    """
    total_momentum = FourMomentum() # Null vector
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


def is_jet(particle):
    return (particle.status==1) and (abs(particle.id) in mssm.jet_hard)


def has_physical_jets(event, pt_cut=0):
    for p in event.particles:
        pt = FourMomentum.from_LHEparticle(p).transverse_momentum()
        if pt_cut:
            if is_jet(p) and pt <= pt_cut: return True
        else:
            if is_jet(p): return True

    return False


def get_jets(event):
    jets = []
    for p in event.particles:
        if is_jet(p):
            jets.append(p)

    return FourMomentum.from_LHEparticles(jets)



# LHE file tools
def get_final_state_particles(event):
    """

    """
    fs_particles = { id : [] for id in [p.id for p in event.particles] }

    for p in event.particles:
        # if particle is final state, add to list
        if p.status == 1:
            fs_particles[p.id].append(p)
    
    # return dictionary
    return fs_particles


def read_LHEfile(file, pt_cut=20):
    events = lhe.readLHE(file)

    new_events = []
    n = 0
    for e in events:
        # Get jets from event if there are any, check pt_cut
        # Event is accepted if n_jet == 0, or n_jet > 0 and leading_pt < pt_cut.
        jets = get_jets(e)
        n_jets = len(jets)
        if n_jets > 0:
            leading_jet_pt = max([p.transverse_momentum() for p in jets])

            # if leading pt is above the cut, skip event (contine next iteration)
            if leading_jet_pt > pt_cut:
                continue

        new_events.append(e)
        n += 1

    return new_events, n


def combine_LHE_files(file_1, file_2, xsec_1=0, xsec_2=0, pt_cut=20):
    """
    :param file_1: path to the first lhe-file
    :param file_2: path to the second lhe-file
    :param xsec_1: total xsec for process in file_1
    :param xsec_2: total xsec for process in file_2

    return: List of events from file_1 and file_2 in proportion to their cross sections, resp.
    """
    events_1 = lhe.readLHE(file_1)
    events_1_set = set([e for e in events_1])
    n1 = len(events_1_set)
    if not xsec_1:
        events_1_init = lhe.readLHEInit(file_1)
        events_1_info = events_1_init['procInfo'][0]
        xsec_1 = events_1_info["xSection"]
    print("Dataset 1 initialised (%e events), xsec: %e pb."%(n1, xsec_1))

    events_2 = lhe.readLHE(file_2)
    events_2_set = set([e for e in events_2])
    n2 = len(events_2_set)
    if not xsec_2:
        events_2_init = lhe.readLHEInit(file_2)
        events_2_info = events_2_init['procInfo'][0]
        xsec_2 = events_2_info['xSection']
    print("Dataset 2 initialised (%e events), xsec: %e pb."%(n2, xsec_2))

    # Sampling probabilities for set 1 and set 2 resp.
    p1 = xsec_1/(xsec_1 + xsec_2)
    p2 = xsec_2/(xsec_1 + xsec_2) # 1 - p1
    print("p1 = %f\np2 = %f"%(p1, p2))
    print("pt_cut = %f GeV"%pt_cut)

    new_events = []
    # Main loop (drawing samples while both sets are non empty)
    n = 0
    while n1 and n2:
        # pick a number 0<x<1 randomly
        x = np.random.uniform(0.,1.)

        # pick an event depending on outcome, remove element after selection
        if x < p1:
            e = events_1_set.pop()
            n1 -= 1
        else:
            e = events_2_set.pop()
            n2 -= 1

        # Get jets from event if there are any, check pt_cut
        # Event is accepted if n_jet == 0, or n_jet > 0 and leading_pt < pt_cut.
        jets = get_jets(e)
        n_jets = len(jets)
        if n_jets > 0:
            leading_jet_pt = max([p.transverse_momentum() for p in jets])

            # if leading pt is above the cut, skip event (contine next iteration)
            if leading_jet_pt > pt_cut:
                continue

        new_events.append(e)
        n += 1

    # return combined data set, and number of events (tuple)
    return new_events, n



# Plotting tools
class Plot:
    def __init__(self, xlabel, ylabel, title, logy=False, logx=False):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.logy_enabled = logy
        self.logx_enabled = logx
        self.series = []
        self.histograms = []
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1, 1, 1)

        #Set title, axis labels, axis scales etc.
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(0,600)
        #self.ax.set_ylim(0,4000)

        if self.logy_enabled: self.ax.set_yscale('log')
        if self.logx_enabled: self.ax.set_xscale('log')

        self.ax.grid()

    def add_series(self, x, y, **kwargs):
        """
        :param x: 1D np.array with x-data points
        :param y: 1D np.array with y-data points
        """
        if x.ndim == 1 and y.ndim == 1 and len(x) == len(y):
            self.series.append((x, y, kwargs))
        else:
            raise Exception("Error: appended \
                    series has to be an 1-D np.array and have equal dimensions.")

    def add_histogram(self, counts, bins, **kwargs):
        """
        :param counts: 1D np.array of the counts
        :param bins: 1D np.array of the bin edges, or string to specify how
        to calculate bin edges (see numpy doc on np.histogram)
        """
        if counts.ndim == 1:
            self.histograms.append((counts, bins, kwargs))
        else:
            raise Exception("Error: appended \
                    histogram data has to be an 1-D np.array.")

    def plot_series(self):
        for s in self.series:
            self.ax.plot(s[0], s[1], **s[2])
        self.ax.legend()

    def plot_histograms(self):
        for hist in self.histograms:
            count = hist[0]
            bins = hist[1]
            self.ax.bar(bins[:-1] + np.diff(bins) / 2, count, np.diff(bins), **hist[2])
        self.ax.legend()

    def plot_all(self):
        self.plot_series()
        self.plot_histograms()

    def display(self):
        plt.show()

    def savefig(self, filename):
        self.ax.savefig(filename)

    @staticmethod
    def from_config_file(filename):
        """
        Create plot with specific config from a prewritten textfile with configs??
        Formate of file:
            ...
        """
        pass
