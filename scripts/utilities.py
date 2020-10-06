import numpy as np
import scipy as sp
import pylhe as lhe
import matplotlib.pyplot as plt

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
        p1 = self.three_momentum()
        new_p = FourMomentum()
        new_p.e = self.e + p.e
        new_p.px = self.px + p.px
        new_p.py = self.py + p.py
        new_p.pz = self.pz + p.pz
        return new_p

    def __sub__(self, p):
        return self + (-1.0 * p)

    def norm(self):
        """
        The Minkowski norm of a four vector
        """
        return np.sqrt(self * self)

    def three_momentum(self):
        """
        :return: Spatial part of the four momentum (3-momentum) as np.array [px, py, pz]
        """
        return np.array([self.px, self.py, self.pz])

    def transverse_momentum(self, vector_out=False):
        """
        Get transverse momentum pT (assuming transverse plane is orthogonal to z-axis)
        """
        p = np.array([self.px, self.py])

        if vector_out:
            return p
        else:
            return np.linalg.norm(p) 

    def print(self, unit_e='', unit_p=''):
        for mu in ['e','px','py','pz']:
            attribute = getattr(self, mu)
            if mu == 'e':
                print("%s: %e %s" %(mu, attribute, unit_e))
            else:
                print("%s: %e %s" %(mu, attribute, unit_p))

    @staticmethod
    def from_LHEparticle(lhe_particle):
        e = lhe_particle.e
        px = lhe_particle.px
        py = lhe_particle.py
        pz = lhe_particle.pz
        return FourMomentum(e, px, py, pz)


# General tools for HEP
def invariant_mass(particle_momenta):
    """
    :param particle_momenta: list of FourMomentum objects
    :return: invariant mass of particle 1, particle 2, particle 3, ... 
    [ M = sqrt((e1+e2+...)**2 - (p1+p2+...)**2) ]
    """
    total_momentum = FourMomentum() # Null vector
    for p in particle_momenta:
        total_momentum = total_momentum + p
    return total_momentum.norm()


def is_onshell(p, m, tol=1e-1):
    """
    :param p: FourMomentum object (of some particle)
    :param m: mass of particle
    :param tol: tolerance to be on-shell
    :return: True if onshell, False otherwise
    """
    return np.abs(p.norm() - m) <= tol


# LHE file tools
def get_final_state_events(file, particle_ids):
    """
    :param file: the path to the LHE-file with events
    :param particle_ids: list of the particle ID's of interest (which final 
    state particles you're interested in). For instance: selektron+ has id 1000011 

    :return: an MxN matrix with m events and the n final-state particles of interest, number of events
    """
    matrix = []
    n_events = 0
    events = lhe.readLHE(file)
    for e in events:
        particles = []
        event_particles = {int(p.id) : p for p in e.particles}
        event_ids = list(event_particles)
        for fs_id in particle_ids:
            if fs_id in event_ids:
                particles.append(event_particles[int(fs_id)])  
            else:
                print("Error: Could not find particle %s in %s, event %d"%(fs_id, file, n_events+1))
                exit(1)
                
        matrix.append(particles)
        n_events += 1

    return matrix, n_events


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


# Deprecated #
def plot_hist(data_points, bins="auto", xlabel="", ylabel="", title="", logy=False, logx=False):
    plt.hist(data_points, bins=bins, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    if logx: plt.xscale('log')
    plt.ylabel(ylabel)
    if logy: plt.yscale('log')
    plt.grid()

