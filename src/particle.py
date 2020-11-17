from numpy import inf
from scipy import constants


class Particle(object):
    def __init__(self, name, mass, lifetime=inf):
        self.Name = name
        self.M = mass / constants.e * constants.c ** 2 / 1e6 if mass < 1e-10 else mass
        self.Tau = lifetime

    def __repr__(self):
        return '{} particle with mass {:1.1f} MeV and lifetime {:1.1e} s'.format(self.Name, self.M, self.Tau)


Muon = Particle('Muon', constants.physical_constants['muon mass'][0], 2.1969e-6)
Pion = Particle('Pion', 139.57018, 2.6033e-8)
Electron = Particle('Electron', constants.electron_mass)
Positron = Particle('Positron', constants.electron_mass)
Proton = Particle('Proton', constants.proton_mass)
