#!/usr/bin/env python
from root_draw import *
import periodictable as pt
from numpy import log


class Element(object):
    def __init__(self, el: pt.core.Element, excitation_energy, cd, x0, x1, a, k, d0=0., density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.IE = excitation_energy * 1e-6
        self.Density = choose(density, el.density)
        self.C = cd
        self.X0 = x0
        self.X1 = x1
        self.A0 = a
        self.K = k
        self.D0 = d0
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV


class Particle(object):
    def __init__(self, name, mass, lifetime):
        self.Name = name
        self.M = mass / constants.e * constants.c**2 / 1e6 if mass < 1e-10 else mass
        self.Tau = lifetime

    def __repr__(self):
        return '{} particle with mass {:1.1f} MeV and lifetime {:1.1e} s'.format(self.Name, self.M, self.Tau)


K = 4 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4  # MeV * cm^2 / mol
Si = Element(pt.silicon, 173, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14)
Dia = Element(pt.carbon, 78, 2.868, -0.0178, 2.3415, 0.26142, 2.8697, density=3.5, name='Diamond')
Cu = Element(pt.copper, 322, 4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08)

Muon = Particle('Muon', constants.physical_constants['muon mass'][0], 2.1969e-6)
Pion = Particle('Pion', 139.57018, 2.6033e-8)
Electron = Particle('Electron', constants.electron_mass, 2.6033e-8)


def w_max(p, m):
    """returns maximum energy transfer in a single collsion with a free electron in [MeV]"""
    return 2 * M_E * beta_gamma(p, m) ** 2 / (1 + lorentz_factor(calc_speed(p, m)) / m + (M_E / m) ** 2)


def get_x(p, part: Particle):
    return log10(p / part.M)


def density_correction(p, part: Particle, el: Element):
    x = get_x(p, part)
    if x >= el.X1:
        return 2 * log(10) * x - el.C
    elif el.X0 <= x < el.X1:
        return 2 * log(10) * x - el.C + el.A0 * (el.X1 - x) ** el.K
    else:
        return el.D0 * 10 ** (2 * (x - el.X0))


def bethe_bloch(p, part: Particle, el: Element, dens_corr=True):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV * cm^2 / g]"""
    b = calc_speed(p, part.M)
    d = density_correction(p, part, el) if dens_corr else 0
    return K * el.Z / el.A / b ** 2 * (.5 * log(2 * M_E * b ** 2 * lorentz_factor(b) * w_max(p, part.M) / el.IE ** 2) - b ** 2 - d / 2)


def draw_density_correction(part: Particle, el: Element, pmin=10, pmax=1e6):
    f = Draw.make_tf1('Density Correction for {}s in {}'.format(part.Name, el.Name), density_correction, pmin, pmax, part=part, el=el)
    Draw.histo(f, logx=True, logy=True)
    format_histo(f, x_tit='#beta#gamma', y_tit='Value', y_off=1.1, center_y=True, center_x=True)


def draw_bethe(part: Particle, el: Element, pmin=90, pmax=1e6):
    f = Draw.make_tf1('Bethe-Bloch for {}s in {}'.format(part.Name, el.Name), bethe_bloch, pmin, pmax, part=part, el=el)
    f0 = Draw.make_tf1('', bethe_bloch, pmin, pmax, 4, part=part, el=el, dens_corr=False)
    Draw.histo(f, logx=True, logy=True)
    Draw.legend([f, f0], ['Bethe Bloch', 'without #delta'], 'l')
    format_histo(f, x_tit='{} Momentum [Mev/c]'.format(part.Name), y_tit='Mass Stopping Power [MEV cm^{2}/g]', y_off=1.3, center_x=True, center_y=True, y_range=[1, 2 * f.GetMaximum()], x_off=1.3)
    f0.Draw('same')
    f0.SetLineStyle(7)


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


if __name__ == '__main__':
    pass
