#!/usr/bin/env python
from root_draw import *
import periodictable as pt
from numpy import log, genfromtxt


Dir = dirname(realpath(__file__))


class Element(object):
    def __init__(self, el: pt.core.Element, density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(Dir, 'data', '{}.txt'.format(self.Name))
        self.A0, self.K, self.X0, self.X1, self.IE, self.C, self.D0 = genfromtxt(self.DataFile, skip_header=4, max_rows=1)
        self.IE *= 1e-6  # convert to MeV
        self.Density = choose(density, el.density)
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV

    def draw_data(self):
        g = Draw.make_tgrapherrors(*genfromtxt(self.DataFile, usecols=[1, 2], skip_header=10).T, markersize=.7)
        g.Draw('p')
        return g


class Particle(object):
    def __init__(self, name, mass, lifetime):
        self.Name = name
        self.M = mass / constants.e * constants.c**2 / 1e6 if mass < 1e-10 else mass
        self.Tau = lifetime

    def __repr__(self):
        return '{} particle with mass {:1.1f} MeV and lifetime {:1.1e} s'.format(self.Name, self.M, self.Tau)


K = 4 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4  # MeV * cm^2 / mol
Si = Element(pt.silicon)
Dia = Element(pt.carbon, density=3.52, name='Diamond')
Cu = Element(pt.copper)

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


def offset(p, part: Particle, el: Element):
    return K * el.Z / el.A / calc_speed(p, part.M) ** 2 * density_correction(p, part, el) / 2


def bethe_bloch(p, part: Particle, el: Element, dens_corr=True):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV * cm^2 / g]"""
    b = calc_speed(p, part.M)
    d = density_correction(p, part, el) if dens_corr else 0
    return K * el.Z / el.A / b ** 2 * (.5 * log(2 * M_E * beta_gamma(p, part.M) ** 2 * w_max(p, part.M) / el.IE ** 2) - b ** 2 - d / 2)


def draw_density_correction(part: Particle, el: Element, pmin=10, pmax=1e6):
    f = Draw.make_tf1('Density Correction for {}s in {}'.format(part.Name, el.Name), density_correction, pmin, pmax, part=part, el=el)
    Draw.histo(f, logx=True, logy=True)
    format_histo(f, x_tit='#beta#gamma', y_tit='Value', y_off=1.1, center_y=True, center_x=True)


def get_minimum(p: Particle = Pion, el: Element = Dia):
    pmin = Draw.make_tf1('', bethe_bloch, 100, 1e6, part=p, el=el).GetMinimumX()
    info('Minimum: {} MeV (betagamma = {}'.format(pmin, beta_gamma(pmin, p.M)))
    return pmin


def draw_bethe(part: Particle, el: Element, pmin=10, pmax=1e6, draw_data=False):
    f = Draw.make_tf1('Bethe-Bloch for {}s in {}'.format(part.Name, el.Name), bethe_bloch, pmin, pmax, part=part, el=el)
    f0 = Draw.make_tf1('', bethe_bloch, pmin, pmax, 4, part=part, el=el, dens_corr=False)
    Draw.histo(f, logx=True, logy=True, grid=True)
    leg = Draw.legend([f, f0], ['Bethe Bloch', 'without #delta'] + ['b'] if draw_data else [], 'l')
    format_histo(f, x_tit='{} Momentum [Mev/c]'.format(part.Name), y_tit='Mass Stopping Power [MEV cm^{2}/g]', y_off=1.3, center_x=True, center_y=True, y_range=[1, 2 * f.GetMaximum()], x_off=1.3)
    f0.Draw('same')
    f0.SetLineStyle(7)
    if draw_data and part.Name == 'Muon':
        g = el.draw_data()
        leg.AddEntry(g, 'TCut = 0.05 MeV', 'p')
    return f


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


if __name__ == '__main__':
    pass
