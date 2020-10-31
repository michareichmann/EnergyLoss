#!/usr/bin/env python
from root_draw import *
import periodictable as pt
from numpy import log


class Element(object):
    def __init__(self, el: pt.core.Element, excitation_energy, density=None):
        self.Name = el.name.title()
        self.Z = el.number
        self.A = el.mass
        self.IE = excitation_energy * 1e-6
        self.Density = choose(density, el.density)


class Particle(object):
    def __init__(self, name, mass, lifetime):
        self.Name = name
        self.M = mass / constants.e * constants.c**2 / 1e6
        self.Tau = lifetime


K = 4 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4  # MeV * cm^2 / mol
Si = Element(pt.silicon, 173)
Dia = Element(pt.carbon, 78, density=3.5)
Cu = Element(pt.copper, 322)

Muon = Particle('Muon', constants.physical_constants['muon mass'][0], 2.1969e-6)


def w_max(p, m):
    """returns maximum energy transfer in a single collsion with a free electron in [MeV]"""
    return 2 * M_E * beta_gamma(p, m) ** 2 / (1 + lorentz_factor(calc_speed(p, m)) / m + (M_E / m) ** 2)


def plasma_energy(el: Element):
    return sqrt(el.Density * el.Z / el.A) * 28.816 * 1e-6


def density_correction(p, part: Particle, el: Element):
    return log(plasma_energy(el) / el.IE) + log(beta_gamma(p, part.M)) - .5


def bethe_bloch(p, part: Particle, el: Element, dens_corr=True):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV * cm^2 / g]"""
    b = calc_speed(p, part.M)
    d = density_correction(p, part, el) if dens_corr else 0
    return K * el.Z / el.A / b ** 2 * (.5 * log(2 * M_E * b ** 2 * lorentz_factor(b) * w_max(p, part.M) / el.IE ** 2) - b ** 2 - d)


def draw_density_correction(part: Particle, el: Element, pmin=10, pmax=1e6):
    def f(p, pars):
        _ = pars
        return density_correction(p[0], part, el)
    g = TF1('Density Correction for {}s in {}'.format(part.Name, el.Name), f, pmin, pmax)
    Draw.add(f)
    Draw.histo(g, logx=True, logy=True)


def draw_bethe(part: Particle, el: Element, pmin=10, pmax=1e6):
    def f(p, pars, dens_corr):
        _ = pars
        return bethe_bloch(p[0], part, el, dens_corr)
    f0, f1 = partial(f, dens_corr=True), partial(f, dens_corr=False)
    g = TF1('Bethe-Bloch for {}s in {}'.format(part.Name, el.Name), f0, pmin, pmax)
    g1 = TF1('Corr', f1, pmin, pmax)
    Draw.add(f0, f1, g1)
    Draw.histo(g, logx=True, logy=True)
    format_histo(g, x_tit='{} Momentum [Mev/c]'.format(part.Name), y_tit='Mass Stopping Power', y_off=1.3, center_x=True, center_y=True, y_range=[1, 2 * g.GetMaximum()], x_off=1.3)
    g1.Draw('same')
    g1.SetLineStyle(7)


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


if __name__ == '__main__':

    pass