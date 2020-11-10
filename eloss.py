#!/usr/bin/env python
from draw import *
import periodictable as pt
from numpy import log, genfromtxt, inf, diff


Dir = dirname(realpath(__file__))
draw = Draw(join(Dir, 'main.ini'))


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

    def __repr__(self):
        header = ['Name', 'Z', 'A [g/mol]', 'ρ [g/cm3]', 'a', 'k', 'x0', 'x1', 'I [MeV]', 'C', 'δ0', 'EPlasma [MeV]']
        return print_table([[self.Name, self.Z, self.A, self.Density, self.A0, self.K, self.X0, self.X1, '{:.2e}'.format(self.IE), self.C, self.D0, '{:.2e}'.format(self.EPlasma)]], header, prnt=False)

    def draw_data(self):
        g = Draw.make_tgrapherrors(*genfromtxt(self.DataFile, usecols=[1, 2], skip_header=10).T, markersize=.7)
        g.Draw('p')
        return g


class Particle(object):
    def __init__(self, name, mass, lifetime=inf):
        self.Name = name
        self.M = mass / constants.e * constants.c**2 / 1e6 if mass < 1e-10 else mass
        self.Tau = lifetime

    def __repr__(self):
        return '{} particle with mass {:1.1f} MeV and lifetime {:1.1e} s'.format(self.Name, self.M, self.Tau)


K = 4 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4  # MeV * cm^2 / mol
Si = Element(pt.silicon)
Dia = Element(pt.carbon, density=3.52, name='Diamond')
Cu = Element(pt.copper)
Ar = Element(pt.argon, density=1.662e-3)

Muon = Particle('Muon', constants.physical_constants['muon mass'][0], 2.1969e-6)
Pion = Particle('Pion', 139.57018, 2.6033e-8)
Electron = Particle('Electron', constants.electron_mass)
Proton = Particle('Proton', constants.proton_mass)


def w_max(p, m):
    """returns maximum energy transfer in a single collsion with a free electron in [MeV]"""
    return 2 * M_E * beta_gamma(p, m) ** 2 / (1 + lorentz_factor(calc_speed(p, m)) / m + (M_E / m) ** 2)


def get_x(p, part: Particle):
    return log10(p / part.M)


def offset(p, part: Particle, el: Element):
    return K * el.Z / el.A / calc_speed(p, part.M) ** 2 * density_correction(p, part, el) / 2


def density_correction(p, part: Particle, el: Element):
    x = get_x(p, part)
    if x >= el.X1:
        return 2 * log(10) * x - el.C
    elif el.X0 <= x < el.X1:
        return 2 * log(10) * x - el.C + el.A0 * (el.X1 - x) ** el.K
    else:
        return el.D0 * 10 ** (2 * (x - el.X0))


def bethe_bloch(bg, part: Particle, el: Element, dens_corr=True):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV / cm]"""
    p = part.M * bg
    b = calc_speed(p, part.M)
    d = density_correction(p, part, el) if dens_corr else 0
    return K * el.Density * el.Z / el.A / b ** 2 * (.5 * log(2 * M_E * bg ** 2 * w_max(p, part.M) / el.IE ** 2) - b ** 2 - d / 2)


def delta_rays(wmin, dx, p, part: Particle, el: Element):
    b = calc_speed(p, part.M)
    wmax = w_max(p, part.M)
    return .5 * K * el.Z / el.A / b ** 2 * ((1 / wmin - 1 / wmax) - b ** 2 / wmax * log(wmax / wmin)) * dx * el.Density


def get_minimum(p: Particle = Pion, el: Element = Dia):
    f = Draw.make_tf1('', bethe_bloch, 100, 1e6, part=p, el=el)
    emin, pmin = f.GetMinimum(), f.GetMinimumX()
    info('Minimum: {:1.1f} Mev cm2/g at {:1.1f} MeV (betagamma = {:1.2f})'.format(emin, pmin, beta_gamma(pmin, p.M)))
    return emin, pmin


def get_str(p, el):
    return 'for {}s in {}'.format(p.Name, el.Name)


def get_rel_to_mip(p, part: Particle = Pion, el: Element = Dia):
    v = bethe_bloch(p, part, el) / get_minimum(part, el)[0]
    info('Relativ energy loss compared to a MIP {}: {:1.2f}'.format(get_str(part, el), v))
    return v


def get_fall_coeff(p: Particle = Pion, el: Element = Dia):
    x = arange(10, 100)
    g = Draw.make_tgrapherrors(x, [bethe_bloch(ix, p, el) for ix in x])
    fit = TF1('fit', '[0] * (x - [1])^[2]', 10, 100)
    g.Fit(fit, 'q0')
    info('Fall coefficient for {}s in {}: {:1.2f} ({:1.2f})'.format(p.Name, el.Name, fit.GetParameter(2), fit.GetParError(2)))


def draw_density_correction(part: Particle, el: Element, pmin=10, pmax=1e6):
    f = Draw.make_tf1('Density Correction for {}s in {}'.format(part.Name, el.Name), density_correction, pmin, pmax, part=part, el=el)
    Draw.histo(f, logx=True, logy=True)
    format_histo(f, x_tit='#beta#gamma', y_tit='Value', y_off=1.1, center_y=True, center_x=True)


def draw_bethe(part: Particle, el: Element, bg_min=1, bg_max=1e3, draw_data=False, y_range=None):
    f = Draw.make_tf1('', bethe_bloch, bg_min, bg_max, part=part, el=el)
    f0 = Draw.make_tf1('Bethe-Bloch for {}s in {}'.format(part.Name, el.Name), bethe_bloch, bg_min, bg_max, 4, part=part, el=el, dens_corr=False)
    draw.histo(f0, logx=True, grid=True, w=2, h=1.2, lm=.07, bm=.4)
    leg = Draw.legend([f, f0], ['Bethe Bloch', 'without #delta'] + (['b'] if draw_data else []), 'l', x2=.96, w=.2, scale=1.7)
    y_range = choose(y_range, [1, 10])
    format_histo(f0, x_tit='#beta#gamma'.format(part.Name), y_tit='Linear Stopping Power [MeV/cm]', center_x=True, center_y=True, y_range=y_range, x_off=1.3, y_off=.65, line_style=7,
                 line_color=draw.get_color(2, 1), tit_size=.05, lab_size=.05)
    format_histo(f, line_color=draw.get_color(2, 0))
    f.Draw('same')
    Draw.x_axis(y_range[0] - diff(y_range)[0] * .23 / .67, bg_min, bg_max, '{} Momentum [MeV/c]'.format(part.Name), array([bg_min, bg_max]) * part.M, center=True, log=True, off=1.3, tit_size=.05,
                lab_size=.05)
    if draw_data and part.Name == 'Muon':
        g = el.draw_data()
        leg.AddEntry(g, 'TCut = 0.05 MeV', 'p')
    return f


def draw_drays(p, dx, part: Particle, el: Element, wmin=1e-3, y_range=[1e-5, 10]):
    f = Draw.make_tf1('Delta Rays for {}s in {} cm {}'.format(part.Name, dx, el.Name), delta_rays, wmin, w_max(p, part.M), dx=dx, p=p, part=part, el=el)
    Draw.histo(f, logx=True, logy=True, lm=.13, bm=.22)
    format_histo(f, x_tit='T_{min} [MeV]', y_tit='N_{#delta}', y_off=1.3, center_y=True, center_x=True, x_off=1.4, y_range=y_range, tit_size=.05, lab_size=.045)


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


if __name__ == '__main__':
    pass
