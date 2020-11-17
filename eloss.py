#!/usr/bin/env python
from draw import *
import periodictable as pt
from numpy import log, genfromtxt, inf, diff
from typing import Any

Dir = dirname(realpath(__file__))


class Element(object):
    def __init__(self, el: pt.core.Element, e_eh=1., density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(Dir, 'data', '{}.txt'.format(self.Name))
        self.A0, self.K, self.X0, self.X1, self.IE, self.C, self.D0 = genfromtxt(self.DataFile, skip_header=4, max_rows=1)
        self.IE *= 1e-6  # convert to MeV
        self.Density = choose(density, el.density)
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV
        self.EEH = e_eh  # eV

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
        self.M = mass / constants.e * constants.c ** 2 / 1e6 if mass < 1e-10 else mass
        self.Tau = lifetime

    def __repr__(self):
        return '{} particle with mass {:1.1f} MeV and lifetime {:1.1e} s'.format(self.Name, self.M, self.Tau)


K = 4 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4  # MeV * cm^2 / mol
Si = Element(pt.silicon, e_eh=3.68)
Dia = Element(pt.carbon, e_eh=13.3, density=3.52, name='Diamond')
Cu = Element(pt.copper)
Ar = Element(pt.argon, density=1.662e-3)
Pb = Element(pt.lead)

Muon = Particle('Muon', constants.physical_constants['muon mass'][0], 2.1969e-6)
Pion = Particle('Pion', 139.57018, 2.6033e-8)
Electron = Particle('Electron', constants.electron_mass)
Positron = Particle('Positron', constants.electron_mass)
Proton = Particle('Proton', constants.proton_mass)


class Eloss(object):

    def __init__(self, part: Particle, el: Element, thickness=500, absolute=False):
        self.Linear = not absolute
        self.P = part
        self.El = el
        self.T = thickness * 1e-4  # um to cm
        self.X = self.T * self.El.Density
        self.C = 2 * constants.pi * constants.Avogadro * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e4 * self.El.Z / self.El.A  # MeV / cm
        self.F0 = None
        self.F = self.make_f()

        self.Draw = Draw(join(Dir, 'main.ini'))

    def __call__(self, p, part: Particle = None, el: Element = None):
        self.reload(part, el)
        return self.F(p / self.P.M)

    def reload(self, p=None, e=None):
        if p is not None or e is not None:
            self.P = choose(p, self.P)
            self.El = choose(e, self.El)
            self.F = self.make_f()

    def make_f(self):
        return TF1()

    def f(self, bg):
        pass

    def draw(self, color=2, line_style=1, xmin=.1, xmax=1e3, y_range=None):
        y_range = array(choose(y_range, [1, 10]))
        self.Draw(self.F, logx=True, grid=True, w=2, h=1.2, lm=.1, bm=.4, rm=None if self.Linear else .1)
        y_tit = 'Linear Stopping Power [MeV/cm]' if self.Linear else 'dE/dx in {:1.0f} #mum {} [keV]'.format(self.T * 1e4, self.El.Name)
        format_histo(self.F, line_style=line_style, color=color, x_range=[xmin, xmax], y_range=y_range, x_tit='#beta#gamma', y_tit=y_tit,
                     center_x=True, center_y=True, x_off=1.3, y_off=1, tit_size=.05, lab_size=.05)
        Draw.x_axis(y_range[0] - diff(y_range)[0] * .23 / .67, xmin, xmax, '{} Momentum [MeV/c]'.format(self.P.Name), array([xmin, xmax]) * self.P.M, center=True, log=True, off=1.3,
                    tit_size=.05, lab_size=.05)
        if not self.Linear:
            Draw.y_axis(xmax, *y_range, 'Induced eh-pairs #times 1000', y_range / self.El.EEH, center=True, off=1, tit_size=.05, lab_size=.05)

    def draw_same(self, color=4, line_style=1):
        format_histo(self.F, line_color=color, line_style=line_style)
        self.F.Draw('same')

    def density_correction(self, bg):
        x = log10(bg)
        el = self.El
        if x >= el.X1:
            return 2 * log(10) * x - el.C
        elif el.X0 <= x < el.X1:
            return 2 * log(10) * x - el.C + el.A0 * (el.X1 - x) ** el.K
        else:
            return el.D0 * 10 ** (2 * (x - el.X0))

    def get_minimum(self, p=None, el=None):
        self.reload(p, el)
        emin, bg = self.F.GetMinimum(.1, 1e3), self.F.GetMinimumX(.1, 1e3)
        info('Minimum: {:1.2f} Mev/cm at {:1.0f} MeV (betagamma = {:1.2f})'.format(emin, bg * self.P.M, bg))
        return emin, bg * self.P.M

    def get_w_max(self, bg):
        """returns maximum energy transfer in a single collsion with a free electron in [MeV]"""
        return 2 * M_E * bg ** 2 / (1 + 2 * bg / beta(bg) * M_E / self.P.M + (M_E / self.P.M) ** 2)

    def get_rel_to_mip(self, p):
        v = self(p) / self.get_minimum()[0]
        info('Relativ energy loss compared to a MIP {}: {:1.2f}'.format(self.get_str(), v))
        return v

    def get_str(self):
        return 'for {}s in {}'.format(self.P.Name, self.El.Name)


class LandauVavilovBichsel(Eloss):

    def __init__(self, part: Particle, el: Element, t=500, absolute=False):
        super().__init__(part, el, t, absolute)

    def make_f(self):
        f = Draw.make_tf1('Landau Vavilov Bichsel {}'.format(self.get_str()), self.f, .1, 1e6)
        f.SetNpx(1000)
        return f

    def f(self, bg):
        xi = self.get_xi(bg)
        lin_fac = 1 / self.T if self.Linear else 1000
        return xi * (log(2 * M_E * bg ** 2 / self.El.IE) + log(xi / self.El.IE) + .2 - calc_speed(bg * self.P.M, self.P.M) ** 2 - self.density_correction(bg)) * lin_fac

    def get_xi(self, bg):
        return self.C * self.X / calc_speed(bg * self.P.M, self.P.M) ** 2


class BetheBloch(Eloss):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV / cm]"""

    def __init__(self, part: Particle, el: Element, t=500, dens_corr: Any = True, wcut=None, absolute=False):
        self.DensityCorr = dens_corr
        self.WCut = wcut
        super().__init__(part, el, t, absolute)

    def get_fall_coeff(self, p=None, e=None, show=False):
        self.reload(p, e)
        x = linspace(.3, .8, 100)
        g = Draw.make_tgrapherrors(x, [self.f(ix) for ix in x * lorentz_factor(x)])
        self.Draw(g, show=show)
        fit = TF1('fit', '[0] * (x - [1])^[2]', .1, 1)
        g.Fit(fit, 'q')
        info('Fall coefficient for {}: {:1.2f} ({:1.2f})'.format(self.get_str(), fit.GetParameter(2), fit.GetParError(2)))
        return ufloat(fit.GetParameter(2), fit.GetParError(2))

    def make_f(self):
        f = Draw.make_tf1('Bethe Bloch {}'.format(self.get_str()), self.f, .1, 1e6)
        f.SetNpx(1000)
        return f

    def f(self, bg):
        b = beta(bg)
        d = self.density_correction(bg) if self.DensityCorr else 0
        w_max = self.get_w_max(bg)
        w_cut = w_max if self.WCut is None else min(self.WCut, w_max)
        lin_fac = 1 if self.Linear else self.T * 1000
        return self.C * self.El.Density / b ** 2 * (log(2 * M_E * bg ** 2 * w_cut / self.El.IE ** 2) - 2 * b * b * (1 + w_cut / w_max) - d + self.get_correction(bg)) * lin_fac

    def get_correction(self, bg):
        b, g = beta(bg), gamma(bg)
        if self.P.Name == 'Electron':
            return 1 + b * b - (2 * g - 1) / (g * g) * log(2) + ((g - 1) / g) ** 2 / 8
        elif self.P.Name == 'Positron':
            return log(2) + 2 * b * b - b * b / 12 * (23 + 14 / (g + 1) + 10 / (g + 1) ** 2 + 4 / (g + 1) ** 3)
        return 0

    def get_w_max(self, bg):
        return .5 * M_E * (gamma(bg) - 1) if self.P.Name == 'Electron' else M_E * (gamma(bg) - 1) if self.P.Name == 'Positron' else super(BetheBloch, self).get_w_max(bg)

    def delta_rays(self, wmin, p):
        b0 = calc_speed(p, self.P.M)
        wmax = self.get_w_max(beta_gamma(p, self.P.M))
        return self.C / b0 ** 2 * ((1 / wmin - 1 / wmax) - b0 ** 2 / wmax * log(wmax / wmin)) * self.X

    def draw_delta_rays(self, p, wmin=1e-3, y_range=None):
        f = Draw.make_tf1('Delta Rays for {}s in {:1.0f} #mum {}'.format(self.P.Name, self.T * 1e4, self.El.Name), self.delta_rays, wmin, self.get_w_max(p), p=p)
        Draw.histo(f, logx=True, logy=True, lm=.13, bm=.22)
        format_histo(f, x_tit='T_{min} [MeV]', y_tit='N_{#delta}', y_off=1.3, center_y=True, center_x=True, x_off=1.4, y_range=choose(y_range, [1e-5, 10]), tit_size=.05, lab_size=.045)


a = LandauVavilovBichsel(Muon, Si)
bb = BetheBloch(Pion, Dia)
draw = Draw(join(Dir, 'main.ini'))
d = BetheBloch(Positron, Pb)
e = BetheBloch(Electron, Pb)


def draw_dens_correction(p, el, y_range=None):
    f, f0 = BetheBloch(p, el), BetheBloch(p, el, dens_corr=False)
    f0.draw(color=draw.get_color(2, 1), line_style=7, y_range=y_range)
    f.draw_same(draw.get_color(2, 0))
    Draw.legend([f.F, f0.F], ['Bethe Bloch', 'without #delta'], 'l', x2=.96, w=.2, scale=1.4)


def draw_restricted(p, el, w_cut=2, y_range=None):
    b0 = BetheBloch(p, el)
    n = 2 + make_list(w_cut).size
    fs = [BetheBloch(p, el, dens_corr=d, wcut=w) for d, w in [(1, None), (0, None)] + [(1, iw) for iw in make_list(w_cut) * b0.get_minimum()[0]]]
    fs[1].draw(color=draw.get_color(n, n - 1), line_style=7, y_range=y_range)
    for i, f in enumerate(fs[2:], 1):
        f.draw_same(draw.get_color(n, i), 9)
    fs[0].draw_same(draw.get_color(n, 0))
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'without #delta'] + ['W_{{cut}}={}dE/dx_{{min}}'.format(i) for i in make_list(w_cut)], 'l', x2=.96, w=.23, scale=1.4)


def draw_bethe_mpv(p=Pion, el=Dia, t=500, w_cut=2, y_range=None):
    b0 = BetheBloch(p, el)
    fs = [BetheBloch(p, el, t, absolute=True), BetheBloch(p, el, t, absolute=True, wcut=w_cut * b0.get_minimum()[0]), LandauVavilovBichsel(p, el, t, True)]
    fs[1].draw(color=draw.get_color(3, 2), line_style=9, y_range=y_range)
    fs[0].draw_same(draw.get_color(3, 0))
    fs[2].draw_same(draw.get_color(5, 1), line_style=6)
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'W_{{cut}}={}dE/dx_{{min}}'.format(w_cut), 'Landau-Vavilov-Bichsel'], 'l', x2=.89, w=.33, scale=1.4)


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


if __name__ == '__main__':
    pass
