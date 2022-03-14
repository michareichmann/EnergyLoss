from numpy import log, tan, sum, polyfit, delete, polyval
from numpy.random import normal

from helpers.utils import calc_speed, bg2b
from plotting.draw import *
from src.element import *
from src.particle import *


class Scattering:

    DiaPad = Dia.rad_length(500) + 2 * Au.rad_length(.2) + 2 * Cr.rad_length(.05)
    SiPlane = Si.rad_length(285 + 500) + PCB.rad_length(700)
    S = 2.032  # cm

    def __init__(self, part: Particle, el: Element = None, t=500):
        self.P = part
        self.E = el
        self.T = t
        self.F = Draw.make_tf1('scat', self.f, .1, 1e6, title=str(self), npx=1000)

        self.Draw = Draw(join(Dir, 'main.ini'))

    def __call__(self, p, x=None):
        return self.f(p / self.P.M, x)

    def __str__(self):
        return f'Multiple Scattering for {self.P.Name}s' + (' in {self.T}μm {self.E.Name}' if self.E else '')

    def __repr__(self):
        return self.__str__()

    def f(self, bg, x=None):
        p = self.P.M * bg
        xfac = self.E.rad_length(self.T) if x is None else x
        return 1e3 * 13.6 / (p * calc_speed(p, self.P.M)) * sqrt(xfac) * (1 + .0038 * log(xfac / bg2b(bg) ** 2))  # mrad

    def draw(self, ymax=None, xmin=10, xmax=1e4, **dkw):
        ymax = choose(ymax, min(5000, 1.1 * self.F(xmin)))
        self.Draw(self.F, y_range=[0, ymax], x_range=[xmin, xmax], **prep_kw(dkw, y_tit='#sigma_{#theta} [mrad]', **Draw.mode(4)))
        Draw.x_axis(-ymax * .2 / .64, xmin, xmax, f'{self.P.Name} Momentum [GeV/c]', array([xmin, xmax]) * self.P.M / 1000, center=True, log=True, off=1.3, tit_size=.05, lab_size=.05)
        return self

    def simulate_residual(self, pl=0, p=260, x=None, n=1e6, unbias=False):
        x = array(choose(x, [0, self.S, 5.8, 7.4, 6 * self.S, 7 * self.S])) * 1e4  # cm to um
        s = array([self(p, x0) for x0 in [self.SiPlane, self.SiPlane, self.DiaPad, self.DiaPad, self.SiPlane]]) * 1e-3  # mrad to rad
        y0 = [tan(normal(0, i_s, size=int(n))) * i_x for i_s, i_x in zip(s, diff(x))]
        y = array([zeros(int(n)), y0[0], sum(y0[:4], axis=0), sum(y0, axis=0)])
        x = x[[0, 1, -2, -1]]
        fits = polyfit(delete(x, pl), delete(y, pl, axis=0), deg=1) if unbias else polyfit(x, y, deg=1)
        return polyval(fits, x[pl]) - y[pl]

    def draw_residuals(self, pl=0, p=260, x=None, n=1e6, unbias=False, **dkw):
        self.Draw.distribution(self.simulate_residual(pl, p, x, n, unbias), **prep_kw(dkw, x_tit='Residual [#mum]'))

    def print_residuals(self, p=260):
        for i in range(4):
            print(f'Residual plane {i}: {mean_sigma(self.simulate_residual(i, p))[1]: 5.1f} μm')
