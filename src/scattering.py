from helpers.draw import *
from numpy import log
from src.element import *
from src.particle import *


class Scattering:

    def __init__(self, part: Particle, el: Element, t=500):
        self.P = part
        self.E = el
        self.T = t
        self.F = Draw.make_tf1('scat', self.f, .1, 1e6, title=str(self), npx=1000)

        self.Draw = Draw(join(Dir, 'main.ini'))

    def __call__(self, p):
        return self.f(p / self.P.M)

    def __str__(self):
        return f'Multiple Scattering for {self.P.Name}s in {self.T}μm {self.E.Name}'

    def __repr__(self):
        return self.__str__()

    def f(self, bg):
        p = self.P.M * bg
        xfac = self.T * 1e-4 * self.E.Density / self.E.X0  # um -> cn
        return 1000 * 13.6 / (p * calc_speed(p, self.P.M)) * sqrt(xfac) * (1 + .0038 * log(xfac / bg2b(bg) ** 2))  # mrad

    def draw(self, ymax=None, xmin=10, xmax=1e4, **dkw):
        ymax = choose(ymax, min(5000, 1.1 * self.F(xmin)))
        self.Draw(self.F, y_range=[0, ymax], x_range=[xmin, xmax], **prep_kw(dkw, y_tit='#sigma_{#theta} [mrad]', **Draw.mode(4)))
        Draw.x_axis(-ymax * .2 / .64, xmin, xmax, f'{self.P.Name} Momentum [GeV/c]', array([xmin, xmax]) * self.P.M / 1000, center=True, log=True, off=1.3, tit_size=.05, lab_size=.05)
        return self
