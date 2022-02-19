from helpers.draw import *
from numpy import log, diff
from typing import Any
from src.element import *
from src.particle import *


class Eloss(object):

    def __init__(self, part: Particle, el: Element, thickness=500, abst=False, mass=False):
        self.Linear = not abst
        self.Mass = mass  # mass stopping power
        self.P = part
        self.El = el
        self.T = thickness * 1e-4  # um to cm
        self.X = self.T * self.El.Density
        self.C = K * self.El.Z / self.El.A  # [MeV/cm]
        self.F0 = None
        self.F = self.make_f()

        self.Draw = Draw(join(Dir, 'main.ini'))

    def __call__(self, p, part: Particle = None, el: Element = None):
        self.reload(part, el)
        return self.F(p / self.P.M)

    def __add__(self, other):
        tmp = deepcopy(self)
        tmp.F = Draw.make_tf1('sum', lambda x: self.f(x) + (other.f(x) if isinstance(other, Eloss) else other(x)), .1, 1e6)
        return tmp

    def reload(self, p=None, el=None):
        if p is not None or el is not None:
            self.P = choose(p, self.P)
            self.El = choose(el, self.El)
            self.F = self.make_f()

    def make_f(self):
        return TF1()

    def f(self, bg):
        return 0

    def get_y_tit(self):
        return 'Mass Stopping Power [MeV cm^{2}/g]' if self.Mass else 'Linear Stopping Power [MeV/cm]' if self.Linear else 'dE/dx in {:1.0f} #mum {} [keV]'.format(self.T * 1e4, self.El.Name)

    def draw(self, color=2, line_style=1, xmin=.1, xmax=1e3, y_range=None, logy=False, draw_ekin=False):
        y_range = array(choose(y_range, ax_range(self.get_emin(), self.f(.2), .05, 0)))
        self.Draw(self.F, logx=True, grid=True, w=2, h=1.2, lm=.1, bm=.4, rm=None if self.Linear else .1, logy=logy)
        format_histo(self.F, line_style=line_style, color=color, x_range=[xmin, xmax], y_range=y_range, x_tit='#beta#gamma', y_tit=self.get_y_tit(),
                     center_x=True, center_y=True, x_off=1.3, y_off=1, tit_size=.05, lab_size=.05)
        tit = '{} {} [MeV{}]'.format(self.P.Name, *['Kinetic Energy', ''] if draw_ekin else ['Momentum', '/c'])
        x_vals = p2e(array([xmin, xmax]) * self.P.M, self.P.M) if draw_ekin else array([xmin, xmax]) * self.P.M
        self.F.SetMinimum(y_range[0])
        y_pos = y_range[0] * (y_range[0] / y_range[1]) ** (.2 / .64) if logy else y_range[0] - diff(y_range)[0] * .2 / .64
        Draw.x_axis(y_pos, xmin, xmax, tit, x_vals, center=True, log=True, off=1.3, tit_size=.05, lab_size=.05)
        if not self.Linear:
            Draw.y_axis(xmax, *y_range, 'Induced eh-pairs #times 1000', y_range / self.El.EEH, center=True, off=1, tit_size=.05, lab_size=.05, log=logy)
        return self

    def draw_same(self, color=4, line_style=1):
        format_histo(self.F, line_color=color, line_style=line_style)
        self.F.Draw('same')
        return self

    def density_correction(self, bg):
        x = log10(bg)
        el = self.El
        if x >= el.x1:
            return 2 * log(10) * x - el.C
        elif el.x0 <= x < el.x1:
            return 2 * log(10) * x - el.C + el.a * (el.x1 - x) ** el.k
        else:
            return el.D0 * 10 ** (2 * (x - el.x0))

    def e_correction(self, bg):
        b, g = bg2b(bg), bg2g(bg)
        if self.P.Name == 'Electron':
            return 1 + b * b - (2 * g - 1) / (g * g) * log(2) + ((g - 1) / g) ** 2 / 8
        elif self.P.Name == 'Positron':
            return log(2) + 2 * b * b - b * b / 12 * (23 + 14 / (g + 1) + 10 / (g + 1) ** 2 + 4 / (g + 1) ** 3)
        return 0

    def get_minimum(self, p=None, el=None, prnt=True):
        self.reload(p, el)
        emin, bg = self.F.GetMinimum(.1, 1e3), self.F.GetMinimumX(.1, 1e3)
        info('Minimum: {:1.2f} Mev/cm at {:1.0f} MeV (betagamma = {:1.2f})'.format(emin, bg * self.P.M, bg), prnt=prnt)
        return emin, bg * self.P.M

    def get_emin(self, prnt=False):
        return self.get_minimum(prnt=prnt)[0]

    @property
    def pmin(self):
        return self.get_minimum(prnt=False)[1]

    def eh_pairs(self, p):
        return self.F(p / self.P.M) / self.El.EEH

    def get_w_max(self, bg):
        """returns maximum energy transfer in a single collsion with a free electron in [MeV]"""
        return 2 * M_E * bg ** 2 / (1 + 2 * bg / bg2b(bg) * M_E / self.P.M + (M_E / self.P.M) ** 2)

    def get_lin_fac(self):
        return 1 if self.Mass else self.El.Density if self.Linear else self.X * 1000

    def get_rel_to_mip(self, p):
        v = self(p) / self.get_minimum()[0]
        info('Relativ energy loss compared to a MIP {}: {:1.2f}'.format(self.get_str(), v))
        return v

    def get_str(self):
        return 'for {}s in {}'.format(self.P.Name, self.El.Name)


class LandauVavilovBichsel(Eloss):

    def __init__(self, part: Particle, el: Element, t=500, abst=False, mass=False):
        super().__init__(part, el, t, abst, mass)

    def make_f(self):
        f = Draw.make_tf1('Landau Vavilov Bichsel {}'.format(self.get_str()), self.f, .1, 1e6)
        f.SetNpx(1000)
        return f

    def f(self, bg):
        xi = self.get_xi(bg)
        fac = 1 / self.X if self.Mass else 1 / self.T if self.Linear else 1000
        return xi * (log(2 * M_E * bg ** 2 / self.El.IE) + log(xi / self.El.IE) + .2 - calc_speed(bg * self.P.M, self.P.M) ** 2 - self.density_correction(bg) + self.e_correction(bg)) * fac

    def get_xi(self, bg):
        return self.C * self.X / calc_speed(bg * self.P.M, self.P.M) ** 2

    def get_kappa(self, bg):
        return self.get_xi(bg) / self.get_w_max(bg)


class BetheBloch(Eloss):
    """returns the mean energy loss of particle of momentum p and mass m in el [MeV / cm]"""

    def __init__(self, part: Particle, el: Element, t=500, dens_corr: Any = True, wcut=None, abst=False, mass=False):
        self.DensityCorr = dens_corr
        self.WCut = wcut
        super().__init__(part, el, t, abst, mass)

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
        f = Draw.make_tf1('bethe{}'.format(self.P.Name[:2]), self.f, .1, 1e6, title='Bethe Bloch {}'.format(self.get_str()))
        f.SetNpx(1000)
        return f

    def f(self, bg):
        b = bg2b(bg)
        d = self.density_correction(bg) if self.DensityCorr else 0
        w_max = self.get_w_max(bg)
        w_cut = w_max if self.WCut is None else min(self.WCut, w_max)
        return self.get_lin_fac() * self.C / (b * b) * (log(2 * M_E * bg * bg * w_cut / (self.El.IE * self.El.IE)) - b * b * (1 + w_cut / w_max) - d + self.e_correction(bg))

    def get_w_max(self, bg):
        return .5 * M_E * (bg2g(bg) - 1) if self.P.Name == 'Electron' else M_E * (bg2g(bg) - 1) if self.P.Name == 'Positron' else super(BetheBloch, self).get_w_max(bg)

    def delta_rays(self, wmin, p):
        b0 = calc_speed(p, self.P.M)
        wmax = self.get_w_max(pm2bg(p, self.P.M))
        return self.C / b0 ** 2 * ((1 / wmin - 1 / wmax) - b0 ** 2 / wmax * log(wmax / wmin)) * self.X

    def draw_delta_rays(self, p, wmin=1e-3, y_range=None):
        f = Draw.make_tf1('Delta Rays for {}s in {:1.0f} #mum {}'.format(self.P.Name, self.T * 1e4, self.El.Name), self.delta_rays, wmin, self.get_w_max(p), p=p)
        Draw.histo(f, logx=True, logy=True, lm=.13, bm=.22)
        format_histo(f, x_tit='T_{min} [MeV]', y_tit='N_{#delta}', y_off=1.3, center_y=True, center_x=True, x_off=1.4, y_range=choose(y_range, [1e-5, 10]), tit_size=.05, lab_size=.045)


class Bremsstrahlung(Eloss):

    def __init__(self, part: Particle, el: Element, t=500, abst=False, mass=False):
        super().__init__(part, el, t, abst, mass)
        self.N = constants.physical_constants['Avogadro constant'][0] * self.El.Density / self.El.A
        self.F0 = self.N * 1e4 * self.El.Z ** 2 * constants.physical_constants['classical electron radius'][0] ** 2 * constants.alpha / self.El.Density * self.get_lin_fac()

    def get_minimum(self, **kw):
        return 1, 1

    def make_f(self):
        f = Draw.make_tf1('brems', self.f, .1, 1e6, title='Bremsstrahlung {}'.format(self.get_str()))
        f.SetNpx(1000)
        return f

    def low(self, e):
        return self.F0 * e * (4 * log(2 * e / self.P.M) - 4/3)

    def high(self, e):
        return self.F0 * e * (4 * log(183 / self.El.Z ** (1/3)) + 2/9)

    def draw_exact(self):
        pass

    def f(self, bg):
        return self.get_lin_fac() / self.El.X0 * p2e(bg * self.P.M, self.P.M)
