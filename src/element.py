from helpers.draw import choose, sqrt, Draw, join, dirname, realpath, print_table, beta_gamma, M_E, e2p
from numpy import genfromtxt, array
import periodictable as pt
from src.particle import Muon


class Element(object):
    def __init__(self, el: pt.core.Element, rad_length, e_eh=1., density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(dirname(dirname(realpath(__file__))), 'data', 'muons', '{}.txt'.format(self.Name))
        self.EDataFile = join(dirname(dirname(realpath(__file__))), 'data', 'electrons', '{}.txt'.format(self.Name))
        self.X0 = rad_length
        self.a, self.k, self.x0, self.x1, self.IE, self.C, self.D0 = genfromtxt(self.DataFile, skip_header=4, max_rows=1)
        self.IE *= 1e-6  # convert to MeV
        self.Density = choose(density, el.density)
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV
        self.EEH = e_eh  # eV

    def __repr__(self):
        header = ['Name', 'Z', 'A [g/mol]', 'ρ [g/cm3]', 'a', 'k', 'x0', 'x1', 'I [MeV]', 'C', 'δ0', 'EPlasma [MeV]']
        return print_table([[self.Name, self.Z, self.A, self.Density, self.a, self.k, self.x0, self.x1, '{:.2e}'.format(self.IE), self.C, self.D0, '{:.2e}'.format(self.EPlasma)]], header, prnt=False)

    def get_data(self, col, linear, mass, t):
        x, y = genfromtxt(self.DataFile, usecols=[0, col], skip_header=10).T
        return array([beta_gamma(x, Muon.M), y * (1 if mass else self.Density if linear else t * self.Density)])

    def get_e_data(self, col, linear, mass, t):
        x, y = genfromtxt(self.EDataFile, usecols=[0, col], skip_header=8).T
        return array([beta_gamma(e2p(x, M_E), M_E), y * (1 if mass else self.Density if linear else t * self.Density)])

    def get_ionisation(self, linear=True, mass=False, t=500):
        return self.get_data(2, linear, mass, t)

    def get_radlos(self, linear=True, mass=False, t=500):
        return self.get_data(6, linear, mass, t)

    def get_brems(self, linear=True, mass=False, t=500):
        return self.get_data(3, linear, mass, t)

    def draw_ionisation(self, linear=True, mass=False, t=500):
        Draw.make_tgrapherrors(*self.get_ionisation(linear, mass, t), markersize=.7).Draw('p')

    def draw_radloss(self, linear=True, mass=False, t=500):
        Draw.make_tgrapherrors(*self.get_radlos(linear, mass, t), markersize=.7).Draw('p')

    def draw_full_radloss(self, n=-1):
        d = Draw()
        x, y = self.get_radlos(mass=True)
        d.graph(x[:n], y[:n], logx=True)

    def draw_full_brems(self, n=-1):
        d = Draw()
        x, y = self.get_brems(mass=True)
        d.graph(x[:n], y[:n], logx=True)

    def draw_brems(self, linear=True, mass=False, t=500, color=1, style=1):
        g = Draw.make_tgrapherrors(*self.get_e_data(2, linear, mass, t), markersize=.7, color=color, line_style=style, lw=2)
        g.Draw('l')
        fit = Draw.make_f('exact', 'pol1', 1, 1e3)
        g.Fit(fit, 'qs0', '', 50, 1e3)
        p1 = fit(50) / 50  # force the function to go to zero below 100 MeV
        return Draw.make_tf1('exact', lambda x: x * p1 if x < 50 else fit(x), 1, 1e3, color=color, style=style, w=2)


Si = Element(pt.silicon, 21.82, e_eh=3.68)
Dia = Element(pt.carbon, 42.70, e_eh=13.3, density=3.52, name='Diamond')
Cu = Element(pt.copper, 12.86)
Ar = Element(pt.argon, 19.55, density=1.662e-3)
Pb = Element(pt.lead, 6.37)
