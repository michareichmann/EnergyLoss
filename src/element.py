from draw import choose, sqrt, Draw, join, dirname, realpath, print_table, beta_gamma
from numpy import genfromtxt, array
import periodictable as pt
from src.particle import Muon


class Element(object):
    def __init__(self, el: pt.core.Element, rad_length, e_eh=1., density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(dirname(dirname(realpath(__file__))), 'data', '{}.txt'.format(self.Name))
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


Si = Element(pt.silicon, 21.82, e_eh=3.68)
Dia = Element(pt.carbon, 42.70, e_eh=13.3, density=3.52, name='Diamond')
Cu = Element(pt.copper, 12.86)
Ar = Element(pt.argon, 19.55, density=1.662e-3)
Pb = Element(pt.lead, 6.37)
