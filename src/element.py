from draw import choose, sqrt, Draw, join, dirname, realpath, print_table, beta_gamma
from numpy import genfromtxt, array
import periodictable as pt
from src.particle import Muon


class Element(object):
    def __init__(self, el: pt.core.Element, e_eh=1., density=None, name=None):
        self.Name = choose(name, el.name.title())
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(dirname(dirname(realpath(__file__))), 'data', '{}.txt'.format(self.Name))
        self.A0, self.K, self.X0, self.X1, self.IE, self.C, self.D0 = genfromtxt(self.DataFile, skip_header=4, max_rows=1)
        self.IE *= 1e-6  # convert to MeV
        self.Density = choose(density, el.density)
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV
        self.EEH = e_eh  # eV

    def __repr__(self):
        header = ['Name', 'Z', 'A [g/mol]', 'ρ [g/cm3]', 'a', 'k', 'x0', 'x1', 'I [MeV]', 'C', 'δ0', 'EPlasma [MeV]']
        return print_table([[self.Name, self.Z, self.A, self.Density, self.A0, self.K, self.X0, self.X1, '{:.2e}'.format(self.IE), self.C, self.D0, '{:.2e}'.format(self.EPlasma)]], header, prnt=False)

    def get_data(self, linear=True, mass=False, t=500):
        x, y = genfromtxt(self.DataFile, usecols=[1, 2], skip_header=10).T
        return array([beta_gamma(x, Muon.M), y * (1 if mass else self.Density if linear else t * self.Density)])

    def draw_data(self, linear=True, mass=False, t=500):
        Draw.make_tgrapherrors(*self.get_data(linear, mass, t), markersize=.7).Draw('p')


Si = Element(pt.silicon, e_eh=3.68)
Dia = Element(pt.carbon, e_eh=13.3, density=3.52, name='Diamond')
Cu = Element(pt.copper)
Ar = Element(pt.argon, density=1.662e-3)
Pb = Element(pt.lead)
