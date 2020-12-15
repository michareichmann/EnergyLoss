from helpers.draw import choose, sqrt, Draw, join, dirname, realpath, print_table, beta_gamma, M_E, e2p, file_exists, constants, discrete_int, interpolate_y, kramers_kronig, ensure_dir
from numpy import genfromtxt, array, pi, concatenate, savetxt, zeros, ones
import periodictable as pt
from src.particle import Muon

hbar = constants.physical_constants['Planck constant over 2 pi in eV s'][0]
Dir = dirname(dirname(realpath(__file__)))


class Element(object):

    def __init__(self, el: pt.core.Element, rad_length, e_eh=1., density=None, name=None, symbol=None):
        self.Name = choose(name, el.name.title())
        self.Symbol = choose(symbol, el.symbol)
        self.Z = el.number
        self.A = el.mass
        self.DataFile = join(Dir, 'data', 'muons', '{}.txt'.format(self.Name))
        self.EDataFile = join(Dir, 'data', 'electrons', '{}.txt'.format(self.Name))
        self.X0 = rad_length
        self.a, self.k, self.x0, self.x1, self.IE, self.C, self.D0 = genfromtxt(self.DataFile, skip_header=4, max_rows=1)
        self.IE *= 1e-6  # convert to MeV
        self.Density = choose(density, el.density)
        self.EPlasma = sqrt(self.Density * self.Z / self.A) * 28.816 * 1e-6  # MeV
        self.EEH = e_eh  # eV
        self.Draw = Draw(join(Dir, 'main.ini'))

        # STRAGGLING
        self.NE = constants.physical_constants['Avogadro constant'][0] * self.Density * self.Z / self.A
        self.N = constants.physical_constants['Avogadro constant'][0] * self.Density / self.A
        self.E, self.E1, self.E2, self.PIC = self.get_photo_data()

    def __repr__(self):
        header = ['Name', 'Z', 'A [g/mol]', 'ρ [g/cm3]', 'a', 'k', 'x0', 'x1', 'I [MeV]', 'C', 'δ0', 'EPlasma [MeV]']
        return print_table([[self.Name, self.Z, self.A, self.Density, self.a, self.k, self.x0, self.x1, '{:.2e}'.format(self.IE), self.C, self.D0, '{:.2e}'.format(self.EPlasma)]], header, prnt=False)

    # -------------------------------------
    # region GET
    def get_data(self, col, linear, mass, t):
        x, y = genfromtxt(self.DataFile, usecols=[0, col], skip_header=10).T
        return array([beta_gamma(x, Muon.M), y * (1 if mass else self.Density if linear else t * self.Density)])

    def get_e_data(self, col, linear, mass, t):
        x, y = genfromtxt(self.EDataFile, usecols=[0, col], skip_header=8).T
        return array([beta_gamma(e2p(x, M_E), M_E), y * (1 if mass else self.Density if linear else t * self.Density)])

    def get_refraction_data(self):
        f = join(Dir, 'data', 'photo', '{}.txt'.format(self.Name))
        if file_exists(f):
            e, lam, n, k = genfromtxt(f).T
            return array([e, n ** 2 - k ** 2, 2 * n * k]).T

    def get_xray_data(self):
        f = join(Dir, 'data', 'xray', '{}.txt'.format(self.Name))
        if file_exists(f):
            e, mu, mu0 = genfromtxt(f).T
            e *= 1e6  # [MeV] -> [eV]
            return array([e, ones(e.size), mu * hbar * constants.speed_of_light * self.Density * 100 / e]).T

    def get_photo_data(self):
        file_name = join(Dir, 'data', 'dielectric', '{}.txt'.format(self.Name))
        if not file_exists(file_name):
            x_ray_data, refr_data = self.get_refraction_data(), self.get_xray_data()
            if x_ray_data is None or refr_data is None:
                return zeros((4, 1))
            data = concatenate([x_ray_data, refr_data])
            e, e1, e2 = data[data[:, 0].argsort()].T
            e_a = constants.speed_of_light * hbar * sqrt(4 * pi * constants.physical_constants['classical electron radius'][0] * self.NE * 1e6)  # [eV]
            # e2 /= discrete_int(e, e * e2) * 2 / pi / e_a ** 2  # sum rule: int(E*e2 dE) = pi * e_a^2 / 2
            # e1 = kramers_kronig(e, e2)
            data = array([e, e1, e2, e2 * e / ((e1 ** 2 + e2 ** 2) * self.N * constants.speed_of_light * hbar) * 1e16]).T
            # data = array([e, e1, e2, e2 * e / (sqrt(abs(e1)) * self.N * constants.speed_of_light * hbar) * 1e16]).T
            ensure_dir(join(Dir, 'data', 'dielectric'))
            savetxt(join(Dir, 'data', 'dielectric', '{}.txt'.format(self.Name)), data, ['%.3e', '%.7e', '%.7e', '%.7e'], header='E [ev]  ε1            ε2            σ [Mb]')
        return genfromtxt(file_name).T

    def get_ionisation(self, linear=True, mass=False, t=500):
        return self.get_data(2, linear, mass, t)

    def get_radlos(self, linear=True, mass=False, t=500):
        return self.get_data(6, linear, mass, t)

    def get_brems(self, linear=True, mass=False, t=500):
        return self.get_data(3, linear, mass, t)

    def e1(self, e):
        i = next(i for i, ie in enumerate(self.E) if e < ie) - 1
        return interpolate_y(*self.E[i:i + 2], *self.E1[i:i + 2], e)

    def e2(self, e):
        i = next(i for i, ie in enumerate(self.E) if e < ie) - 1
        return interpolate_y(*self.E[i:i + 2], *self.E2[i:i + 2], e)
    # endregion GET
    # -------------------------------------

    # -------------------------------------
    # region DRAW
    def draw_ionisation(self, linear=True, mass=False, t=500):
        Draw.make_tgrapherrors(*self.get_ionisation(linear, mass, t), markersize=.7).Draw('p')

    def draw_radloss(self, linear=True, mass=False, t=500):
        Draw.make_tgrapherrors(*self.get_radlos(linear, mass, t), markersize=.7).Draw('p')

    def draw_full_radloss(self, n=-1):
        x, y = self.get_radlos(mass=True)
        self.Draw.graph(x[:n], y[:n], logx=True)

    def draw_full_brems(self, n=-1):
        x, y = self.get_brems(mass=True)
        self.Draw.graph(x[:n], y[:n], logx=True)

    def draw_brems(self, linear=True, mass=False, t=500, color=1, style=1):
        g = Draw.make_tgrapherrors(*self.get_e_data(2, linear, mass, t), markersize=.7, color=color, line_style=style, lw=2)
        g.Draw('l')
        fit = Draw.make_f('exact', 'pol1', 1, 1e3)
        g.Fit(fit, 'qs0', '', 50, 1e3)
        p1 = fit(50) / 50  # force the function to go to zero below 100 MeV
        return Draw.make_tf1('exact', lambda x: x * p1 if x < 50 else fit(x), 1, 1e3, color=color, style=style, w=2)

    def draw_photo_ionisation(self):
        self.Draw.graph(self.E, self.PIC, 'Photoionisation Cross Section', x_tit='Photon Energy [eV]', y_tit='#sigma_{#gamma} [Mb]', draw_opt='al', logx=True, logy=True)
    # endregion DRAW
    # -------------------------------------


Si = Element(pt.silicon, 21.82, e_eh=3.68)
Dia = Element(pt.carbon, 42.70, e_eh=13.3, density=3.52, name='Diamond', symbol='Dia')
Cu = Element(pt.copper, 12.86)
Ar = Element(pt.argon, 19.55, density=1.662e-3)
Pb = Element(pt.lead, 6.37)
