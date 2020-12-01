from numpy import sqrt, pi, exp, array, log
from helpers.draw import Draw, format_histo, dirname, realpath, join, constants, M_E, beta, discrete_int, arctan, choose
from src.element import Element, hbar

Dir = dirname(dirname(realpath(__file__)))


class CrossSection(object):

    def __init__(self, el: Element):
        self.El = el
        self.KR = 2 * pi * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e6 * 1e4  # ev cm2
        self.Draw = Draw(join(Dir, 'main.ini'))

    @staticmethod
    def get_emax(bg):
        return bg ** 2 * 2 * M_E * 1e6  # eV

    def draw_rf(self, bg, xmin=1, xmax=1e5):
        f = Draw.make_tf1('rf', lambda x: self.rutherford(x, bg), xmin, xmax, title='Rutherford Cross Section')
        self.Draw.histo(f, logx=True, logy=True)

    def rutherford(self, e, bg):
        b2 = beta(bg) ** 2
        return self.KR / b2 * (1 - b2 * e / self.get_emax(bg)) / e ** 2 * 1e18  # Mb

    def get_theta(self, b2):
        return arctan(self.El.E2 * b2 / (1 - self.El.E1 * b2))

    def get_pai(self, b):
        b2, c = b ** 2, constants.alpha / b ** 2 / pi
        ints = [discrete_int(self.El.E[:i], self.El.PIC[:i]) for i in range(self.El.E.size)]
        a0 = c * self.El.PIC / self.El.E / self.El.Z * log(((1 - b2 * self.El.E1) ** 2 + b2 ** 2 * self.El.E2 ** 2) ** -.5)
        a1 = c * 1 / self.El.NE / hbar / constants.speed_of_light * (b2 - self.El.E1 / self.El.E1 ** 2 + self.El.E2 ** 2) * self.get_theta(b2)
        a2 = c * self.El.PIC / self.El.E / self.El.Z * log(2 * M_E * 1e6 * b2 / self.El.E)
        a3 = c * 1 / self.El.E ** 2 / self.El.Z * ints
        # return array([a0, a1, a2, a3])
        return a0 + a1 + a2 + a3

    def draw_pai(self, b=beta(4)):
        self.Draw.graph(self.El.E, self.get_pai(b), draw_opt='al', logy=True, logx=True)

    def draw_pai_rf(self, bg=4, y_range=None):
        pai, rf = self.get_pai(beta(bg)), self.rutherford(self.El.E, bg)
        g = self.Draw.graph(self.El.E, pai/rf, logx=True, logy=True)
        format_histo(g, y_range=choose(y_range, [.1, 20]))
        Draw.horizontal_line(1, 0, 1e5)

    def mfp(self, bg):
        n = self.El.NE / self.El.Z
        return 1 / (n * discrete_int(self.El.E, self.get_pai(beta(bg)))) * 1e22  # um

    def sigma(self, bg):
        n = constants.physical_constants['Avogadro constant'][0] * self.El.Density / self.El.A
        return n * discrete_int(self.El.E, self.get_pai(beta(bg))) * 1e-22  # 1/um

    def draw_mfp(self):
        f = Draw.make_tf1('mfp', self.mfp, .1, 1e3)
        format_histo(f, x_tit='#beta#gamma', y_tit='#lambda [#mum]')
        self.Draw(f, logx=True)

    def cde(self, bg):
        e, cs = self.El.E, self.get_pai(bg)
        return array([discrete_int(e[:i], cs[:i]) for i in range(e.size)]) / discrete_int(e, cs)

    def draw_cde(self, bg=4):
        self.Draw.graph(self.El.E, self.cde(bg), 'CDE', x_tit='Energy [ev]', y_tit='CDE', draw_opt='al', logx=True)


class Landau(object):

    Dir = dirname(dirname(realpath(__file__)))

    def __init__(self):
        self.F = Draw.make_tf1('Landau', self.f, -10, 10)
        self.Draw = Draw(join(Landau.Dir, 'main.ini'))

    @staticmethod
    def f(x):
        return 1 / sqrt(2 * pi) * exp((abs(x) - 1) / 2 - exp(abs(x) - 1))

    def draw(self):
        self.Draw(self.F)

    def root(self, mpv=100, s=10):
        f = Draw.make_f('Landau', 'landau', -100, 1000, [1, mpv, s])
        f.SetNpx(1000)
        format_histo(f, x_range=[0, 500])
        self.Draw(f)


