from numpy import sqrt, pi, exp, array, log, euler_gamma, sin, inf, linspace, where
from numpy.random import rand, poisson
from helpers.draw import Draw, format_histo, dirname, realpath, join, constants, M_E, bg2b, discrete_int, arctan, choose, beta_gamma, do_pickle, log_bins, make_meta_path, PBar
from src.element import Element, hbar
from src.particle import Particle
from src.eloss import K

Dir = dirname(dirname(realpath(__file__)))
MetaDir = join(Dir, 'data', 'meta')


class CrossSection(object):

    E

    def __init__(self, el: Element):
        self.El = el
        self.KR = 2 * pi * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e6 * 1e4  # ev cm2
        self.Draw = Draw(join(Dir, 'main.ini'))
        self.Er, self.G = 18.5, 2.9

    @staticmethod
    def get_emax(bg):
        return bg ** 2 * 2 * M_E * 1e6  # eV

    def draw_rf(self, bg, xmin=1, xmax=1e5):
        f = Draw.make_tf1('rf', lambda x: self.rutherford(x, bg), xmin, xmax, title='Rutherford Cross Section')
        self.Draw.histo(f, logx=True, logy=True)

    def rutherford(self, e, bg):
        b2 = bg2b(bg) ** 2
        return self.KR / b2 * (1 - b2 * e / self.get_emax(bg)) / e ** 2 * 1e18  # Mb

    def e2(self, e):
        return self.Er ** 2 * self.G / e ** 3 / (1 + (self.G / e) ** 2)

    def e1(self, e):
        return 1 - self.Er ** 2 / e ** 2 / (1 + (self.G / e) ** 2)

    def repl(self):
        cut = (self.El.E > 10) & (self.El.E < 20)
        self.El.E2[cut] = self.e2(self.El.E[cut])
        self.El.E1[cut] = self.e1(self.El.E[cut])
        self.El.PIC = self.El.E2 * self.El.E / (self.El.N * constants.speed_of_light * hbar) * 1e16
        c = (self.El.E > 5.9) & (self.El.E < 7.1)
        self.El.E2[c] = [5.95, 5.0, 4.3, 3.7, 3.1]

    def get_theta(self, b2):
        return arctan(self.El.E2 * b2 / (1 - self.El.E1 * b2))

    def get_transverse(self, b):
        t1 = self.El.PIC / self.El.E / self.El.Z * log(((1 - b ** 2 * self.El.E1) ** 2 + b ** 4 * self.El.E2 ** 2) ** -.5)
        t2 = 1 / self.El.NE / hbar / constants.speed_of_light * (b ** 2 - self.El.E1 / (self.El.E1 ** 2 + self.El.E2 ** 2)) * self.get_theta(b ** 2)
        return constants.alpha / b ** 2 / pi * (t1 + t2)

    def get_longitudinal(self, b):
        return constants.alpha / b ** 2 / pi * self.El.PIC / self.El.E / self.El.Z * log(2 * M_E * 1e6 * b ** 2 / self.El.E)

    def get_pai(self, b):
        b2, c = b ** 2, constants.alpha / b ** 2 / pi
        ints = [discrete_int(self.El.E[:i], self.El.PIC[:i]) for i in range(self.El.E.size)]
        a0 = c * self.El.PIC / self.El.E / self.El.Z * log(((1 - b2 * self.El.E1) ** 2 + b2 ** 2 * self.El.E2 ** 2) ** -.5)
        a1 = c * 1 / self.El.NE / hbar / constants.speed_of_light * (b2 - self.El.E1 / (self.El.E1 ** 2 + self.El.E2 ** 2)) * self.get_theta(b2)
        a2 = c * self.El.PIC / self.El.E / self.El.Z * log(2 * M_E * 1e6 * b2 / self.El.E)
        a3 = c * 1 / self.El.E ** 2 / self.El.Z * ints
        # return array([a0, a1, a2, a3])
        return (a0 + a1 + a2 + a3) * constants.physical_constants['Rydberg constant times hc in eV'][0]

    def get_mfp(self, bg):
        n = self.El.NE / self.El.Z
        return 1 / (n * discrete_int(self.El.E, self.get_pai(bg2b(bg)))) * 1e22  # um

    def get_cde(self, bg):
        e, cs = self.El.E, self.get_pai(bg)
        return array([discrete_int(e[:i], cs[:i]) for i in range(e.size)]) / discrete_int(e, cs)

    def get_eloss(self, p, cde):
        i = where(cde > p)[0][0] - 1
        return (p - cde[i]) * (self.El.E[i + 1] - self.El.E[i]) / (cde[i + 1] - cde[i]) + self.El.E[i + 1]

    def draw_pai(self, b=bg2b(4)):
        self.Draw.graph(self.El.E, self.get_pai(b), draw_opt='al', logy=True, logx=True)

    def draw_pai_rf(self, bg=4, y_range=None):
        pai, rf = self.get_pai(bg2b(bg)), self.rutherford(self.El.E, bg)
        g = self.Draw.graph(self.El.E, pai/rf, logx=True, logy=True)
        format_histo(g, y_range=choose(y_range, [.1, 20]))
        Draw.horizontal_line(1, 0, 1e5)

    def sigma(self, bg):
        return self.El.N * discrete_int(self.El.E, self.get_pai(bg2b(bg))) * 1e-22  # 1/um

    def draw_mfp(self):
        def f():
            x = log_bins(100, .1, 1e3)[1]
            return array([x, [self.get_mfp(ix) for ix in x]]).astype('f8')
        data = do_pickle(make_meta_path(MetaDir, 'MFP', self.El.Symbol), f)
        self.Draw.graph(*data, 'Mean Free Path', x_tit='#beta#gamma', y_tit='#lambda [#mum]', logx=True, draw_opt='al', center_tit=True, x_off=1.2)

    def draw_cde(self, bg=4):
        self.Draw.graph(self.El.E, self.get_cde(bg), 'CDE', x_tit='Energy [ev]', y_tit='CDE', draw_opt='al', logx=True)

    def draw(self, bg, t=300, n=1e5):
        pbar = PBar()
        pbar.start(n)
        collissions = poisson(t / self.get_mfp(bg), int(n))
        cde = self.get_cde(bg)
        e = []
        for n_col in collissions:
            e.append(sum(self.get_eloss(p, cde) for p in rand(n_col)))
            pbar.update()
        self.Draw.distribution(e)


class Landau(object):

    def __init__(self, part: Particle, el: Element, p=260, t=500):

        bg = beta_gamma(p, part.M)
        b2 = bg2b(bg) ** 2
        self.Xi = t * 1e-4 * el.Density * K * el.Z / el.A / b2 * 1e3  # [keV]
        self.EPrime = el.IE ** 2 * (1 - b2) / (2 * M_E * b2) * exp(b2) * 1e3  # [keV]
        self.MPV0 = self.Xi * (log(self.Xi / self.EPrime) + 1 - euler_gamma - .223)  # [keV]
        self.MPV = self.get_mpv(p, part.Symbol, t, el.Symbol)
        self.ROOTMPV = 0.22278 * self.Xi + self.MPV
        self.Title = 'Energy Loss of {}MeV {}s in {}#mum {}'.format(p, part.Name, t, el.Name)

        self.Draw = Draw(join(Dir, 'main.ini'))

    def lamda(self, x):
        return x / self.Xi - (log(self.Xi / self.EPrime) - 1 + euler_gamma)

    def f(self, x):
        return 1 / pi * Draw.make_tf1('i', lambda u: exp(-u * log(u) - self.lamda(x) * u) * sin(pi * u) if self.lamda(x) > -4 else 0).Integral(0, inf)

    @staticmethod
    def f_approx(x):
        return 1 / sqrt(2 * pi) * exp((abs(x) - 1) / 2 - exp(abs(x) - 1))

    def get_mpv(self, p, np, t, ne):
        def f():
            return Draw.make_tf1('f', self.f).GetMaximumX(self.MPV0 - 2 * self.Xi, self.MPV0)
        return do_pickle(make_meta_path(MetaDir, 'Landau', 'MPV', suffix='{}{}_{}{}'.format(p, np, t, ne)), f)

    def draw(self, xmin=None, xmax=None):
        xmin, xmax = choose(xmin, self.MPV - 3 * self.Xi), choose(xmax, self.MPV + 15 * self.Xi)
        f = Draw.make_f('Landau', 'landau', xmin, xmax, [1, self.ROOTMPV, self.Xi], npx=500, title=self.Title, x_tit='Energy Loss [keV]', y_tit='Probability')
        self.Draw(f)

    def draw_real(self):
        # f = Draw.make_tf1('Landau', self.f, self.MPV - self.Xi, self.MPV * 10 * self.Xi)
        x = linspace(self.MPV - 3 * self.Xi, self.MPV + 10 * self.Xi, 100)
        self.Draw.graph(x, [self.f(i) for i in x], x_tit='Energy Loss [keV]', y_tit='Probability', draw_opt='al', lm=.12, y_off=1.6, title=self.Title)

    def root(self, mpv=100, s=10):
        f = Draw.make_f('Landau', 'landau', -100, 1000, [1, mpv, s])
        f.SetNpx(1000)
        format_histo(f, x_range=[0, 500])
        self.Draw(f)
