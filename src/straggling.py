from numpy import log, euler_gamma, inf, sum
from numpy.random import randint, poisson
from helpers.draw import *
from src.element import Element, hbar
from src.eloss import LandauVavilovBichsel
from src.particle import Particle

Dir = dirname(dirname(realpath(__file__)))
MetaDir = join(Dir, 'data', 'meta')


class Straggling(object):

    def __init__(self, part: Particle, el: Element, p=260, t=300):

        self.El = el
        self.P = part
        self.Draw = Draw(join(Dir, 'main.ini'))
        self.PBar = PBar()

        # constants
        self.BG = beta_gamma(p, part.M)
        self.B2 = bg2b(self.BG)
        self.T = t
        self.KR = 2 * pi * constants.physical_constants['classical electron radius'][0] ** 2 * M_E * 1e6 * 1e4  # ev cm2
        self.C = constants.alpha / (self.B2 * pi) * constants.physical_constants['Rydberg constant times hc in eV'][0]
        self.WMax = 2 * M_E * self.BG ** 2 / (1 + 2 * self.BG / bg2b(self.BG) * M_E / self.P.M + (M_E / self.P.M) ** 2) * 1e6
        self.Theta = arctan(self.El.E2 * self.B2 / (1 - self.El.E1 * self.B2))

    def set_bg(self, bg=None):
        if bg is not None:
            self.BG = bg
            self.B2 = bg2b(bg)
            self.C = constants.alpha / (self.B2 * pi) * constants.physical_constants['Rydberg constant times hc in eV'][0]

    # -------------------------------------
    # region GET
    def rutherford(self, e):
        emax = self.BG ** 2 * 2 * M_E * 1e6  # eV
        return self.KR / self.B2 * (1 - self.B2 * e / emax) / e ** 2 * 1e18 * constants.physical_constants['Rydberg constant times hc in eV'][0]  # Mb

    def get_transversal(self):
        t1 = self.El.PIC / self.El.E / self.El.Z * log(((1 - self.B2 * self.El.E1) ** 2 + self.B2 ** 2 * self.El.E2 ** 2) ** -.5)
        t2 = 1 / self.El.NE / hbar / constants.speed_of_light * (self.B2 - self.El.E1 / (self.El.E1 ** 2 + self.El.E2 ** 2)) * self.Theta
        return self.C * (t1 + t2)

    def get_longitudinal(self):
        return self.C * self.El.PIC / self.El.E / self.El.Z * log(2 * M_E * 1e6 * self.B2 / self.El.E)

    def get_large(self):
        ints = [discrete_int(self.El.E[:i], self.El.PIC[:i]) for i in range(self.El.E.size)]
        return self.C / self.El.E ** 2 / self.El.Z * ints

    def get_pai(self):
        return self.get_longitudinal() + self.get_transversal() + self.get_large()

    def get_mfp(self, bg=None):
        old = self.BG
        self.set_bg(bg)
        n = self.El.NE / self.El.Z
        mfp = 1 / (n * discrete_int(self.El.E, self.get_pai())) * 1e22  # um
        self.set_bg(old)
        return mfp

    def get_cde(self):
        e, cs = self.El.E, self.get_pai()
        return array([discrete_int(e[:i], cs[:i]) for i in range(e.size)]) / discrete_int(e, cs)

    def make_table(self, n=1e6):
        p, e = self.get_cde(), self.El.E
        vals = []
        info('generating look up table ...')
        self.PBar.start(n)
        for i in range(int(n)):
            j = where(p * n > i)[0][0] - 1
            vals.append((i / n - p[j]) * (e[j + 1] - e[j]) / (p[j + 1] - p[j]) + e[j + 1] if e[j] < self.WMax else self.WMax)
            self.PBar.update()
        return array(vals)

    def get_eloss(self, p, cde):
        i = where(cde > p)[0][0] - 1
        return (p - cde[i]) * (self.El.E[i + 1] - self.El.E[i]) / (cde[i + 1] - cde[i]) + self.El.E[i + 1]

    def get_sigma(self):
        return self.El.N * discrete_int(self.El.E, self.get_pai()) * 1e-22  # 1/um
    # endregion GET
    # -------------------------------------

    # -------------------------------------
    # region DRAW
    def draw_pai(self):
        self.Draw.graph(self.El.E, self.get_pai(), draw_opt='al', logy=True, logx=True, x_tit='Photon Energy [eV]', y_tit='#sigma_{#gamma} [Mb]', center_tit=True, x_off=1.2)

    def draw_rf(self, xmin=1, xmax=1e5):
        f = Draw.make_tf1('rf', lambda x: self.rutherford(x), xmin, xmax, title='Rutherford Cross Section')
        self.Draw(f, logx=True)
        format_histo(f, x_tit='Photon Energy [eV]', y_tit='#sigma_{R} [Mb]', y_off=1.35, center_tit=True, x_off=1.2)

    def draw_pai_rf(self, y_range=None):
        pai, rf = self.get_pai(), self.rutherford(self.El.E)
        g = self.Draw.graph(self.El.E, pai/rf, logx=True, logy=True, draw_opt='al', x_off=1.2, x_tit='Photon Energy [eV]', y_tit='#sigma_{#gamma} / #sigma_{R}')
        format_histo(g, y_range=choose(y_range, ax_range(.1, max(get_graph_y(g, err=False)), 0, .5)), center_tit=1, x_range=[1, 1e5])
        Draw.horizontal_line(1, 0, 1e8)

    def draw_mfp(self):
        def f():
            x = log_bins(100, .1, 1e3)[1]
            return array([x, [self.get_mfp(ix) for ix in x]]).astype('f8')
        data = do_pickle(make_meta_path(MetaDir, 'MFP', self.El.Symbol), f)
        self.Draw.graph(*data, 'Mean Free Path', x_tit='#beta#gamma', y_tit='#lambda [#mum]', logx=True, draw_opt='al', center_tit=True, x_off=1.2)

    def draw_cde(self):
        self.Draw.graph(self.El.E, self.get_cde(), 'CDE', x_tit='Photon Energy [ev]', y_tit='CDE', draw_opt='al', logx=True, center_tit=True, x_off=1.2)

    def draw(self, n=1e5, np=1e5, bin_size=None):
        table = self.make_table(np)
        info('simulating collisions ...')
        e = []
        self.PBar.start(n)
        for n_col in poisson(self.T / self.get_mfp(), int(n)):
            e.append(sum(table[randint(0, np, n_col)]) / 1e3)  # to [keV]
            self.PBar.update()
        h = self.Draw.distribution(e, make_bins(*ax_range(min(e), mean(e) * 2, .1), bin_size, n=sqrt(n) if bin_size is None else None), normalise=True, show=False)
        x, y = get_hist_vecs(h, err=False)
        return self.Draw.graph(x, y, draw_opt='al', x_tit='Energy Loss [keV]', y_tit='Probability')
    # endregion DRAW
    # -------------------------------------


class Landau(object):

    def __init__(self, part: Particle, el: Element, p=260, t=500):

        x_off = .22278
        bg = beta_gamma(p, part.M)
        b2 = bg2b(bg) ** 2
        eloss = LandauVavilovBichsel(part, el, t)
        self.Xi = eloss.get_xi(bg) * 1e3  # [MeV] -> [keV]
        self.C = -log(self.Xi * 2 * M_E * bg ** 2 / el.IE ** 2 * 1e-3) + b2 - 1 + euler_gamma + eloss.density_correction(bg)
        self.MPV = self.Xi * -(self.C + x_off)
        self.ROOTMPV = x_off * self.Xi + self.MPV
        self.Title = 'Energy Loss of {}MeV {}s in {}#mum {}'.format(p, part.Name, t, el.Name)

        self.Draw = Draw(join(Dir, 'main.ini'))

    def lamda(self, x):
        return x / self.Xi + self.C

    def f(self, x):
        return 1 / pi * Draw.make_tf1('i', lambda u: exp(-u * log(u) - self.lamda(x) * u) * sin(pi * u) if self.lamda(x) > -4 else 0).Integral(0, inf)

    @staticmethod
    def f_approx(x):
        return 1 / sqrt(2 * pi) * exp((abs(x) - 1) / 2 - exp(abs(x) - 1))

    def draw(self, xmin=None, xmax=None, same=False):
        xmin, xmax = choose(xmin, self.MPV - 3 * self.Xi), choose(xmax, self.MPV + 15 * self.Xi)
        f = Draw.make_f('Landau', 'landau', xmin, xmax, [1, self.ROOTMPV, self.Xi], npx=500, title=self.Title, x_tit='Energy Loss [keV]', y_tit='Probability')
        f.Draw('same') if same else self.Draw(f)
        return f

    def draw_same(self, scale=1):
        f = self.draw(same=True)
        format_histo(f, line_style=7, line_color=1)
        f.SetParameter(0, scale / f.GetMaximum())
        return f

    def draw_real(self):
        x = linspace(self.MPV - 3 * self.Xi, self.MPV + 10 * self.Xi, 100)
        self.Draw.graph(x, [self.f(i) for i in x], x_tit='Energy Loss [keV]', y_tit='Probability', draw_opt='al', lm=.12, y_off=1.6, title=self.Title)
