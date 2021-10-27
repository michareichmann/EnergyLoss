from numpy import log, euler_gamma, inf, sum, convolve
from numpy.random import randint
from scipy.stats import poisson
from helpers.draw import *
from src.element import Element, hbar
from src.eloss import LandauVavilovBichsel
from src.particle import Particle
from ROOT import gRandom

MetaDir = join(Dir, 'data', 'meta')


class Straggling(object):

    def __init__(self, part: Particle, el: Element, p=260, t=300):

        self.El = el
        self.E = el.E
        self.P = part
        self.Draw = Draw(join(Dir, 'main.ini'))
        self.PBar = PBar()

        # constants
        self.BG = pm2bg(p, part.M)
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
            self.Theta = arctan(self.El.E2 * self.B2 / (1 - self.El.E1 * self.B2))

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
        ints = [discrete_int(self.El.E[:i + 1], self.El.PIC[:i + 1]) for i in range(self.El.E.size)]
        return self.C / self.El.E ** 2 / self.El.Z * ints

    def get_pai(self):
        return self.get_longitudinal() + self.get_transversal() + self.get_large()

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

    def get_moment(self, n=0, bg=None):
        old = self.BG
        self.set_bg(bg)
        v = self.El.N * discrete_int(self.El.E, self.get_pai() * self.El.E ** n) * 1e-22  # 1/um
        self.set_bg(old)
        return v

    def get_ccs(self, bg=None):
        return self.get_moment(0, bg)

    def get_stopping(self, bg=None):
        return self.get_moment(1, bg) * 1e-2  # [um] -> [cm], [eV] -> [MeV]

    def get_width(self, bg=None):
        return self.get_moment(2, bg)

    def get_mfp0(self, bg=None):
        return 1 / self.get_ccs(bg)

    def get_mfp(self, bg=None):
        data = genfromtxt(join(Dir, 'data', 'SiRef.txt'))
        bg = choose(bg, self.BG)
        bgs, ccs = data[:, [0, 2]].T
        return 1 / points2x(ccs, bgs, bg) if bg <= bgs[-1] else 1 / ccs[-1]

    def get_convolutions(self, n=10, xmax=1000, step=.1):
        x = arange(0, xmax, step)
        s = self.get_pai()
        y1 = array([points2y(self.E, s, ix) for ix in x])
        convs = [y1]
        for _ in range(n - 1):
            convs.append(convolve(y1, convs[-1])[:x.size] * step)
        return array(convs)
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
        for n_col in poisson.rvs(self.T / self.get_mfp(), size=int(n)):
            e.append(sum(table[randint(0, np, n_col)]) / 1e3)  # to [keV]
            self.PBar.update()
        h = self.Draw.distribution(e, make_bins(*ax_range(min(e), mean(e) * 2, .1), bin_size, n=sqrt(n) if bin_size is None else None), normalise=True, show=False)
        x, y = get_hist_vecs(h, err=False)
        return self.Draw.graph(x, y, draw_opt='al', x_tit='Energy Loss [keV]', y_tit='Probability')

    def draw_conv(self, n=5, xmax=2000, step=.1):
        x = arange(0, xmax, step)
        graphs = [Draw.make_tgrapherrors(x, c, color=self.Draw.get_color(n)) for c in self.get_convolutions(n, xmax, step)]
        tits = [str(i) for i in range(1, n + 1)] if n <= 10 else None
        self.Draw.multigraph(graphs, 'Cross Section Convolutions', tits, x_range=[0, 100], y_range=[1e-6, .12], draw_opt='al', logy=True, logx=True)

    def test0(self):
        n = int(log2(self.T)) + 2
        # t0 = self.get_ccs() * self.T / 2 ** n
        t0 = self.T / 2 ** n / self.get_mfp()
        convolutions = self.get_convolutions(10, 30000, 1)
        d_poission = poisson.pmf(arange(1, 11), t0)
        y = sum(d_poission.reshape(d_poission.size, 1) * convolutions, axis=0)
        y /= discrete_int(arange(y.size), y)
        for i in range(n):
            y = convolve(y, y)[:int(10000 * (1.3 ** (n + 1)))]
            y /= discrete_int(arange(y.size), y)
        self.Draw.graph(arange(y.size) / 1e3, y, draw_opt='al')

    # endregion DRAW
    # -------------------------------------


class Landau(object):

    XOff = -.22278

    def __init__(self, part: Particle, el: Element, p=260, t=500):

        bg = pm2bg(p, part.M)
        b2 = bg2b(bg) ** 2
        eloss = LandauVavilovBichsel(part, el, t)
        self.Xi = eloss.get_xi(bg) * 1e3  # [MeV] -> [keV]
        self.C = -log(self.Xi * 2 * M_E * bg ** 2 / el.IE ** 2 * 1e-3) + b2 - 1 + euler_gamma + eloss.density_correction(bg)
        self.MPV = self.Xi * (Landau.XOff - self.C)
        self.Location = Landau.mpv2loc(self.MPV, self.Xi)
        self.Title = 'Energy Loss of {}MeV {}s in {}#mum {}'.format(p, part.Name, t, el.Name)

        self.Draw = Draw(join(Dir, 'main.ini'))
        self.F = Draw.make_f('Landau', 'landau', -50, self.MPV * 5, [1, self.Location, self.Xi])

    def __call__(self, x):
        return self.f(x)

    def lamda(self, x):
        return x / self.Xi + self.C

    def f(self, x):
        return 1 / pi * Draw.make_tf1('i', lambda u: exp(-u * log(u) - self.lamda(x) * u) * sin(pi * u) if self.lamda(x) > -4 else 0).Integral(0, inf)

    @staticmethod
    def f_approx(x):
        return 1 / sqrt(2 * pi) * exp((abs(x) - 1) / 2 - exp(abs(x) - 1))

    @staticmethod
    def loc2mpv(loc, width):
        return loc + Landau.XOff * width

    @staticmethod
    def mpv2loc(mpv, width):
        return mpv - Landau.XOff * width

    def draw(self, xmin=None, xmax=None, same=False):
        xmin, xmax = choose(xmin, self.MPV - 3 * self.Xi), choose(xmax, self.MPV + 15 * self.Xi)
        f = Draw.make_f('Landau', 'landau', xmin, xmax, [1, self.Location, self.Xi], npx=500, title=self.Title, x_tit='Energy Loss [keV]', y_tit='Probability')
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

    def get_mean(self, cut_off, n=1000):
        x = linspace(-50, cut_off, n)
        return mean_sigma(x, [self.F(ix) for ix in x])[0]

    def rpv(self, n=1e5):
        return array([gRandom.Landau(self.Location, self.Xi) for _ in range(int(n))])

    def get_mean_scale(self, n=1e5, cut_off=410, show=False):
        x = self.rpv(n)
        s = linspace(.02, 2, 50)
        m = array([mean(v[v < cut_off]) for v in [x * i for i in s]])
        g = self.Draw.graph(m[m < cut_off - 100], (self.MPV * s)[m < cut_off - 100], draw_opt='al', show=show)
        return FitRes(g.Fit('pol2', 'qs'))

    def get_r(self):
        return self.Xi / self.MPV


def draw_mean_vs_t(n=10, cut_off=2.3 * 410, tmax=500):
    x = linspace(10, tmax, n)
    from src.particle import Pion
    from src.element import Dia
    landaus = [Landau(Pion, Dia, t=t) for t in x]
    # y = [lan.get_mean(cut_off) for lan in landaus]
    m, w, mp = array([[lan.get_mean(cut_off), lan.Xi, lan.MPV] for lan in landaus]).T
    y = w / mp
    Draw(join(Dir, 'main.ini')).graph(x, y, 'Mean vs. Thickness', x_tit='Thicknes [#mum]', y_tit='Mean Pulse Height [keV]')
