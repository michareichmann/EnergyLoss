from numpy import sqrt, pi, exp
from helpers.draw import Draw, format_histo, dirname, realpath, join


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


