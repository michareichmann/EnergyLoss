#!/usr/bin/env python

from src.eloss import *

draw = Draw(join(Eloss.Dir, 'main.ini'))


def draw_dens_correction(p, el, y_range=None):
    f, f0 = BetheBloch(p, el), BetheBloch(p, el, dens_corr=False)
    f0.draw(color=draw.get_color(2, 1), line_style=7, y_range=y_range)
    f.draw_same(draw.get_color(2, 0))
    Draw.legend([f.F, f0.F], ['Bethe Bloch', 'without #delta'], 'l', x2=.96, w=.2, scale=1.4)


def draw_restricted(p, el, w_cut: Any = 2, t=500, absolute=False, mass=False, y_range=None):
    b0 = BetheBloch(p, el, absolute=absolute, mass=mass)
    n = 2 + make_list(w_cut).size
    fs = [BetheBloch(p, el, t, d, w, absolute, mass) for d, w in [(1, None), (0, None)] + [(1, iw) for iw in make_list(w_cut) * b0.get_minimum()[0]]]
    fs[1].draw(color=draw.get_color(n, n - 1), line_style=7, y_range=y_range)
    for i, f in enumerate(fs[2:], 1):
        f.draw_same(draw.get_color(n, i), 9)
    fs[0].draw_same(draw.get_color(n, 0))
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'without #delta'] + ['W_{{cut}}={}dE/dx_{{min}}'.format(i) for i in make_list(w_cut)], 'l', x2=.96, w=.23, scale=1.4)


def draw_bethe(p, el, t=500, absolute=False, mass=False, y_range=None):
    draw_restricted(p, el, None, t, absolute, mass, y_range)


def draw_bethe_mpv(p=Pion, el=Dia, t=500, w_cut=2, y_range=None):
    b0 = BetheBloch(p, el)
    fs = [BetheBloch(p, el, t, absolute=True), BetheBloch(p, el, t, absolute=True, wcut=w_cut * b0.get_minimum()[0]), LandauVavilovBichsel(p, el, t, True)]
    fs[1].draw(color=draw.get_color(3, 2), line_style=9, y_range=y_range)
    fs[0].draw_same(draw.get_color(3, 0))
    fs[2].draw_same(draw.get_color(5, 1), line_style=6)
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'W_{{cut}}={}dE/dx_{{min}}'.format(w_cut), 'Landau-Vavilov-Bichsel'], 'l', x2=.89, w=.33, scale=1.4)


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))
