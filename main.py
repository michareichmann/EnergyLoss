#!/usr/bin/env python

from src.eloss import *
from src.straggling import *

draw = Draw(join(Dir, 'main.ini'))


def draw_dens_correction(p, el, y_range=None):
    f, f0 = BetheBloch(p, el), BetheBloch(p, el, dens_corr=False)
    f0.draw(color=draw.get_color(2, 1), line_style=7, y_range=y_range)
    f.draw_same(draw.get_color(2, 0))
    Draw.legend([f.F, f0.F], ['Bethe Bloch', 'without #delta'], 'l', x2=.96, w=.2, scale=1.4)


def draw_restricted(p, el, w_cut: Any = 2, t=500, abst=False, mass=False, y_range=None):
    b0 = BetheBloch(p, el, abst=abst, mass=mass)
    n = 2 + make_list(w_cut).size
    fs = [BetheBloch(p, el, t, d, w, abst, mass) for d, w in [(1, None), (0, None)] + [(1, iw) for iw in make_list(w_cut) * b0.get_minimum()[0]]]
    fs[1].draw(color=draw.get_color(n, n - 1), line_style=7, y_range=y_range)
    for i, f in enumerate(fs[2:], 1):
        f.draw_same(draw.get_color(n, i), 9)
    fs[0].draw_same(draw.get_color(n, 0))
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'without #delta'] + ['W_{{cut}}={}dE/dx_{{min}}'.format(i) for i in make_list(w_cut)], 'l', x2=.96, w=.23, scale=1.4)


def draw_bethe(p, el, t=500, abst=False, mass=False, y_range=None):
    draw_restricted(p, el, None, t, abst, mass, y_range)


def draw_bethe_mpv(p=Pion, el=Dia, t=500, w_cut=2, y_range=None):
    b0 = BetheBloch(p, el)
    fs = [BetheBloch(p, el, t, abst=True), BetheBloch(p, el, t, abst=True, wcut=w_cut * b0.get_minimum()[0]), LandauVavilovBichsel(p, el, t, True)]
    fs[1].draw(color=draw.get_color(3, 2), line_style=9, y_range=y_range)
    fs[0].draw_same(draw.get_color(3, 0))
    fs[2].draw_same(draw.get_color(5, 1), line_style=6)
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'W_{{cut}}={}dE/dx_{{min}}'.format(w_cut), 'Landau-Vavilov-Bichsel'], 'l', x2=.89, w=.33, scale=1.4)


def draw_ep(el=Dia, t=500, abst=False, mass=False, logy=False, y_range=None, xmin=.1, xmax=1e3):
    bethe_el = BetheBloch(Electron, el, t, abst=abst, mass=mass).draw(y_range=y_range, xmin=xmin, xmax=xmax, logy=logy, color=Draw.get_colors(10)[0])
    bethe_pos = BetheBloch(Positron, el, t, abst=abst, mass=mass).draw_same(Draw.get_colors(10)[0], 2)
    brems = Bremsstrahlung(Electron, el, t, abst, mass).draw_same(Draw.get_colors(10)[7], 10)
    data = el.draw_brems(not abst, mass, t=t / 10, color=Draw.get_colors(10)[9])
    total = (bethe_el + data).draw_same(1)
    Draw.legend([bethe_el.F, bethe_pos.F, total.F, brems.F, data], ['Ionisation e^{-}', 'Ionisation e^{+}', 'total Eloss', 'Bremsstrahlung appr.', 'Bremstrahlung exact'], 'l')
    return total


def beta_gamma_range():
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(beta_gamma(260, M_PI), beta_gamma(1.2e5, M_PI)))


def get_losses(p, el, t=500):
    return array([BetheBloch(part, el, t, abst=True)(p) for part in [Pion, Muon, Positron, Proton]])


def print_energies(p, el: Element, t=500, table=False):
    particles = [Pion, Muon, Positron, Proton]
    values = []
    for part in particles:
        e = BetheBloch(part, el, t, abst=True)(p)
        values.append(['{:1.0f}'.format(e), '{:1.0f}'.format(round(e * 1000 / el.EEH, -2))])
        info('Energy loss of {} MeV {}s in {} μm {}: {:1.0f} keV ({:1.0f} eh-pairs)'.format(p, part.Name, t, el.Name, e, round(e * 1000 / el.EEH, -2)))
    if table:
        print(make_latex_table_row(concatenate(array(values).T)))


def draw_straggling(part=Pion, el=Dia, p=260, t=500, n=1e6, bin_size=1):
    g = Straggling(part, el, p, t).draw(n, bin_size=bin_size)
    f = Landau(part, el, p, t).draw_same(max(get_graph_y(g, err=False)))
    Draw.legend([g, f], ['Straggling', 'Landau'], 'l')


if __name__ == '__main__':
    z = Straggling(Pion, Si)
    a = Landau(Pion, Dia, 260, 500)
