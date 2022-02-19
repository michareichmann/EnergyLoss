#!/usr/bin/env python

from src.eloss import *
from src.straggling import *
from src.scattering import *
from src.dose import Dose, a2f
import helpers.latex as latex

ddraw = Draw(join(Dir, 'main.ini'))


def draw_dens_correction(p, el, y_range=None):
    f, f0 = BetheBloch(p, el), BetheBloch(p, el, dens_corr=False)
    f0.draw(color=ddraw.get_color(2, 1), line_style=7, y_range=y_range)
    f.draw_same(ddraw.get_color(2, 0))
    Draw.legend([f.F, f0.F], ['Bethe Bloch', 'without #delta'], 'l', x2=.96, w=.2, scale=1.4)


def draw_density_correction(p: Particle = Pion, el: Element = Dia, xmin=.1, xmax=1e3, **dkw):
    b = BetheBloch(p, el)
    f = Draw.make_tf1('dens', b.density_correction, xmin, xmax)
    ddraw(f, **prep_kw(dkw, x_tit='#beta#gamma', y_tit='Density-Effect Correction', logx=True))
    [Draw.vertical_line(10 ** x, 0, 1e3) for x in [el.x0, el.x1]]


def draw_restricted(p, el, w_cut: Any = 2, t=500, abst=False, mass=False, y_range=None):
    b0 = BetheBloch(p, el, abst=abst, mass=mass)
    n = 2 + make_list(w_cut).size
    fs = [BetheBloch(p, el, t, d, w, abst, mass) for d, w in [(1, None), (0, None)] + [(1, iw) for iw in make_list(w_cut) * b0.get_minimum()[0]]]
    fs[1].draw(color=ddraw.get_color(n, n - 1), line_style=7, y_range=y_range)
    for i, f in enumerate(fs[2:], 1):
        f.draw_same(ddraw.get_color(n, i), 9)
    fs[0].draw_same(ddraw.get_color(n, 0))
    Draw.legend([f.F for f in fs], ['Bethe Bloch', 'without #delta'] + ['W_{{cut}}={}dE/dx_{{min}}'.format(i) for i in make_list(w_cut)], 'l', x2=.96, w=.23, scale=1.4)


def draw_bethe(p, el, t=500, abst=False, mass=False, y_range=None):
    draw_restricted(p, el, None, t, abst, mass, y_range)


def draw_bethe_mpv(p=Pion, el=Dia, t=500, w_cut=2, y_range=None):
    b0 = BetheBloch(p, el)
    fs = [BetheBloch(p, el, t, abst=True), BetheBloch(p, el, t, abst=True, wcut=w_cut * b0.get_minimum()[0]), LandauVavilovBichsel(p, el, t, True)]
    fs[1].draw(color=ddraw.get_color(3, 2), line_style=9, y_range=y_range)
    fs[0].draw_same(ddraw.get_color(3, 0))
    fs[2].draw_same(ddraw.get_color(5, 1), line_style=6)
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
    info('Beta Gamma Range: {:1.1f} ~ {:1.1f}'.format(pm2bg(260, M_PI), pm2bg(1.2e5, M_PI)))


def get_losses(p, el, t=500):
    return array([BetheBloch(part, el, t, abst=True)(p) for part in [Pion, Muon, Positron, Proton]])


def print_eloss(p, latex=False, eh=False, mpv=False):
    particles = [Pion, Muon, Positron, Proton]
    els = [(Dia, 500), (Si, 300), (Si, 100)]
    rows = [[] for _ in range(len(els))]
    eloss = (LandauVavilovBichsel if mpv else BetheBloch)
    for i, (el, t) in enumerate(els):
        e_mip = eloss(Pion, el, t=t, abst=True).get_minimum()[0]
        rows[i] = [el.Name, str(t), '{:1.1f}'.format(el.SEH * t / 1000) if eh else '{:1.0f}'.format(e_mip)]
        for part in particles:
            e = eloss(part, el, t, abst=True)(p)
            rows[i].append('{:1.1f}'.format(e / e_mip * el.SEH * t / 1000) if eh else '{:1.2f}'.format(e / e_mip))
    if not latex:
        header = ['Material', 'Thickness', 'MIP'] + [p.Name for p in particles]
        print_table(rows, header)
    else:
        for row in rows:
            print(make_latex_table_row(row, endl=False))


def print_eh_loss(p, tex=False):
    els = [(Dia, 500), (Si, 300), (Si, 100)]
    rows = []
    for e, t in els:
        m, mpv = [f(Pion, e, t, abst=True) for f in [BetheBloch, LandauVavilovBichsel]]
        d, fac = e.SEH * t / 1000, m.eh_pairs(p) / m.eh_pairs(m.pmin)
        rows.append([e.Name, str(t)] + [f'{i:.1f}' for i in [d, m.eh_pairs(m.pmin), mpv.eh_pairs(mpv.pmin), d * fac, m.eh_pairs(p), mpv.eh_pairs(p)]])
    if not tex:
        print_table(rows, header=['Material', 'Thickness'] + ['Data', 'Mean', 'MPV'] * 2)
    else:
        [print(make_latex_table_row(row, endl=False)) for row in rows]


def draw_straggling(part=Pion, el=Dia, p=260, t=500, n=1e6, bin_size=1):
    g = Straggling(part, el, p, t).draw(n, bin_size=bin_size)
    f = Landau(part, el, p, t).draw_same(max(get_graph_y(g, err=False)))
    Draw.legend([g, f], ['Straggling', 'Landau'], 'l')


def draw_scattering(part=Electron, el=Dia, t=500, ymax=None, xmin=100, xmax=1e5, **dkw):
    s = Scattering(part, el, t)
    s.draw(show=False)
    return s.draw(ymax, xmin, xmax, **dkw).F


def get_beam_dose(f=None, t=None):
    f = choose(f, ufloat(10e6, 3e6))
    t = choose(t, ufloat(2, .5)) * 3600
    d = Dose(Pion, Dia, t=500, p=260)
    return f, t, d.Eloss, d(t, f)


def get_source_dose(a=None, t=None, r=None):
    a = choose(a, ufloat(30e6, 5e6))  # Bq
    t = choose(t, ufloat(4, .5)) * 3600  # hr
    r = choose(r, ufloat(1.5, .3))  # cm
    e = BetheBloch(Electron, Dia, abst=True).get_emin() * 1.08  # 90Sr 8% more ionising than mip (from kramberger)
    return a, r, t, e, Dose(Electron, Dia, eloss=e)(f=a2f(a, r), t=t)


def print_doses(f0=None, t0=None, tex=False):
    f0, t0, e0, d0 = get_beam_dose(f0, t0)
    a, r, t1, e1, d1 = get_source_dose()
    if tex:
        mc, m, u = latex.makecell, latex.math, latex.unit
        header = latex.bold('Method') + [mc('A', '[MBq]'), mc('r', '[cm]'), mc(m('\\Phi'), u('mhzcm')), mc('t', '[h]'), mc('dE/dx', '[MeV]')] + latex.bold(mc('D', u('micro rad')))
        rows = [['pion beam', '-', '-'] + latex.si(f0 / 1e6, t0.n / 3600, fmt='.1f') + latex.si(e0 / 1e3, fmt='.2f') + latex.si(d0 * 1e6, fmt='.1f'),
                ['\\isotope[90]{Sr} source'] + latex.si(a / 1e6, fmt='.0f') + latex.si(r, a2f(a, r) / 1e6, t1.n / 3600, fmt='.1f') + latex.si(e1 / 1e3, fmt='.2f') + latex.si(d1 * 1e6, fmt='.1f')]
        print(latex.table(header, rows))
    else:
        info(f'Beam dose ({f0 / 1e6} MHz/cm² for {t0 / 3600} hrs): {d0} rad')
        info(f'Source dose ({a / 1e6} MBq in a distance of {r} cm for {t1 / 3600} hrs): {d1} rad')
    return d0, d1


if __name__ == '__main__':
    zd = Straggling(Pion, Dia, 260, 500)
    zs = Straggling(Pion, Si, 260, 300)
    z = Straggling(Electron, Si, 10e3, 148)
