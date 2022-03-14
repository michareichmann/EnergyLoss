#!/usr/bin/env python
from os.path import dirname, realpath, join
from helpers.utils import choose, print_table
from numpy import genfromtxt, array, mean, sqrt
from plotting.draw import Draw, format_histo

Dir = dirname(dirname(realpath(__file__)))
DataDir = join(Dir, 'data', 'semiconductors')
BandGap = genfromtxt(join(DataDir, 'bandgap.txt'), dtype=['U7'] * 2 + ['f'] * 3)
ThermalCond = genfromtxt(join(DataDir, 'thermal.txt'), dtype=['U7'] * 2 + ['f'])
Mobility = genfromtxt(join(DataDir, 'mobilities.txt'), dtype=['U7'] * 2 + ['f'] * 2)
Cubic = genfromtxt(join(DataDir, 'cubic.txt'), dtype=['U7'] + ['f'] * 3)
NonCubic = genfromtxt(join(DataDir, 'noncubic.txt'), dtype=['U7'] + ['f'] * 4)


class Semiconductor(object):

    def __init__(self, name, lattice=None):
        self.Name = name
        self.Lattice = choose(lattice, self.load_data(BandGap, 1, 0)).strip('()')
        self.BandGap = self.load_bandgap()
        self.ThermalCond = self.load_data(ThermalCond, 2)
        self.EMobility = self.load_data(Mobility, 2)
        self.HMobility = self.load_data(Mobility, 3)
        self.Mobility = mean([self.EMobility, self.HMobility])
        self.Draw = Draw(join(Dir, 'main.ini'))
        self.Name = 'Diamond' if name == 'C' else name

    def __repr__(self):
        header = ['Name', 'Lattice', 'Bandgap [eV]', 'μe [cm2/Vs]', 'μh [cm2/Vs]', 'K [W/cmK]']
        return print_table([[self.Name, self.Lattice] + ['{:.3f}'.format(w) if w is not None else '?' for w in [self.BandGap, self.EMobility, self.HMobility, self.ThermalCond]]], header=header,
                           prnt=False)

    def load_data(self, d, i, ilat=None):
        data = d[d['f0'] == self.Name]
        data = list(data[data['f1'] == '({})'.format(self.Lattice)]) if ilat is None else list(data)
        ilat = 0 if ilat is None else ilat
        return data[ilat][i] if data else None

    def load_bandgap(self):
        data = array([self.load_data(BandGap, i) for i in range(2, 5)])
        return None if data[0] is None else min(data[data > 0])


sc = ['C', 'Si', 'SiC', 'Ga', 'GaAs', 'GaN', 'GaP', 'GaN', 'CdS', 'PbS', 'InP', 'InN', 'AlAs', 'AlSb', 'AlN', 'AlP', 'ZnS', 'ZnO', 'ZnSe']
common = ['Si', 'Ge', 'GaAs', '3C-SiC', 'GaN', 'GaP', 'CdS']
Si = Semiconductor('Si')
C = Semiconductor('C')


def draw_map():
    s = [Semiconductor(n, lat) for n, lat in zip(Mobility['f0'], Mobility['f1'])]
    draw = s[0].Draw
    colors = draw.get_colors(2)
    graphs = [Draw.make_tgrapherrors([i.BandGap], [i.Mobility], markersize=sqrt(i.ThermalCond) + .5, color=colors[i.Name in common]) for i in s]
    mg = draw.multigraph(graphs, 'Semiconducor Properties', x_tit='Band Gap [eV]', y_tit='#splitline{Average Charge Carrier}{       Mobility [cm^{2}/Vs]}', color=False, logy=True, w=2,
                         bm=.25, lm=.12, grid=True)
    format_histo(mg, y_range=[30, 1e5], center_tit=True, x_off=1.2, y_off=.9, lab_size=.06, tit_size=.06)
    Draw.legend([Draw.make_tgrapherrors([0], [0], color=colors[i], markersize=3) for i in [1, 0]], ['common semiconductors', 'uncommon semiconductors'], 'p', nentries=3, w=.3)
    for i in s:
        al = 13 if i.Name in ['InAs', '3C-SiC', 'GaAs', 'GaP'] else 33 if i.Name in ['AlAs'] else 23
        draw.tlatex(i.BandGap, i.Mobility * (.98 - .5 / 4.7 * sqrt(i.ThermalCond)), i.Name, size=.03, align=al)

