#!/usr/bin/env python
# --------------------------------------------------------
#       calculate dose for a detector
# created on February 19th 2022 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------
from plotting.draw import *
from src.element import *
from src.particle import *
from src.eloss import BetheBloch
from scipy.constants import electron_volt


class Dose:

    def __init__(self, part: Particle, el: Element, p=260, eloss=None):
        self.Eloss = choose(eloss, BetheBloch(part, el)(p))  # MeV/cm
        self.P = part
        self.E = el
        self.Draw = Draw(join(Dir, 'main.ini'))

    def __call__(self, t, f):
        return self.f(t, f)

    def __repr__(self):
        return f'Dose for {self.P.Name}s in {self.E.Name}'

    def f(self, t, f):
        return f * t * self.Eloss / self.E.Density * electron_volt * 1e6 * 1e3  # g -> kg, MeV -> eV


def a2f(a, r):
    return a / (4 * pi * r ** 2)
