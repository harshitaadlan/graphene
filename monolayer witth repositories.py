#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:48:58 2021

@author: harshita
"""


#%%
import matplotlib.pyplot as plt

import pybinding as pb
pb.pltutils.use_style()
from math import sqrt
from math import sqrt, pi
from pybinding.repository import graphene

from math import sqrt, pi

model = pb.Model(graphene.monolayer(), pb.translational_symmetry())
solver = pb.solver.lapack(model)

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2,Gamma)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
