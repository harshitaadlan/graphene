#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:50:10 2021

@author: harshita
"""
import matplotlib.pyplot as plt

import pybinding as pb
pb.pltutils.use_style()
from math import sqrt
from math import sqrt, pi
from pybinding.repository import graphene
plt.figure(figsize=(3.841, 7.195), dpi=100)
model = pb.Model(graphene.bilayer())
model.plot(axes='yz')
model = pb.Model(graphene.bilayer(), pb.translational_symmetry())
model = pb.Model(graphene.bilayer(), pb.translational_symmetry())
model.plot()
solver = pb.solver.lapack(model)
bands = solver.calc_bands(K1, Gamma, M, K2, Gamma)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
plt.savefig('myfig.png', dpi=1000)