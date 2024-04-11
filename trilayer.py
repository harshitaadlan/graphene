#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:54:04 2021

@author: harshita
"""


import matplotlib.pyplot as plt

import pybinding as pb
pb.pltutils.use_style()
from math import sqrt

def trilayer():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping
    t1 = -0.3
    t3 = -0.1
    t4 = -0.12
    lat = pb.Lattice(a1=[a/2, a/2*1.732],
                     a2=[-a/2,a/2*1.732])
    lat.add_sublattices(('B2',[0, -a_cc, 2*a_cc]),('B1',[0,0,2*a_cc]), ('A1', [0, 0, 0]),('A2', [0, a_cc, 0]), ('C1',[0, 0, -2*a_cc]), ('C2',[0,-a_cc,-2*a_cc]))
    lat.add_hoppings(
        # inside the main cell 
        ([0,  0], 'B2', 'B1', t),
        
        ([0,  0], 'B1', 'A1', t),
        ([0,  0], 'A1', 'C1', t),
        ([0,  0], 'A1', 'A2', t),
        ([0,  0], 'C1', 'C2', t),
        # between neighboring cells
        ([0, -1 ], 'B2', 'B1', t),
        ([1, 0], 'C1', 'C2', t),
        ([0, -1 ], 'C2', 'C1', t),
        ([1, 0], 'B1', 'B2', t),
        ([1, 0 ], 'A2', 'A1', t),
        ([0, 1 ], 'A2', 'A1', t)
        )
    return lat

lattice = trilayer()
#lattice.plot()
model = pb.Model(lattice)
print("-------------------------------------------------------------------------------")
print("ABC stacking trilayer graphene lattice")
model.plot(axes='yz')

lat = trilayer()
lat.plot()
plt.show()
print("-------------------------------------------------------------------------------")
lat = bilayer()
lattice.plot()
lat.plot_brillouin_zone()
plt.show()
print("-------------------------------------------------------------------------------")
model = pb.Model(
    lat,
    pb.translational_symmetry()
) 
model.plot()
print('hamiltonian:',model.hamiltonian)
print("------------------------------------------------------------------------------")
print('hamiltonian to dense:',model.hamiltonian.todense())
from math import sqrt, pi

solver = pb.solver.lapack(model)

a_cc = 0.142
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2)
print("------------------------------------------------------------------------------")
print('eigenvalues:',solver.eigenvalues)
print("------------------------------------------------------------------------------")
print(' eigenvectors:',solver.eigenvectors)
print("------------------------------------------------------------------------------")
plt.figure(figsize=(10.841, 20.195), dpi=100)
plot1=bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
print('plot 1:',plot1)
print("------------------------------------------------------------------------------")
model.lattice.plot_brillouin_zone(decorate=False)
plt.savefig('myfig.png', dpi=100)
plot2=bands.plot_kpath(point_labels=['K', r'$\Gamma$', 'M', 'K'])
print('plot2',plot2)