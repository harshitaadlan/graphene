#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:54:21 2021

@author: harshita
"""


#ENTIRE MONOLAYER
#%%

import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
pb.pltutils.use_style()
import pybinding as pb
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
from math import sqrt

def monolayer_graphene():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B', [0,  a_cc/2]))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat

lat = monolayer_graphene()
lat.plot()
plt.show()
lat = monolayer_graphene()
#lattice.plot()
lat.plot_brillouin_zone()
plt.show()
model = pb.Model(
    lat,
    pb.translational_symmetry()
) 
model.plot(axes='yz')
#model.plot()
print('hamiltonian:',model.hamiltonian)
print('hamiltonian to dense:',model.hamiltonian.todense())
from math import sqrt, pi

solver = pb.solver.lapack(model)

a_cc = 0.152
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2)
print('eigenvalues:',solver.eigenvalues)
print(' eigenvectors:',solver.eigenvectors)
plot1=bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
print('plot 1:',plot1)
model.lattice.plot_brillouin_zone(decorate=False)
#plot2=bands.plot_kpath(point_labels=['K', r'$\Gamma$', 'M', 'K'])
#print('plot2',plot2)
#%%



import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
pb.pltutils.use_style()
import pybinding as pb

from math import sqrt

def monolayer_graphene():
    a = 0.34595   # [nm] unit cell length
    a_cc = 0.482  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B', [0,  a_cc/2]))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat

lat = monolayer_graphene()
lat.plot()
plt.show()
lat = monolayer_graphene()
#lattice.plot()
lat.plot_brillouin_zone()
plt.show()
model = pb.Model(
    lat,
    pb.translational_symmetry()
) 
#model.plot()
print('hamiltonian:',model.hamiltonian)
print('hamiltonian to dense:',model.hamiltonian.todense())
from math import sqrt, pi

solver = pb.solver.lapack(model)

a_cc = 0.152
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2, Gamma)
print('eigenvalues:',solver.eigenvalues)
print(' eigenvectors:',solver.eigenvectors)
plot1=bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
print('plot 1:',plot1)
model.lattice.plot_brillouin_zone(decorate=False)
#plot2=bands.plot_kpath(point_labels=['K', r'$\Gamma$', 'M', 'K'])
#print('plot2',plot2)

#%%
import matplotlib.pyplot as plt

import pybinding as pb
pb.pltutils.use_style()
from math import sqrt

def bilayer():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a/2, a/2*1.732],
                     a2=[-a/2,a/2*1.732])
    lat.add_sublattices(('A', [0, -a_cc]),('B1',[0,0]), ('B2', [0,  a_cc]))
    lat.add_hoppings(
        # inside the main cell
        #([0,  0], 'A', 'B1', t), 
        ([0,  0], 'B1', 'B2', t),
        
        ([0,  0], 'B1', 'A', t), 
        # between neighboring cells
        ([1, 0], 'B1', 'B2', t),
        ([0, -1 ], 'B1', 'B2', t),
        ([1, 0 ], 'B1', 'A', t),
        ([0, 1 ], 'B1', 'A', t)
        )
    return lat

lattice = bilayer()
lattice.plot()

#%%
lat = bilayer()
lat.plot()
plt.show()
lat = bilayer()
#lattice.plot()
lat.plot_brillouin_zone()
plt.show()
model = pb.Model(
    lat,
    pb.translational_symmetry()
) 
#model.plot()
print('hamiltonian:',model.hamiltonian)
print('hamiltonian to dense:',model.hamiltonian.todense())
from math import sqrt, pi

solver = pb.solver.lapack(model)

a_cc = 0.142
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2)
print('eigenvalues:',solver.eigenvalues)
print(' eigenvectors:',solver.eigenvectors)
plt.figure(figsize=(3.841, 7.195), dpi=100)
plot1=bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
print('plot 1:',plot1)
model.lattice.plot_brillouin_zone(decorate=False)
plt.savefig('myfig.png', dpi=1000)
#plot2=bands.plot_kpath(point_labels=['K', r'$\Gamma$', 'M', 'K'])
#print('plot2',plot2)a
