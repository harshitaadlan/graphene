#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:40:56 2021

@author: harshita
"""


#%%

#SQ
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
pb.pltutils.use_style()
import pybinding as pb

d = 0.2  # [nm] unit cell length
t = 1    # [eV] hopping energy

# create a simple 2D lattice with vectors a1 and a2
lattice = pb.Lattice(a1=[d, 0], a2=[0, d])
lattice.add_sublattices(('A', [0, 0]) ) # add an atom called 'A' at position [0, 0])
lattice.add_hoppings(([0, 1], 'A', 'A', t),([1, 0], 'A', 'A', t))# (relative_index, from_sublattice, to_sublattice, energy)
lattice.plot()  # plot the lattice that was just constructed
plt.show()      # standard matplotlib show() function
#%%
def square_lattice(d, t):
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_sublattices(('A', [0, 0]))
    lat.add_hoppings(([0, 1], 'A', 'A', t),
                     ([1, 0], 'A', 'A', t))
    return lat

# we can quickly set a shorter unit length `d`
lattice = square_lattice(d=2, t=0)
lattice.plot()
plt.show()