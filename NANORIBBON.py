#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:44:49 2021

@author: harshita
"""
#NANORIBBONS

import pybinding as pb
from math import sqrt
a_cc = 0.142
def monolayer_graphene():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a,0],
                     a2=[a/2, a/2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B2', [0,  a_cc/2]))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B2', t),
        # between neighboring cells
        ([1, -1], 'A', 'B2', t),
        ([0, -1], 'A', 'B2', t)
    )
    return lat

lattice = monolayer_graphene()
lattice.plot()
plt.show()

nr_model = pb.Model(lattice,pb.rectangle(1.2),pb.translational_symmetry(a1=True, a2=False))
nr_model.plot()
plt.show()
nr_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm

solver = pb.solver.lapack(nr_model)
a = a_cc * sqrt(3)  # ribbon unit cell length
#bands = solver.calc_bands(-pi/a, pi/a)
#bands.plot()
#plt.show()

nr2_model = pb.Model(lattice,pb.rectangle(1.2),pb.translational_symmetry(a1=False, a2=True))
nr2_model.plot()
plt.show()
nr2_model.lattice.plot_vectors(position=[0.6, -0.25])

def monolayer_graphene_4atom():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping
    lat = pb.Lattice(a1=[a, 0], a2=[0, 3*a_cc])
    lat.add_sublattices(('A',  [  0, -a_cc/2]),
                        ('B',  [  0,  a_cc/2]),
                        ('A2', [a/2,    a_cc]),
                        ('B2', [a/2,  2*a_cc]))
    lat.add_hoppings(
        # inside the unit sell
        ([0, 0], 'A',  'B',  t),
        ([0, 0], 'B',  'A2', t),
        ([0, 0], 'A2', 'B2', t),
        # between neighbouring unit cells
        ([-1, -1], 'A', 'B2', t),
        ([ 0, -1], 'A', 'B2', t),
        ([-1,  0], 'B', 'A2', t),
    )
    return lat

lattice = monolayer_graphene_4atom()
plt.figure(figsize=(5, 5))
lattice.plot()
plt.show()

model = pb.Model(monolayer_graphene_4atom())
model.plot()
plt.show()
model.lattice.plot_vectors(position=[-0.13, -0.13])


nr3_model = pb.Model(lattice,pb.primitive(a1=5),pb.translational_symmetry(a1=False, a2=True))
nr3_model.plot()
plt.show()
nr3_model.lattice.plot_vectors(position=[-0.59, -0.6])

solver = pb.solver.lapack(nr3_model)
d = 3 * a_cc  # ribbon unit cell length
bands2 = solver.calc_bands([0, -pi/d], [0, pi/d])
bands2.plot(point_labels=['$-\pi / 3 a_{cc}$', '$\pi / 3 a_{cc}$'])
bands2.plot()
plt.show()
#%%import pybinding as pb
from math import sqrt
a_cc = 0.142
def monolayer_graphene():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a,0],
                     a2=[a/2, a/2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B2', [0,  a_cc/2]))
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B2', t),
        # between neighboring cells
        ([1, -1], 'A', 'B2', t),
        ([0, -1], 'A', 'B2', t)
    )
    return lat

lattice = monolayer_graphene()
lattice.plot()
plt.show()

nr_model = pb.Model(lattice,pb.rectangle(1.2),pb.translational_symmetry(a1=True, a2=False))
nr_model.plot()
plt.show()
nr_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm

solver = pb.solver.lapack(nr_model)
a = a_cc * sqrt(3)  # ribbon unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot()
plt.show()

nr2_model = pb.Model(lattice,pb.rectangle(1.2),pb.translational_symmetry(a1=False, a2=True))
nr2_model.plot()
plt.show()
nr2_model.lattice.plot_vectors(position=[0.6, -0.25])

def monolayer_graphene_4atom():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping
    lat = pb.Lattice(a1=[a, 0], a2=[0, 3*a_cc])
    lat.add_sublattices(('A',  [  0, -a_cc/2]),
                        ('B',  [  0,  a_cc/2]),
                        ('A2', [a/2,    a_cc]),
                        ('B2', [a/2,  2*a_cc]))
    lat.add_hoppings(
        # inside the unit sell
        ([0, 0], 'A',  'B',  t),
        ([0, 0], 'B',  'A2', t),
        ([0, 0], 'A2', 'B2', t),
        # between neighbouring unit cells
        ([-1, -1], 'A', 'B2', t),
        ([ 0, -1], 'A', 'B2', t),
        ([-1,  0], 'B', 'A2', t),
    )
    return lat

lattice = monolayer_graphene_4atom()
plt.figure(figsize=(5, 5))
lattice.plot()
plt.show()

model = pb.Model(monolayer_graphene_4atom())
model.plot()
plt.show()
model.lattice.plot_vectors(position=[-0.13, -0.13])


nr3_model = pb.Model(lattice,pb.primitive(a1=5),pb.translational_symmetry(a1=False, a2=True))
nr3_model.plot()
plt.show()
nr3_model.lattice.plot_vectors(position=[-0.59, -0.6])

solver = pb.solver.lapack(nr3_model)
d = 3 * a_cc  # ribbon unit cell length
bands2 = solver.calc_bands([0, -pi/d], [0, pi/d])
bands2.plot(point_labels=['$-\pi / 3 a_{cc}$', '$\pi / 3 a_{cc}$'])
bands2.plot()
plt.show()

def ring(inner_radius, outer_radius):
    """Ring shape defined by an inner and outer radius"""
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)
    return pb.FreeformShape(contains, width=[2*outer_radius, 2*outer_radius])

nr_ring = pb.Model(
    monolayer_graphene_4atom(),
    ring(inner_radius=1.4, outer_radius=2),
    pb.translational_symmetry(a1=3.8, a2=False)
)
plt.figure(figsize=[8, 3])
nr_ring.plot()
plt.show()

solver = pb.solver.arpack(nr_ring, k=20) # only the 20 lowest states
a = 3.8  # [nm] unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot(point_labels=['$-\pi / a$', '$\pi / a$'])
plt.show()