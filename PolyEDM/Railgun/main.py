#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator
import cProfile
import threading
import time
import multiprocessing as mp


## Helper Functions

# Magnetic field equations
# Functions defining the fields along each axis for the SFS and WFS states



## Constants and Terms

# Units
mm = ms = 1e-3
um = us = 1e-6

# Physical quantities
amu = 1.660539040*10**-27 # amu to Kg; NIST
hbar = 1.054571800*10**-34 #J s; NIST
e = 1.6021766208*10**-19 # C; NIST
kb = 1.38064852*10**-23 # J/K
me = 9.10938356*10**-31 # kg; NIST
mYb = 173.045 # amu, for 174Yb, which has an abundance of 31.896%; CIAAW 2015
mO = 15.999 # amu
mH =1.00794 # amu

# Properties of YbOH molecule
mYbOH = (mYb + mO + mH)
m = mYbOH * amu
mu_bohr = (-e*hbar) / (2*me) # J/T

# Terms and Measurements
xterms = yterms = zterms = 400 # ~400 is a good run time, higher the better.
radius = 2.5 # mm
zlen = 20 # mm

#Terms for time
time_last = 5*ms
time_terms = int(1e5) # 10000 measurements of time.
time_linspace = np.linspace(0, time_last, time_terms) # s

# Let mu.B > 0 = WFS state
alpWFS = mu_bohr / m

# Let mu.B < 0 = SFS state
alpSFS = -mu_bohr / m


## Gridding and Linear Spaces

# Generating Linear Space
x = np.linspace(-radius, radius, xterms)*mm
y = np.linspace(-radius, radius, yterms)*mm
z = np.linspace(0,zlen, zterms)*mm
xstepsize = ystepsize = (x.max() - x.min())/len(x) # Assuming uniform stepsize in x,y
zstepsize = (z.max() - z.min())/len(z)

# Produce a 3D grid
XX, YY, ZZ = np.meshgrid(x, y, z, indexing = 'ij', sparse=True)

# Calculating the gradient from the given B-field
B_dx, B_dy, B_dz = np.gradient(full_field(XX, YY, ZZ), xstepsize, ystepsize, zstepsize)

# Interpolate the gradient of the fullfield (Across x, y, z)
Bdxfn = RegularGridInterpolator((x, y, z), B_dx, bounds_error = False, fill_value = 0)
Bdyfn = RegularGridInterpolator((x, y, z), B_dy, bounds_error = False, fill_value = 0)
Bdzfn = RegularGridInterpolator((x, y, z), B_dz, bounds_error = False, fill_value = 0)


## Calculations

# We set the initial state of molecule
init_state = alpWFS
B_str = 1

#import multiprocessing
#multiprocessing.cpu_count()

xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []

if __name__ == "__main__":
    num_particles = mp.cpu_count() * 40 #change the constant to adjust particle count.
    ics = []
    for i in range(num_particles):
        r0 = [rand_gauss(0,0.03*mm), rand_gauss(0,0.03*mm), 0]
        v0 = [rand_gauss(0,0.11456), rand_gauss(0,0.11456), rand_gauss(30,2.5)]
        sum = r0 + v0
        ics.append(sum)
    s0 = np.array(ics)
    print("multiprocessing: \n", end='')
    tstart = time.time()
    p = mp.Pool(mp.cpu_count())
    mp_solutions = p.map(solve_equations, s0)
    tend = time.time()
    tmp = tend - tstart
    print("Total runtime: ", tmp)
    print("Total Particles Created: ", num_particles)

    # multi processing speeds it up about ~4.5 times faster on 6 cores.
    # Took about 61 seconds to compute.
    # serial (single core processing) takes about
    # We can use cloud computing to rent about 8 cores and even more.
    # LOOK IN GPU CODING TO MAKE IT EVEN FASTERRR!!!!!!!

    for sol in mp_solutions:
        xt.append(sol[:,0])
        yt.append(sol[:,1])
        zt.append(sol[:,2])
        vxt.append(sol[:,3])
        vyt.append(sol[:,4])
        vzt.append(sol[:,5])

#%% TRUNCATING PHASE SPACE VECTORS TO BE WITHIN AREA OF THE BORE

xtcut, ytcut, ztcut = [], [], []
vxtcut, vytcut,vztcut = [], [], []

for n in range(num_particles):
    # Vectors such that plotting them will always be within the bore
    xtcut.append(((xt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    ytcut.append(((yt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    ztcut.append(((zt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vxtcut.append(((vxt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vytcut.append(((vyt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vztcut.append(((vzt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)


## PHASE SPACE PLOTS along x and y directions

# Position at initial/final times
pos_initial = 0
pos_final = -1

fig = plt.figure()
ax = fig.add_subplot(111) #1 by 1 by 1 grid.
ax.set_title('$V_x - x$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(xtcut[n][pos_initial]/(3*mm),vxtcut[n][pos_initial]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xtcut[n][pos_final]/(3*mm),vxtcut[n][pos_final]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#   ax.plot(xt[n]/(3*mm),vxt[n]/(sigma_v(3)), alpha = 0.4)
#   ax.axhline(y = vxt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)    # position of the bore
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)   # position of the bore
ax.set_xlabel("$x/\sigma_x$ ")
ax.set_ylabel("$Vx/\sigma_{V_x}$ ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_y - y$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(yt[n][pos_initial]/(3*mm),vyt[n][pos_initial]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_final]/(3*mm),vyt[n][pos_final]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#   ax.plot(yt[n]/(3*mm),vyt[n]/(sigma_v(3)), alpha = 0.4)
#   ax.axhline(y = vyt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")
plt.show()

## Kinetic Energy vs. Z direction

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$\Delta KE_{V_z}$ vs z')
for n in range(len(xt)):
    ax.plot(ztcut[n]*1e3, kinetic_energy(vztcut[n], vztcut[n][0]) / kb)
ax.set_xlabel("z [mm]")
ax.set_ylabel("$KE$ [K]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_z - z$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(ztcut[n][pos_initial] * 1e3,vztcut[n][pos_initial] / (sigma_v(3)), s = 10, color = 'red', alpha = 0.4)
    ax.scatter(ztcut[n][pos_final] * 1e3,vztcut[n][pos_final] / (sigma_v(3)), s = 10, color = 'blue', alpha = 0.4)
    ax.plot(ztcut[n] * 1e3,vztcut[n] / (sigma_v(3)), alpha = 0.4)
ax.set_xlabel("$z$ [mm]")
ax.set_ylabel("$Vz/\sigma_{V_z}$ ")
plt.show()


## HISTOGRAM of Vz Distribution

vz_initial, vz_final = [], []
for n in range(len(xt)):
    vz_initial.append(vzt[n][0])

for n in range(len(xt)):
    vz_final.append(vzt[n][-1])

fig,ax = plt.subplots()
ax.set_title('Initial and final $V_z$ distrubution')
plt.hist(vz_initial,bins=20, normed=True, alpha = 0.4)
plt.hist(vz_final,bins=20, normed=True, alpha = 0.4)
plt.xlim(0,60)
ax.set_xlabel("$Vz$ [m/s]")
ax.set_ylabel("Normalized counts")
plt.show()
