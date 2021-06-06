"""
Created on Sat Apr 27 11:15:12 2019
written by Henry Vu, James Park
"""

# Change directory to where the files are located.



## Python Packages
import os
import numpy as np
from numpy import linalg as lg
from numpy import pi, sin, cos, tan, sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import time

# os.chdir("/home/james/Desktop/Molecular-Beam-Decelerator/Molecular_beam_slowing/Code/Magnetic Field Plot/")
# Change location to wherever textfile_functions exist. (The file is in the code folder as well)
# from textfile_functions import *


## Universal Constants
mm = ms = 1e-3
um = us = 1e-6

stagescale = 20/20
bz_mag_offset = 0.16 # approximately 0.16T if we assume only two stages in the numerical calculation of the Bfield; shift the minimum Bfield value
bz_scale = 0.962 # scale the B field so that the maximum value is approximately 1.16T

B_scale = 1 # Scale the maximum magnitude of the field produced by the analytic Halbach array

# Creating arrays and meshgrids
"""
Change these terms to define the smoothness of the plots.
"""
xterms = 51
yterms = 51
zterms = 41 #change this to 4e3

radius = 10 # mm
zlen = 20 # mm

x = np.linspace(-radius, radius, xterms) * mm
y = np.linspace(-radius, radius, yterms) * mm
z = np.linspace(0, zlen, zterms) * mm

XZ, ZX = np.meshgrid(x, z, indexing='ij', sparse=True)
XY, YX = np.meshgrid(x, y, indexing='ij', sparse=True)

x1 = np.linspace(-radius, radius, xterms) * mm
y1 = np.linspace(-radius, radius, yterms) * mm
z1 = np.linspace(0, zlen, zterms) * mm
e_XZ, e_ZX = np.meshgrid(x1, z1, indexing='ij', sparse=True)


## Ground State Constants
# Molecule constants
B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
gamma= -81.150e-3       # GHz ; spin rotation constant
b_hfs = 4.80e-3  # GHz ; one of the hyperfine constants
c_hfs = 2.46e-3  # GHz ; another hyperfine constant
muB = 14        # GHz/T
gS = -2.0023
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN


## Excited States Constants
muB = 14        # GHz/T
gL = -1
gS = -2.0023
gJ = -0.2#-0.002
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN

wavenum_freq = 100 * (3 * 10**8) * 1e-9 # convert from cm^-1 to GHz
B_rot_excited = B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
b_hfs_excited = 4.80e-3*gJ  # GHz ; one of the hyperfine constants
c_hfs_excited = 2.46e-3*gJ  # GHz ; another hyperfine constant


## Ground State Hamiltonian Calculations
basis = [MolecularState(N=i, I = 1/2, S = 1/2) for i in range(3)]
# expands the basis to contain all the mN, mI, mS sublevels
basis = sublevel_expand(basis)

N = len(basis)

H0 = np.matrix(np.zeros((N, N))) # Create N x N zero matrix
for i in range(N):
    for j in range(N):
        A, B = basis[j], basis[i]
        H0[j, i] = H_rot(A,B)
        H0[j, i] += H_sr(A,B)
        H0[j, i] += H_hfs(A,B)

H_int = np.matrix(np.zeros((N, N)))
for i in range(N):
    for j in range(N):
        A, B = basis[j], basis[i]
        H_int[j, i] = H_mag(A, B)


## Excited State Hamiltonian Calculations
excited_basis = [ExcitedMolecularState(J=(1/2 + i), I=1/2) for i in range(3)]
#expand the basis to contain all the mN, mI, mS sublevels
excited_basis = excited_sublevel_expand(excited_basis)

N_excited = len(excited_basis)

H0_excited = np.matrix(np.zeros((N_excited,N_excited))) # Create N x N zero matrix
for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H0_excited[j,i] = H_rot_excited(A,B)
        H0_excited[j,i] += H_hfs_excited(A,B)

H_int_excited = np.matrix(np.zeros((N_excited, N_excited)))
for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H_int_excited[j,i] = H_mag_excited(A,B)


## Loading Bfield values from text-file

# change the directory to the folder containing the b-field data with the given name below. The file is in this directory as well.
os.chdir("/home/james/Desktop/Molecular-Beam-Decelerator/Molecular_beam_slowing/Code/Magnetic Field Plot/Data")
mag_values = load_txtfile_list("bnorm_actual.txt")

# Creates nested list of b-field values.
bfields = [] #change to lst

z_len = []
for i in range(len(z)):
    bfields.append([]) #change to lst
    z_len.append(i)

loop_range = len(mag_values) // len(z)
for i in range(loop_range):
    for i in z_len:
        popped_value = mag_values.pop(0)
        bfields[i].append(popped_value)


# Bfield values at some Z.
start = list_to_numpy_matrix(x, y, bfields[0])
midpoint = list_to_numpy_matrix(x, y, bfields[20])
end = list_to_numpy_matrix(x, y, bfields[40])


## Plotting Hamiltonian Energy levels with Numerical Bfield Values (GROUND STATE)
# BFIELD VALUES FOR XY---YX AT Z = 0
fig = plt.figure("(Ground State) Hamiltonian of Numerical BField at (Z = 0)")
z0_surface_plots_XY = gs_numerical_hamiltonian_along_z(XY, start)
z0_energy_curves = []
for i in range(36):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    surface_points_XY = rearrange_hamiltonian_points(z0_surface_plots_XY, i)
    z0_energy_curves.append(Energy_Curve(surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Ground State) Hamiltonian of Numerical BField at (Z = 0)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("(Ground State) Energy Levels of Hamiltonian at (Z = 0)")
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[15].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[25].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[35].energy_values)
plt.show()


# BFIELD VALUES FOR XY---YX AT Z = 10
fig = plt.figure("(Ground State) Hamiltonian of Numerical BField at (Z = 10)")
z10_surface_plots_XY = gs_numerical_hamiltonian_along_z(XY, midpoint)
z10_energy_curves = []
for i in range(36):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    z10_surface_points_XY = rearrange_hamiltonian_points(z10_surface_plots_XY, i)
    z10_energy_curves.append(Energy_Curve(z10_surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))
    #ax.plot_surface(XY *1e3, YX *1e3, surface_points_XY)

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Ground State) Hamiltonian of Numerical BField at (Z = 10)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 10)")
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[15].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[25].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[35].energy_values)
plt.show()


## Plotting Hamiltonian Energy levels with Numerical Bfield Values (EXCITED STATE)
# BFIELD VALUES FOR XY---YX AT Z = 0
fig = plt.figure("(Excited_State)Hamiltonian of Numerical BField at (Z = 0)")
z0_surface_plots_XY = es_numerical_hamiltonian_along_z(XY, start)
z0_energy_curves = []
for i in range(24):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    surface_points_XY = rearrange_hamiltonian_points(z0_surface_plots_XY, i)
    z0_energy_curves.append(Energy_Curve(surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Excited State) Hamiltonian of Numerical BField at (Z = 0)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 0)")
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[8].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[14].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[23].energy_values)
plt.show()


# BFIELD VALUES FOR XY---YX AT Z = 10
fig = plt.figure("(Excited State) Hamiltonian of Numerical BField at (Z = 10)")
z10_surface_plots_XY = es_numerical_hamiltonian_along_z(XY, midpoint)
z10_energy_curves = []
for i in range(24):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    z10_surface_points_XY = rearrange_hamiltonian_points(z10_surface_plots_XY, i)
    z10_energy_curves.append(Energy_Curve(z10_surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))
    #ax.plot_surface(XY *1e3, YX *1e3, surface_points_XY)

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Excited State) Hamiltonian of Numerical BField at (Z = 10)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 10)")
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[8].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[14].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[23].energy_values)
plt.show()

