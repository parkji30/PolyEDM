import os
import numpy as np
from numpy import linalg as lg
from numpy import pi, sin, cos, tan, sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy

def SFSy(y):
    """
    Returns the strong field state along the y-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (-6267) * y**2 - 0.106 * y + 1.018


def SFSx(x):
    """
    Returns the strong field state along the x-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (2.518 * 10**4) * x**2 - 0.05364 * x + 1.021


def WFSy(y):
    """
    Returns the weak field state along the y-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (1.081 * 10**10) * y**4 + (1.635 * 10**5) * y**3 \
    - (1.133 * 10**4) * y**2 - 0.6312 * y + 0.02394


def WFSx(x):
    """
    Returns the weak field state along the x-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (7.657*10**9) * x**4 - (1.166*10**5) * x**3 \
    + (3.603*10**4) * x**2 + 0.2786 * x + 0.03799


def zfield_sin(z, stagescale):
    """
    Returns a sinusodial B-field along the Z axis.

    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return sin(315.3 * z / 2 / stagescale)


def zfield_cos(z, stagescale):
    """
    Returns a cos B-field along the Z axis.

    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return cos(315.3 * z / 2 / stagescale)


def fullfield(x, y, z, stagescale):
    """
    Generates the full magnetic field through calculations from the SFS and WFS
    in the x, y direction.

    @type x: float
    @type y: float
    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (SFSx(x) * SFSy(y) * zfield_sin(z, stagescale)**2 + WFSx(x) \
    * WFSy(y) * zfield_cos(z,stagescale)**2) * bz_scale + bz_mag_offset


def delta(i, j):
    """
    Delta function, returns 1 if i==j else 0.

    @type i, j: int
    @rtype: int
    """
    if i == j:
        return 1
    else:
        return 0


def curve_inner_points(curve_points, j):
    """
    Obtains all the energy values for one specific curve from the entire
    Hamiltonian matrix.

    @type values: Numpy Array
    @rtype: Numpy Array
    """
    list = []
    for i in range(len(curve_points)):
        list.append(curve_points[i][j])
    return np.array(list)


def gs_numerical_hamiltonian_along_z(axis, bfield_along_z):
    """
    This hamiltonian calculation is for the ground state.

    Creates a 3-D array containing the points that were operated on by the
    hamiltonian. The bfields values are specific values for a discrete z-value
    encompassing the XY-YX meshgrid.

     This new array is used as the surface plot for the hamiltonian.

    @type axis: 2D numpy array
    @type bfield_along_z: 2D numpy array
    @rtype: 3D numpy array
    """
    surface_plots = []
    for i in range(np.size(axis)):
        surface_plots.append([])
    for i in range(len(surface_plots)):
        for j in range(len(surface_plots)):
            surface_plots[i].append(list(lg.eigh(H0 + B_scale * \
        bfield_along_z[i][j] * H_int)[0]))
    surface_plots = np.array(surface_plots)
    return surface_plots


def es_numerical_hamiltonian_along_z(axis, bfield_along_z):
    """
    This hamiltonian calculation is for the excited state.

    Creates a 3-D array containing the points that were operated on by the
    hamiltonian. The bfields values are specific values for a discrete z-value
    encompassing the XY-YX meshgrid.

     This new array is used as the surface plot for the hamiltonian.

    @type axis: 2D numpy array
    @type bfield_along_z: 2D numpy array
    @rtype: 3D numpy array
    """
    surface_plots = []
    for i in range(np.size(axis)):
        surface_plots.append([])
    for i in range(len(surface_plots)):
        for j in range(len(surface_plots)):
            surface_plots[i].append(list(lg.eigh(H0_excited + B_scale * \
        bfield_along_z[i][j] * H_int_excited)[0]))
    surface_plots = np.array(surface_plots)
    return surface_plots


def rearrange_hamiltonian_points(surface_points, k):
    """
    Rearranges the points of hamiltonian matrix into an array of points
    corresponding to 1 Energy surface in a 3 Dimensional Axis.

    @type surface_points: Numpy Array
    @type k: integer
    @rtype: Numpy Array
    """
    list = []
    for i in range(len(surface_points)):
        nested_list = []
        for j in range(len(surface_points[i])):
            nested_list.append(surface_points[i][j][k])
        list.append(nested_list)
    return np.array(list)


def list_to_numpy_matrix(xdim, ydim, bvalues):
    """
    Converts a list of b field values into a xdim*ydim numpy matrix.

    @type xdim: List
    @type ydimL List
    @type bvalues: List
    @rtype: numpy array
    """
    dim_list = []
    for i in range(len(xdim)):
        b_list = []
        for j in range(len(ydim)):
            b_list.append(bvalues.pop(0))
        temp_np = np.array(b_list)
        dim_list.append(temp_np)
    return np.array(dim_list)
