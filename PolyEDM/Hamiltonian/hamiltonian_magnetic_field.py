import numpy as np
from numpy import linalg as lg
from numpy import pi, sin, cos, tan, sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy

def H_rot(A, B):
    '''
    The rotational hamiltonian calculation for the ground state.

    These delta functions all are under the assumption that S (electron and
    I (nucleus) never change, thus we can omit them and only need to worry
    about their particular projections.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy value)
    '''
    H_rot = delta(A.N,B.N) * delta(A.mN,B.mN) \
                           * delta(A.mS,B.mS) \
                           * delta(A.mI,B.mI) \
                           * (B_rot * A.N * (A.N+1))
    return H_rot


def H_hfs(A, B):
    """
    Hamiltonian hyperfine structure for the ground state.
    Through the wigner 3-J calculations.

    The two c_hfs terms come about due to the non-spherical, but present
    cylindrical symmetry of the diatomic representation of YbOh.
    If we choose to ignore them, then the energy levels will behave well
    as if it were atomic energy levels.

    Note, objects A != B.

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # hfs J.I
    H_hfs = 0
    for q in (-1,0,1):
        H_hfs += b_hfs * delta(A.N, B.N) * delta(A.mN, B.mN) * delta(A.mF(), B.mF()) \
        * (-1)**q * (-1)**(A.S - A.mS) * wigner_3j(A.S, 1, B.S, -A.mS, q, B.mS) \
        * (-1)**(A.I - A.mI) * wigner_3j(A.I, 1, B.I, -A.mI, -q,B.mI) \
        * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) * np.sqrt(A.I * (A.I+1) * (2 * A.I + 1)) \

        H_hfs += c_hfs * ((delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mS,B.mS) \
        * delta(A.mI,B.mI) * (A.mS*A.mI)))

        H_hfs += -c_hfs * ((1/3) * delta(A.N,B.N) * delta(A.mN,B.mN) \
        * delta(A.mF(),B.mF()) * (-1)**q * (-1)**(A.S-A.mS) \
        * wigner_3j(A.S,1,B.S,-A.mS,q,B.mS) * (-1)**(A.I-A.mI) \
        * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.S*(A.S+1) * (2*A.S+1)) \
        * np.sqrt(A.I * (A.I+1) * (2*A.I+1)))
    return H_hfs


def H_mag(A, B):
    """
    Hamiltonian magnetic field.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # external B-field S.B
    H_magnetic = delta(A.mI,B.mI) * delta(A.mS,B.mS) * delta(A.mN,B.mN) \
    * delta(A.N,B.N) * (gS*A.mS + gI*A.mI)
    H_magnetic = H_magnetic * -muB
    return H_magnetic


def H_sr(A, B):
    """
    Hamiltonian Spin rotation

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # spin-rotation S.N
    H_sr = 0
    for q in (-1,0,1):
        H_sr += delta(A.mI, B.mI) * delta(A.N, B.N) * delta(A.mF(), B.mF()) * \
        (-1)**q * (-1) ** (A.S - A.mS) * wigner_3j(A.S, 1, B.S, -A.mS, q, B.mS) * \
        (-1)**(A.N - A.mN) * wigner_3j(A.N, 1, B.N, -A.mN, -q, B.mN)

    H_sr = gamma * H_sr * np.sqrt(A.S * (A.S+1) * (2 * A.S+1)) \
    * np.sqrt(A.N * (A.N+1) * (2 * A.N+1))
    return H_sr


def H_rot_excited(A, B):
    '''
    The rotational hamiltonian calculation for the excited state.

    These delta functions all are under the assumption that S (electron and
    I (nucleus) never change, thus we can omit them and only need to worry
    about their particular projections.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy value)
    '''
    # rotational J(J+1)
    H_rot_excited =  delta(A.mI,B.mI) * delta(A.J,B.J) * delta(A.mJ,B.mJ) \
    * (B_rot * A.J * (A.J+1))
    return H_rot_excited


def H_hfs_excited(A, B):
    """
    Hamiltonian hyperfine structure for the excited state.
    Through the wigner 3-J calculations.

    The two c_hfs terms come about due to the non-spherical, but present
    cylindrical symmetry of the diatomic representation of YbOh.
    If we choose to ignore them, then the energy levels will behave well
    as if it were atomic energy levels.

    Note, objects A != B.

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """

    # hfs I.J
    H_hfs_excited = 0
    for q in (-1,0,1):
        H_hfs_excited += b_hfs_excited * delta(A.J,B.J) \
        * delta(A.excited_mF(), B.excited_mF()) * (-1)**q * (-1)**(A.J-A.mJ) \
        * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * (-1)**(A.I-A.mI) \
        * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) \
        * np.sqrt(A.I*(A.I+1)*(2*A.I+1))

        H_hfs_excited += c_hfs_excited \
        * delta(A.J,B.J) * delta(A.mJ,B.mJ) \
        * delta(A.mI,B.mI) * (A.mJ*A.mI)

        H_hfs_excited += -c_hfs_excited * (1/3) * delta(A.J,B.J) \
        * delta(A.excited_mF(), B.excited_mF()) * (-1)**q \
        * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) \
        * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) \
        * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1))
    return H_hfs_excited


def H_mag_excited(A, B):
    """
    Hamiltonian magnetic field for the excited state.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # external B-field S.B
    H_magnetic_excited = delta(A.mI,B.mI) * delta(A.J,B.J) \
    * delta(A.mJ,B.mJ) * (gJ*A.mJ + gI*A.mI)
    H_magnetic_excited = -muB * H_magnetic_excited
    return H_magnetic_excited