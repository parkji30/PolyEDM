
def SFS_y(y):
    """
    Returns the strong field state in the y-axis

    @type y: float
    @rtype: float
    """
    return (-6267) * y **2 - 0.106 * y + 1.018


def SFS_x(x):
    """
    Returns the strong field state in the x-axis

    @type x: float
    @rtype: float
    """
    return (2.518 * 10**4) * x**2 - 0.05364 * x + 1.021


def WFS_y(y):
    """
    Returns the weak field state in the y-axis

    @type y: float
    @rtype: float
    """
    return (1.081 * 10**10) * y**4 + (1.635*10**5) * y**3 - \
    (1.133 * 10**4) * y**2 - 0.6312 * y + 0.02394


def WFS_x(x):
    """
    Returns the weak field state in the x-axis

    @type x: float
    @rtype: float
    """
    return (7.657 * 10**9) * x**4 - (1.166 * 10**5) * x**3
    + (3.603 * 10**4) * x**2 + 0.2786 * x + 0.03799


def full_field(x, y, z):
    """
    Magnetic field from ZS-paper archived simulations; B field in [Teslas].
    Returns the magnetic field calculation across the x,y,z  plane.

    @type x: float
    @type y: float
    @type z: float
    @rtype: float
    """
    return SFS_x(x) * SFS_y(y) * sin(315.3*z/2)**2 + \
    WFS_x(x) * WFS_y(y) * cos(315.3 * z/2)**2


def rand_gauss(mean, std):
    """
    This function takes a mean value and a standard deviation value and
    output random values proportional to a gaussian distribution.

    @type mean: float
    @type std: float
    @rtype: float
    """
    return np.random.normal(mean, std)


def rand_gauss_trunc(mean, std, cutoff):
    """
    This function is an extension of the rand_gauss function but truncates values
    at a certain cutoff value. If the output is less than the cutoff, repeat this
    until the output is greater than the cutoff.

    @type mean: float
    @type std: float
    @type cutoff: float
    @rtype: float
    """
    val = rand_gauss(mean,std)
    while val <= cutoff:
        val = rand_gauss(mean,std)
    return val


def sigma_v(T):
    """
    Returns the standard deviation of the velocity in the x and y component.
    Is proportional to 3K by design.

    @type T: float
    @rtype: float
    """
    return sqrt(kb*T/m)


def kinetic_energy(vf, v0):
    """
    Returns the change in kinetic energy by the given intial velocity and final
    velocity.

    @type vf: float
    @type v0: float
    @rtype: float
    """
    dKE = (1/2) * m * ((vf)**2 - (v0)**2)
    return dKE

def mol_state(x, y, z):
    """
    Determines the molecular state of the molecule based on its x, y, z
    coordinate. Changes the state from either wfs to sfs or vice versa.

    @type x: float
    @type y: float
    @type z: float
    @rtype: float
    """
   #if (abs(x) < radius*mm) and (abs(y) < radius*mm) and (z < 100*mm):
    if (z < 120*mm):
        if (0 <= z < 10*mm) \
        or (20*mm <= z < 30*mm) \
        or (40*mm <= z < 50*mm) \
        or (60*mm <= z < 70*mm) \
        or (80*mm <= z < 90*mm) \
        or (100*mm <= z < 110*mm):
            current_state = init_state
        else:
            current_state = -init_state #flips the state
    else:
        current_state = 0
    return current_state