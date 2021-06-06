def equation_system(b, time_linspace):
    """
    The ordinary system of equations that will be solved in order to determine
    the position of the molecule based on the b-field.

    @type b: float (bfield)
    @type t: Numpy linspace
    @type: List (change of bfield over time)
    """
    xt, yt, zt, vxt, vyt, vzt = b
    db_over_dt = [vxt,vyt,vzt, B_str * mol_state(xt,yt,zt) * Bdxfn([xt,yt,zt%(20*mm)]),
    B_str * mol_state(xt, yt, zt) * Bdyfn([xt, yt, zt % (20 * mm)]),
    B_str * mol_state(xt, yt, zt) * Bdzfn([xt, yt, zt % (20 * mm)])]
    return db_over_dt


def solve_equations(s0):
    """
    Solves the equation of differential equations based on the initial conditions.

    time_linspace is a global variable that is a numpy linspace.

    @type s0: numpy array
    @rtype: numpy array
    """
    solution = odeint(equation_system, s0, time_linspace)
    return solution
