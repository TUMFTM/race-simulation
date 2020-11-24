import numpy as np
import cvxpy as cp


def opt_strategy_basic(tot_no_laps: int,
                       tire_pars: dict,
                       tires: list) -> np.ndarray:

    """
    author:
    Alexander Heilmeier

    date:
    15.11.2019

    .. description::
    If the tire degradation model is linear we get a quadratic optimization problem when trying to find the optimal
    inlaps for a minimal race time. This is solved using ECOS_BB via cvxpy. quadprog does not allow to use integer
    design variables.

    cvxpy interface description taken from
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/cvxpy_.py

    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    calling a given solver using the CVXPY <http://www.cvxpy.org/> modelling
    language.
    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    initvals : array, shape=(n,), optional
        Warm-start guess vector (not used).
    solver : string, optional
        Solver name in ``cvxpy.installed_solvers()``.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    .. inputs::
    :param tot_no_laps:     number of laps in current race
    :type tot_no_laps:      int
    :param tire_pars:       tire model parameters for every compound -> see param file
    :type tire_pars:        dict
    :param tires:           [[compound 1, age 1], [compound 2, age 2], ...]
    :type tires:            list

    .. outputs::
    :return stint_lengths:  optimized stint lengths
    :rtype stint_lengths:   np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP PROBLEM MATRICES ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # the basic idea is to have one design variable per stint and using the equality constraint to assure that the total
    # number of laps of the race is considered (e.g. 2 stops means 3 design variables)

    # set together tire degradation coefficients
    k_1_lin_array = np.array([tire_pars[x[0]]['k_1_lin'] for x in tires])
    k_0_array = np.array([tire_pars[x[0]]['k_0'] for x in tires])
    age_array = np.array([x[1] for x in tires])

    # get number of stints
    no_stints = len(tires)

    # set up problem matrices (P = H and q = f in quadprog)
    P = np.eye(no_stints) * 0.5 * k_1_lin_array * 2  # * 2 because of standard form
    q = (0.5 + age_array) * k_1_lin_array + k_0_array

    G = np.eye(no_stints) * -1.0  # minimum 1 lap per stint
    h = np.ones(no_stints) * -1.0

    A = np.ones((1, no_stints))  # sum of stints must equal total number of laps
    b = np.array([tot_no_laps])

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP SOLVER SPECIFIC PROBLEM -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create integer design variables
    x = cp.Variable(no_stints, integer=True)

    # create quadratic system matrix
    P = cp.Constant(P)

    # set up problem using objective and constraints
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
    constraints = [G @ x <= h, A @ x == b]
    prob = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------------------------------------------------------
    # CALL SOLVER ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ECOS_BB is able to handle mixed integer quadratic problems
    tmp = prob.solve(solver='ECOS_BB')

    if not np.isinf(tmp):
        stint_lengths = np.round(x.value).astype(np.int32)
    else:
        # no solution found
        stint_lengths = None

    return stint_lengths


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass

# tire age not considered in these equations!
# 1 stop 1 DV (no equality constraints -> more complicated, but less solving effort)
# H = np.array([[0.5 * k_11 + 0.5 * k_12]]) * 2  # * 2 because of standard form
# f = np.array([0.5 * k_11 + k_01 - k_12 * tot_no_laps - 0.5 * k_12 - k_02])
# G = np.array([[-1.0], [1.0]])
# h = np.array([-1.0, tot_no_laps])

# tire age not considered in these equations!
# 2 stop 2 DV (no equality constraints -> more complicated, but less solving effort)
# H = np.array([[0.5 * k_11 + 0.5 * k_13, 0.5 * k_13],
#               [0.5 * k_13, 0.5 * k_12 + 0.5 * k_13]]) * 2  # * 2 because of standard form
# f = np.array([0.5 * k_11 + k_01 - k_13 * tot_no_laps - 0.5 * k_13 - k_03,
#               0.5 * k_12 + k_02 - k_13 * tot_no_laps - 0.5 * k_13 - k_03])
# G = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
# h = np.array([-1.0, -1.0, tot_no_laps - 2, tot_no_laps - 2])
