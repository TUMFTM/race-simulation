import numpy as np
import math


def calc_tire_degradation(tire_age_start: int or float,
                          stint_length: int,
                          compound: str,
                          tire_pars: dict) -> np.ndarray or float:

    """
    author:
    Alexander Heilmeier

    date:
    18.11.2019

    .. description::
    This function returns a float or an array containing the tire degradation time delta(s). The function uses either
    the math (stint_length == 1) or numpy library (stint_length > 1) for best performance.

    linear model:       t_tire = k_0 + k_1_lin * age
    quadratic model:    t_tire = k_0 + k_1_quad * age + k_2_quad * age**2
    cubic model:        t_tire = k_0 + k_1_cub * age + k_2_cub * age**2 + k_3_cub * age**3
    logarithmic model:  t_tire = k_0 + k_1_ln * ln(k_2_ln * age + 1)

    .. inputs::
    :param tire_age_start:  tire age in laps at start of current stint
    :type tire_age_start:   int or float
    :param stint_length:    length of current stint
    :type stint_length:     int
    :param compound:        tire compound of current stint
    :type compound:         str
    :param tire_pars:       tire parameters for current driver
    :type tire_pars:        dict

    .. outputs::
    :return t_tire_degr:    tire degradation time delta(s) in seconds
    :rtype t_tire_degr:     np.ndarray or float
    """

    # check input
    if tire_pars["tire_deg_model"] not in ['lin', 'quad', 'cub', 'ln']:
        raise RuntimeError('Unknown tire degradation model!')

    # CASE 1: calculation for a single lap (using math library)
    if stint_length == 1:
        # linear degradation model
        if tire_pars["tire_deg_model"] == 'lin':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_lin'] * tire_age_start)

        # quadratic tire degradation model
        elif tire_pars["tire_deg_model"] == 'quad':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_quad'] * tire_age_start
                           + tire_pars[compound]['k_2_quad'] * math.pow(tire_age_start, 2))

        # cubic tire degradation model
        elif tire_pars["tire_deg_model"] == 'cub':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_cub'] * tire_age_start
                           + tire_pars[compound]['k_2_cub'] * math.pow(tire_age_start, 2)
                           + tire_pars[compound]['k_3_cub'] * math.pow(tire_age_start, 3))

        # logarithmic degradation model
        else:
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_ln'] * math.log(tire_pars[compound]['k_2_ln']
                                                                      * tire_age_start + 1.0))

    # CASE 2: calculation for more than one lap (using numpy library)
    else:
        laps_tmp = np.arange(tire_age_start, tire_age_start + stint_length)

        # linear degradation model
        if tire_pars["tire_deg_model"] == 'lin':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_lin'] * laps_tmp)

        # quadratic tire degradation model
        elif tire_pars["tire_deg_model"] == 'quad':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_quad'] * laps_tmp
                           + tire_pars[compound]['k_2_quad'] * np.power(laps_tmp, 2))

        # cubic tire degradation model
        elif tire_pars["tire_deg_model"] == 'cub':
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_quad'] * laps_tmp
                           + tire_pars[compound]['k_2_quad'] * np.power(laps_tmp, 2)
                           + tire_pars[compound]['k_2_quad'] * np.power(laps_tmp, 3))

        # logarithmic degradation model
        else:
            t_tire_degr = (tire_pars[compound]['k_0']
                           + tire_pars[compound]['k_1_ln'] * np.log(tire_pars[compound]['k_2_ln'] * laps_tmp + 1.0))

    return t_tire_degr


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
