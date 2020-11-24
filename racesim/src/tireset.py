import matplotlib.pyplot as plt
import numpy as np
import copy
import helper_funcs.src.calc_tire_degradation


class Tireset(object):
    """
    author:
    Alexander Heilmeier

    date:
    01.11.2017

    .. description::
    XXXX
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__compound",
                 "__age_tot",
                 "__age_curstint",
                 "__age_degr",
                 "__tireset_pars")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, compound: str, age: int, tireset_pars: dict) -> None:
        self.compound = compound                  # compound, e.g. S, SUS, US
        self.age_tot = age                        # [-] tireset age in laps (total)
        self.age_curstint = 0                     # [-] tireset age in laps (current stint)
        self.age_degr = float(age)                # [-] tireset age in laps ("virtual" age for tire deg. calculation)
        self.tireset_pars = tireset_pars          # [-] tireset parameters for current compound (driver-specific)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_compound(self) -> str: return self.__compound

    def __set_compound(self, x: str) -> None:
        if x not in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'I', 'W']:
            raise NameError("Unknown name!", x)
        self.__compound = x
    compound = property(__get_compound, __set_compound)

    def __get_age_tot(self) -> int: return self.__age_tot

    def __set_age_tot(self, x: int) -> None:
        if not 0 <= x < 100:
            raise RuntimeError("Unreasonable value!", x)
        self.__age_tot = x
    age_tot = property(__get_age_tot, __set_age_tot)

    def __get_age_curstint(self) -> int: return self.__age_curstint

    def __set_age_curstint(self, x: int) -> None:
        if not 0 <= x < 100:
            raise RuntimeError("Unreasonable value!", x)
        self.__age_curstint = x
    age_curstint = property(__get_age_curstint, __set_age_curstint)

    def __get_age_degr(self) -> float: return self.__age_degr

    def __set_age_degr(self, x: float) -> None:
        if not 0.0 <= x < 100.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__age_degr = x
    age_degr = property(__get_age_degr, __set_age_degr)

    def __get_tireset_pars(self) -> dict: return self.__tireset_pars
    def __set_tireset_pars(self, x: dict) -> None: self.__tireset_pars = x
    tireset_pars = property(__get_tireset_pars, __set_tireset_pars)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def drive_lap(self, cur_fcy_type: str or None = None, lap_frac_normal: float = 1.0) -> None:
        """
        Increase tire age to take degradation into account. cur_fcy_type contains the FCY phase type ('SC' or 'VSC')
        if an FCY phase is active, otherwise None.
        """

        # tire degradation
        if cur_fcy_type is None:
            self.age_degr += 1.0
        elif cur_fcy_type == 'VSC':
            self.age_degr += lap_frac_normal + (1.0 - lap_frac_normal) * self.tireset_pars["mult_tiredeg_fcy"]
        elif cur_fcy_type == 'SC':
            self.age_degr += lap_frac_normal + (1.0 - lap_frac_normal) * self.tireset_pars["mult_tiredeg_sc"]
        else:
            raise RuntimeError("Unknown FCY phase type!")

        self.age_tot += 1
        self.age_curstint += 1

    def t_add_tireset(self) -> float:
        """
        Calculate additional laptime due to tireset, i.e. degradation and cold tires. Tire parameters must be handed
        over because they are dependent of the driver.
        """

        # calculate current tire degradation
        t_add_tireset = helper_funcs.src.calc_tire_degradation.\
            calc_tire_degradation(tire_age_start=self.age_degr,
                                  stint_length=1,
                                  compound=self.compound,
                                  tire_pars=self.tireset_pars)

        # consider cold tires in first lap of a stint
        if self.age_curstint == 0:
            t_add_tireset += self.tireset_pars["t_add_coldtires"]

        return t_add_tireset

    def plot_tireset_degradation(self, no_laps: int) -> None:
        """This method uses the t_add_tireset method above to make sure to see the same effects as the race
        simulation."""

        # create copy of current tireset
        temp_tireset = copy.deepcopy(self)

        # reset tireset age
        temp_tireset.age_tot = 0
        temp_tireset.age_curstint = 1  # to omit cold tire behavior
        temp_tireset.age_degr = 0.0

        # create array for degradation times
        tireset_deg = np.zeros(no_laps + 1)

        for cur_lap in range(0, no_laps + 1):
            tireset_deg[cur_lap] = temp_tireset.t_add_tireset()
            temp_tireset.drive_lap(cur_fcy_type=None, lap_frac_normal=1.0)

        plt.plot(range(0, no_laps + 1), tireset_deg, ".-")

        plt.gca().invert_yaxis()
        plt.xlim([0, no_laps])
        plt.xlabel("lap")
        plt.ylabel("tireset degradation in s")
        plt.title("tireset degradation over laps")
        plt.grid()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
