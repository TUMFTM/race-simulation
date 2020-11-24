from racesim.src.tireset import Tireset
import scipy.stats


class Car(object):
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

    __slots__ = ("__manufacturer",
                 "__color",
                 "__tireset",
                 "__t_car",
                 "__drivetype",
                 "__p_failure",
                 "__t_pit_tirechange_add",
                 "__t_pit_var_fisk_pars",
                 "__t_pit_tirechange_add_rand_mean",
                 "__m_fuel",
                 "__b_fuel_perlap",
                 "__t_pit_refuel_perkg",
                 "__energy",
                 "__energy_perlap",
                 "__t_pit_charge_perkwh",
                 "__mult_consumption_sc",
                 "__mult_consumption_fcy",
                 "__auto_consumption_adjust")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, car_pars: dict, tireset_compound_start: str, tireset_age_start: int, tireset_pars: dict) -> None:
        # set car parameters
        self.manufacturer = car_pars["manufacturer"]            # manufacturer, e.g. Mercedes
        self.color = car_pars["color"]                          # team color, e.g. #00D2BE
        self.t_car = car_pars["t_car"]                          # [s] time to add due to car skills
        self.drivetype = "unknown"                              # initialize drivetype, e.g. combustion
        self.p_failure = car_pars["p_failure"]                  # failure probability of car

        # set pitstop related parameters
        self.t_pit_tirechange_add = car_pars["t_pit_tirechange_add"]    # [s] additional standstill time to change tires
        self.t_pit_var_fisk_pars = car_pars["t_pit_var_fisk_pars"]      # fisk distribution parameters [c, loc, scale]
        # [s] mean of the additional standstill time to change tires when considering random influences (i.e. fisk
        # sampling) -> used to limit range of fisk sampling
        self.t_pit_tirechange_add_rand_mean = scipy.stats.fisk.mean(c=self.t_pit_var_fisk_pars[0],
                                                                    loc=self.t_pit_var_fisk_pars[1],
                                                                    scale=self.t_pit_var_fisk_pars[2])

        # reduction of fuel/energy consumption under an FCY phase
        self.mult_consumption_sc = car_pars["mult_consumption_sc"]      # [-] multiplier for consumption under SC
        self.mult_consumption_fcy = car_pars["mult_consumption_fcy"]    # [-] multiplier for consumption under FCY
        # automatic adjustment of fuel/energy consumption such that car runs out of fuel at the end of the race,
        # increases consumption after FCY phases (cannot decrease consumption!)
        self.auto_consumption_adjust = car_pars["auto_consumption_adjust"]

        # combustion and electric parameters (they are set by the specific class as required)
        self.m_fuel = None
        self.b_fuel_perlap = None
        self.t_pit_refuel_perkg = None
        self.energy = None
        self.energy_perlap = None
        self.t_pit_charge_perkwh = None

        # create tireset
        self.tireset = Tireset(compound=tireset_compound_start,
                               age=tireset_age_start,
                               tireset_pars=tireset_pars)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_manufacturer(self) -> str: return self.__manufacturer
    def __set_manufacturer(self, x: str) -> None: self.__manufacturer = x
    manufacturer = property(__get_manufacturer, __set_manufacturer)

    def __get_color(self) -> str: return self.__color
    def __set_color(self, x: str) -> None: self.__color = x
    color = property(__get_color, __set_color)

    def __get_tireset(self) -> Tireset: return self.__tireset
    def __set_tireset(self, x: Tireset): self.__tireset = x
    tireset = property(__get_tireset, __set_tireset)

    def __get_t_car(self) -> float: return self.__t_car

    def __set_t_car(self, x: float) -> None:
        if not 0.0 <= x < 10.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_car = x
    t_car = property(__get_t_car, __set_t_car)

    def __get_drivetype(self) -> str: return self.__drivetype

    def __set_drivetype(self, x: str) -> None:
        if x not in ["electric", "combustion", "unknown"]:
            raise IOError("Unknown drivetype!")
        self.__drivetype = x
    drivetype = property(__get_drivetype, __set_drivetype)

    def __get_p_failure(self) -> float: return self.__p_failure

    def __set_p_failure(self, x: float) -> None:
        if not 0.0 <= x <= 0.5:
            raise RuntimeError("Failure probability seems too high!", x)
        self.__p_failure = x
    p_failure = property(__get_p_failure, __set_p_failure)

    def __get_t_pit_tirechange_add(self) -> float: return self.__t_pit_tirechange_add

    def __set_t_pit_tirechange_add(self, x: float) -> None:
        if not 0.0 < x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pit_tirechange_add = x
    t_pit_tirechange_add = property(__get_t_pit_tirechange_add, __set_t_pit_tirechange_add)

    def __get_t_pit_var_fisk_pars(self) -> list: return self.__t_pit_var_fisk_pars

    def __set_t_pit_var_fisk_pars(self, x: list) -> None:
        if not len(x) == 3:
            raise RuntimeError("Length of required fisk parameters is 3, %i parameters were given!" % len(x))
        self.__t_pit_var_fisk_pars = x
    t_pit_var_fisk_pars = property(__get_t_pit_var_fisk_pars, __set_t_pit_var_fisk_pars)

    def __get_t_pit_tirechange_add_rand_mean(self) -> float: return self.__t_pit_tirechange_add_rand_mean

    def __set_t_pit_tirechange_add_rand_mean(self, x: float) -> None:
        if not 0.0 < x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pit_tirechange_add_rand_mean = x
    t_pit_tirechange_add_rand_mean = property(__get_t_pit_tirechange_add_rand_mean,
                                              __set_t_pit_tirechange_add_rand_mean)

    def __get_m_fuel(self) -> float: return self.__m_fuel

    def __set_m_fuel(self, x: float) -> None:
        if x is not None and not 0.0 <= x <= 115.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__m_fuel = x
    m_fuel = property(__get_m_fuel, __set_m_fuel)

    def __get_b_fuel_perlap(self) -> float: return self.__b_fuel_perlap

    def __set_b_fuel_perlap(self, x: float) -> None:
        if x is not None and not 0.0 < x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__b_fuel_perlap = x
    b_fuel_perlap = property(__get_b_fuel_perlap, __set_b_fuel_perlap)

    def __get_t_pit_refuel_perkg(self) -> float: return self.__t_pit_refuel_perkg

    def __set_t_pit_refuel_perkg(self, x: float) -> None:
        if x is not None and not 0.0 <= x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pit_refuel_perkg = x
    t_pit_refuel_perkg = property(__get_t_pit_refuel_perkg, __set_t_pit_refuel_perkg)

    def __get_energy(self) -> float: return self.__energy

    def __set_energy(self, x: float) -> None:
        if x is not None and not 0.0 < x <= 100.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__energy = x
    energy = property(__get_energy, __set_energy)

    def __get_energy_perlap(self) -> float: return self.__energy_perlap

    def __set_energy_perlap(self, x: float) -> None:
        if x is not None and not 0.0 < x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__energy_perlap = x
    energy_perlap = property(__get_energy_perlap, __set_energy_perlap)

    def __get_t_pit_charge_perkwh(self) -> float: return self.__t_pit_charge_perkwh

    def __set_t_pit_charge_perkwh(self, x: float) -> None:
        if x is not None and not 0.0 <= x < 1000.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pit_charge_perkwh = x
    t_pit_charge_perkwh = property(__get_t_pit_charge_perkwh, __set_t_pit_charge_perkwh)

    def __get_mult_consumption_sc(self) -> float: return self.__mult_consumption_sc

    def __set_mult_consumption_sc(self, x: float) -> None:
        if not 0.0 < x <= 1.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__mult_consumption_sc = x
    mult_consumption_sc = property(__get_mult_consumption_sc, __set_mult_consumption_sc)

    def __get_mult_consumption_fcy(self) -> float: return self.__mult_consumption_fcy

    def __set_mult_consumption_fcy(self, x: float) -> None:
        if not 0.0 < x <= 1.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__mult_consumption_fcy = x
    mult_consumption_fcy = property(__get_mult_consumption_fcy, __set_mult_consumption_fcy)

    def __get_auto_consumption_adjust(self) -> bool: return self.__auto_consumption_adjust
    def __set_auto_consumption_adjust(self, x: bool) -> None: self.__auto_consumption_adjust = x
    auto_consumption_adjust = property(__get_auto_consumption_adjust, __set_auto_consumption_adjust)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def drive_lap(self, cur_fcy_type: str or None = None, lap_frac_normal: float = 1.0, **kwargs) -> None:
        """
        Drive_lap method must be called at the end of a lap to be able to consider a full fuel mass and a new tire.
        cur_fcy_type contains the FCY phase type ('SC' or 'VSC') if an FCY phase is active, otherwise None.
        """

        # tire degradation
        self.tireset.drive_lap(cur_fcy_type=cur_fcy_type, lap_frac_normal=lap_frac_normal)

    def change_tires(self, tireset_compound: str, tireset_age: int, tireset_pars: dict) -> None:
        self.tireset = Tireset(compound=tireset_compound,
                               age=tireset_age,
                               tireset_pars=tireset_pars)

    def t_add_car(self, **kwargs) -> float:
        """
        Calculation of the additional laptime for the car (i.e. fuel and car skills) and tires. kwargs is used for
        t_lap_sens_mass with combustion cars.
        """

        return self.tireset.t_add_tireset() + self.t_car

    def t_add_pit_standstill(self, t_pit_tirechange_min: float, use_prob_infl: bool, **kwargs) -> float:
        """
        Return pit standstill time (including a random part if use_prob_infl is True) for tire change.
        """

        if use_prob_infl:
            # use fisk distribution for pit stop time, limit value to a senseful range of 3 * mean (fisk distribution
            # sometimes leads to very large values)
            t_pit_tirechange_add = None

            while t_pit_tirechange_add is None or t_pit_tirechange_add > 3 * self.t_pit_tirechange_add_rand_mean:
                t_pit_tirechange_add = scipy.stats.fisk.rvs(c=self.t_pit_var_fisk_pars[0],
                                                            loc=self.t_pit_var_fisk_pars[1],
                                                            scale=self.t_pit_var_fisk_pars[2])

            return t_pit_tirechange_min + t_pit_tirechange_add

        else:
            return t_pit_tirechange_min + self.t_pit_tirechange_add


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
