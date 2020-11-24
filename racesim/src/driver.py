from racesim.src.combustioncar import CombustionCar
from racesim.src.electriccar import ElectricCar
from typing import List
import random


class Driver(object):
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

    __slots__ = ("__carno",
                 "__name",
                 "__initials",
                 "__team",
                 "__t_driver",
                 "__strategy_info",
                 "__p_grid",
                 "__car",
                 "__tireset_pars",
                 "__t_teamorder",
                 "__t_lap_var_sigma",
                 "__t_startperf",
                 "__vel_max",
                 "__p_accident",
                 "__lap_influences")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, driver_pars: dict, car_pars: dict, tireset_pars: dict) -> None:
        # inputs
        self.carno = driver_pars["carno"]                       # [-] car number of the driver
        self.name = driver_pars["name"]                         # driver name
        self.initials = driver_pars["initials"]                 # driver initials, e.g. VET
        self.team = driver_pars["team"]                         # team of the driver, e.g. Ferrari
        self.t_driver = driver_pars["t_driver"]                 # [s] time to add due to driver skills
        self.t_lap_var_sigma = driver_pars["t_lap_var_sigma"]   # [s] sigma of gaussian distribution
        self.t_teamorder = driver_pars["t_teamorder"]           # [s] teamorder time modifier
        self.strategy_info = driver_pars["strategy_info"]       # list with inlaps, compounds, tire ages and refueling
        self.p_grid = driver_pars["p_grid"]                     # [-] position in starting grid
        self.tireset_pars = tireset_pars                        # tireset parameters for all used compounds
        self.t_startperf = driver_pars["t_startperf"]           # [s] {"mean", "sigma"} of gaussian distribution
        self.vel_max = driver_pars["vel_max"]                   # [km/h] Max. race speed trap velocity of driver
        self.p_accident = driver_pars["p_accident"]             # [-] accident probability of driver
        self.lap_influences = {}                                # set empty dict to show that no lap is influenced

        if car_pars["drivetype"] == "combustion":
            self.car = CombustionCar(car_pars=car_pars,
                                     tireset_compound_start=driver_pars["strategy_info"][0][1],
                                     tireset_age_start=driver_pars["strategy_info"][0][2],
                                     tireset_pars=tireset_pars)
        elif car_pars["drivetype"] == "electric":
            self.car = ElectricCar(car_pars=car_pars,
                                   tireset_compound_start=driver_pars["strategy_info"][0][1],
                                   tireset_age_start=driver_pars["strategy_info"][0][2],
                                   tireset_pars=tireset_pars)
        else:
            raise IOError("Car drivetype not defined!")

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_carno(self) -> int: return self.__carno

    def __set_carno(self, x: int) -> None:
        if not 0 < x < 100:
            raise RuntimeError("Unreasonable value!", x)
        self.__carno = x
    carno = property(__get_carno, __set_carno)

    def __get_name(self) -> str: return self.__name
    def __set_name(self, x: str) -> None: self.__name = x
    name = property(__get_name, __set_name)

    def __get_initials(self) -> str: return self.__initials
    def __set_initials(self, x: str) -> None: self.__initials = x
    initials = property(__get_initials, __set_initials)

    def __get_team(self) -> str: return self.__team
    def __set_team(self, x: str) -> None: self.__team = x
    team = property(__get_team, __set_team)

    def __get_t_driver(self) -> float: return self.__t_driver

    def __set_t_driver(self, x: float) -> None:
        if not 0.0 <= x < 20.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_driver = x
    t_driver = property(__get_t_driver, __set_t_driver)

    def __get_t_lap_var_sigma(self) -> float: return self.__t_lap_var_sigma

    def __set_t_lap_var_sigma(self, x: float) -> None:
        if not 0.0 <= x < 2.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_lap_var_sigma = x
    t_lap_var_sigma = property(__get_t_lap_var_sigma, __set_t_lap_var_sigma)

    def __get_strategy_info(self) -> List[List]: return self.__strategy_info
    def __set_strategy_info(self, x: List[List]) -> None: self.__strategy_info = x
    strategy_info = property(__get_strategy_info, __set_strategy_info)

    def __get_p_grid(self) -> int: return self.__p_grid

    def __set_p_grid(self, x: int) -> None:
        if not 0 < x < 30:
            raise RuntimeError("Unreasonable value!", x)
        self.__p_grid = x
    p_grid = property(__get_p_grid, __set_p_grid)

    def __get_car(self) -> CombustionCar or ElectricCar: return self.__car
    def __set_car(self, x: CombustionCar or ElectricCar) -> None: self.__car = x
    car = property(__get_car, __set_car)

    def __get_tireset_pars(self) -> dict: return self.__tireset_pars
    def __set_tireset_pars(self, x: dict) -> None: self.__tireset_pars = x
    tireset_pars = property(__get_tireset_pars, __set_tireset_pars)

    def __get_t_teamorder(self) -> float: return self.__t_teamorder
    def __set_t_teamorder(self, x: float) -> None: self.__t_teamorder = x
    t_teamorder = property(__get_t_teamorder, __set_t_teamorder)

    def __get_t_startperf(self) -> dict: return self.__t_startperf
    def __set_t_startperf(self, x: dict) -> None: self.__t_startperf = x
    t_startperf = property(__get_t_startperf, __set_t_startperf)

    def __get_vel_max(self) -> float: return self.__vel_max
    def __set_vel_max(self, x: float) -> None: self.__vel_max = x
    vel_max = property(__get_vel_max, __set_vel_max)

    def __get_p_accident(self) -> float: return self.__p_accident
    def __set_p_accident(self, x: float) -> None: self.__p_accident = x
    p_accident = property(__get_p_accident, __set_p_accident)

    def __get_lap_influences(self) -> dict: return self.__lap_influences
    def __set_lap_influences(self, x: dict) -> None: self.__lap_influences = x
    lap_influences = property(__get_lap_influences, __set_lap_influences)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def calc_basic_timeloss(self, use_prob_infl: bool, t_lap_sens_mass: float) -> float:
        """
        Calculation of the basic time loss without regarding the race situation.
        """

        if use_prob_infl:
            return (self.car.t_add_car(t_lap_sens_mass=t_lap_sens_mass)
                    + self.t_driver
                    + random.gauss(0.0, self.t_lap_var_sigma))
        else:
            return (self.car.t_add_car(t_lap_sens_mass=t_lap_sens_mass)
                    + self.t_driver)

    def update_lap_influences(self, cur_lap: int, influence_type: str):
        if influence_type not in ["pitoutlap", "pitinlap", "sc", "vsc", "retiring"]:
            raise RuntimeError("Unknown influence type %s!" % influence_type)

        if cur_lap not in self.lap_influences:
            self.lap_influences[cur_lap] = [influence_type]
        else:
            self.lap_influences[cur_lap].append(influence_type)


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
