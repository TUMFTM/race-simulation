from racesim.src.car import Car


class CombustionCar(Car):
    """
    author:
    Alexander Heilmeier

    date:
    12.07.2018

    .. description::
    XXXX
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, car_pars: dict, tireset_compound_start: str, tireset_age_start: int, tireset_pars: dict) -> None:
        # check inputs
        if car_pars["m_fuel"] is None or car_pars["b_fuel_perlap"] is None:
            raise RuntimeError("Parameters m_fuel and b_fuel_perlap must be set for an electric car!")

        # initialize base class object
        Car.__init__(self,
                     car_pars=car_pars,
                     tireset_compound_start=tireset_compound_start,
                     tireset_age_start=tireset_age_start,
                     tireset_pars=tireset_pars)

        # set combustion car parameters
        self.drivetype = "combustion"
        self.m_fuel = car_pars["m_fuel"]                            # [kg] fuel mass
        self.b_fuel_perlap = car_pars["b_fuel_perlap"]              # [kg/lap] fuel consumption per lap
        self.t_pit_refuel_perkg = car_pars["t_pit_refuel_perkg"]    # [s/kg] time per kg fuel added in pit

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def drive_lap(self,
                  cur_fcy_type: str or None = None,
                  lap_frac_normal: float = 1.0,
                  remaining_laps: int = None) -> None:
        """
        Drive_lap method must be called at the end of a lap to be able to consider a full fuel mass and a new tire.
        cur_fcy_type contains the FCY phase type ('SC' or 'VSC') if an FCY phase is active, otherwise None. Remaining
        laps must be given if automatic adjustment of the consumption is active such that the car runs out of fuel or
        energy at the end of the race. It therefore increases consumption after FCY phases. However, it cannot decrease
        consumption. There is no reserve considered here because the regulations limit the maximal allowed consumption
        during the race independently of the really remaining fuel/energy. Accordingly, the cars will always have a
        little reserve for the inlap after the race is finished in reality. This is also the case during the qualifying.
        Therefore, the according time loss (in case of fuel mass) is already included in the base lap time.
        """

        # calculate current consumption
        if self.auto_consumption_adjust:
            if remaining_laps is None:
                raise RuntimeError("Remaining laps must be given if automatic consumption adjustment is active!")

            b_fuel_perlap = self.m_fuel / remaining_laps
            b_fuel_perlap = max(b_fuel_perlap, self.b_fuel_perlap)
        else:
            b_fuel_perlap = self.b_fuel_perlap

        # fuel consumption
        if cur_fcy_type is None:
            self.m_fuel -= b_fuel_perlap
        elif cur_fcy_type == 'VSC':
            # consumption during FCY phases is calculated on the basis of the "normal" consumption
            self.m_fuel -= (lap_frac_normal + (1.0 - lap_frac_normal) * self.mult_consumption_fcy) * self.b_fuel_perlap
        elif cur_fcy_type == 'SC':
            # consumption during FCY phases is calculated on the basis of the "normal" consumption
            self.m_fuel -= (lap_frac_normal + (1.0 - lap_frac_normal) * self.mult_consumption_sc) * self.b_fuel_perlap
        else:
            raise RuntimeError("Unknown FCY phase type!")

        # print warning if fuel mass is below 0kg
        if self.m_fuel < 0.0:
            print("WARNING: Remaining fuel mass of this car is negative: %.2fkg!" % self.m_fuel)

        # assure fuel mass is minimum zero
        if self.m_fuel < 0.0:
            self.m_fuel = 0.0

        # consider tireset degradation
        Car.drive_lap(self, cur_fcy_type=cur_fcy_type, lap_frac_normal=lap_frac_normal)

    def refuel(self, **kwargs) -> None:
        if "m_fuel_add" not in kwargs:
            raise IOError("Missing argument m_fuel_add")

        self.m_fuel += kwargs["m_fuel_add"]

    def t_add_car(self, **kwargs) -> float:
        """
        Calculation of the additional laptime for the car (i.e. fuel and car skills) and tires. kwargs is used for
        t_lap_sens_mass with combustion cars.
        """

        if "t_lap_sens_mass" not in kwargs:
            raise IOError("Missing parameter t_lap_sens_mass!")

        return Car.t_add_car(self) + self.m_fuel * kwargs["t_lap_sens_mass"]

    def t_add_pit_standstill(self, t_pit_tirechange_min: float, use_prob_infl: bool, **kwargs) -> float:
        """
        Return pit standstill time (including a random part if use_prob_infl is True) for tire change and refueling.
        """

        # team dependent standstill time for tire change
        t_pit_standstill = Car.t_add_pit_standstill(self,
                                                    t_pit_tirechange_min=t_pit_tirechange_min,
                                                    use_prob_infl=use_prob_infl)

        # fuel dependent standstill time
        if self.t_pit_refuel_perkg is not None:

            # raise error if racing series allows refueling but argument was not given
            if "m_fuel_add" not in kwargs:
                raise IOError("Missing argument m_fuel_add")

            # check if refueling lasts longer than standstill time for tire change
            if kwargs["m_fuel_add"] * self.t_pit_refuel_perkg > t_pit_standstill:
                t_pit_standstill = kwargs["m_fuel_add"] * self.t_pit_refuel_perkg

        elif kwargs["m_fuel_add"] != 0.0:
            print("WARNING: Fuel mass was added during pit stop but t_pit_refuel_perkg is not set! Will not consider"
                  " the time loss!")

        return t_pit_standstill


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
