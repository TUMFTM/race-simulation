from racesim.src.car import Car


class ElectricCar(Car):
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
        if car_pars["energy"] is None or car_pars["energy_perlap"] is None:
            raise RuntimeError("Parameters energy and energy_perlap must be set for an electric car!")

        # initialize base class object
        Car.__init__(self,
                     car_pars=car_pars,
                     tireset_compound_start=tireset_compound_start,
                     tireset_age_start=tireset_age_start,
                     tireset_pars=tireset_pars)

        # set electric car parameters
        self.drivetype = "electric"
        self.energy = car_pars["energy"]                            # [kWh] energy remaining
        self.energy_perlap = car_pars["energy_perlap"]              # [kWh/lap] energy consumption per lap
        self.t_pit_charge_perkwh = car_pars["t_pit_charge_perkwh"]  # [s/kWh] time per kWh energy added in pit

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def drive_lap(self,
                  cur_fcy_type: str or None = None,
                  lap_frac_normal: float = 1.0,
                  remaining_laps: int = None) -> None:
        """
        Drive_lap method must be called at the end of a lap to be able to consider a new tire. cur_fcy_type contains
        the FCY phase type ('SC' or 'VSC') if an FCY phase is active, otherwise None. Remaining laps must be given if
        automatic adjustment of the consumption is active such that the car runs out of fuel or energy at the end of
        the race. It therefore increases consumption after FCY phases. However, it cannot decrease consumption. There is
        no reserve considered here because the regulations limit the maximal allowed consumption during the race
        independently of the really remaining fuel/energy. Accordingly, the cars will always have a little reserve for
        the inlap after the race is finished in reality.
        """

        # calculate current consumption
        if self.auto_consumption_adjust:
            if remaining_laps is None:
                raise RuntimeError("Remaining laps must be given if automatic consumption adjustment is active!")

            energy_perlap = self.energy / remaining_laps
            energy_perlap = max(energy_perlap, self.energy_perlap)
        else:
            energy_perlap = self.energy_perlap

        # energy consumption
        if cur_fcy_type is None:
            self.energy -= energy_perlap
        elif cur_fcy_type == 'VSC':
            # consumption during FCY phases is calculated on the basis of the "normal" consumption
            self.energy -= (lap_frac_normal + (1.0 - lap_frac_normal) * self.mult_consumption_fcy) * self.energy_perlap
        elif cur_fcy_type == 'SC':
            # consumption during FCY phases is calculated on the basis of the "normal" consumption
            self.energy -= (lap_frac_normal + (1.0 - lap_frac_normal) * self.mult_consumption_sc) * self.energy_perlap
        else:
            raise RuntimeError("Unknown FCY phase type!")

        # print warning if energy is below 0kWh
        if self.energy < 0.0:
            print("WARNING: Remaining energy of this car is negative: %.2fkWh!" % self.energy)

        # assure energy is minimum zero
        if self.energy < 0.0:
            self.energy = 0.0

        # consider tireset degradation
        Car.drive_lap(self, cur_fcy_type=cur_fcy_type, lap_frac_normal=lap_frac_normal)

    def refuel(self, **kwargs) -> None:
        if "energy_add" not in kwargs:
            raise IOError("Missing argument energy_add")

        self.energy += kwargs["energy_add"]

    def t_add_pit_standstill(self, t_pit_tirechange_min: float, use_prob_infl: bool, **kwargs) -> float:
        """
        Return pit standstill time (including a random part if use_prob_infl is True) for tire change and charging.
        """

        # team dependent standstill time for tire change
        t_pit_standstill = Car.t_add_pit_standstill(self,
                                                    t_pit_tirechange_min=t_pit_tirechange_min,
                                                    use_prob_infl=use_prob_infl)

        # energy dependent standstill time
        if self.t_pit_charge_perkwh is not None:

            # raise error if racing series allows recharging but argument was not given
            if "energy_add" not in kwargs:
                raise IOError("Missing argument energy_add")

            # check if recharging lasts longer than standstill time for tire change
            if kwargs["energy_add"] * self.t_pit_charge_perkwh > t_pit_standstill:
                t_pit_standstill = kwargs["energy_add"] * self.t_pit_charge_perkwh

        elif kwargs["energy_add"] != 0.0:
            print("WARNING: Energy was added during pit stop but t_pit_charge_perkwh is not set! Will not consider the"
                  " time loss!")

        return t_pit_standstill


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
