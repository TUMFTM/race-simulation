from racesim.src.race import Race
import numpy as np
import racesim_basic


class RaceReinftrain(Race):
    """
    author:
    Alexander Heilmeier

    date:
    07.07.2020

    .. description::
    This class is used as a replacement for the normal race class during the training of the Virtual Strategy Engineer
    (VSE) (reinforcement learning approach). It overwrites several methods of the race class such that a lap starts and
    ends "virtually" in front of the pit stop decision in each lap.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 race_pars: dict,
                 driver_pars: dict,
                 car_pars: dict,
                 tireset_pars: dict,
                 track_pars: dict,
                 vse_pars: dict,
                 vse_paths: dict,
                 use_prob_infl: bool,
                 create_rand_events: bool,
                 monte_carlo_pars: dict,
                 event_pars: dict) -> None:

        # initialize original race object
        Race.__init__(self,
                      race_pars=race_pars,
                      driver_pars=driver_pars,
                      car_pars=car_pars,
                      tireset_pars=tireset_pars,
                      track_pars=track_pars,
                      vse_pars=vse_pars,
                      vse_paths=vse_paths,
                      use_prob_infl=use_prob_infl,
                      create_rand_events=create_rand_events,
                      monte_carlo_pars=monte_carlo_pars,
                      event_pars=event_pars)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __simulate_lap_start(self) -> None:
        # increment current lap and copy positions from last lap
        self.cur_lap += 1
        self.positions[self.cur_lap] = self.positions[self.cur_lap - 1]

        # check for pitstop outlaps
        self._Race__handle_pitstop_outlap()

        # calculate current lap time for all drivers
        self._Race__calc_laptimes()

        # handle fcy phases -> increase lap times, forbid overtaking etc. if driver is within a FCY phase
        self._Race__handle_fcy()

        # increase car age (i.e. consider fuel mass loss and tire degradation)
        self._Race__increase_car_age()

        # check for driver retirements (must be done after calculating the lap times to obtain a valid race time
        # estimation)
        self._Race__handle_driver_retirements()

        # check overtaking and modify positions and laptimes according to overtaking time losses
        self._Race__handle_overtaking_track()

        # if VSE (virtual strategy engineer) is used it has to take the strategy decisions here
        self._Race__handle_vse()

    def __simulate_lap(self) -> None:
        # check for pitstop inlaps
        self._Race__handle_pitstop_inlap()

        # perform some actions related to FCY phases after final lap times are known for current lap
        self._Race__fcy_phase_checks_aft_final_laptimes()

        # calculate final racetimes at the end of the current lap
        self.racetimes[self.cur_lap, self.bool_driving[self.cur_lap]] = \
            self.racetimes[self.cur_lap - 1, self.bool_driving[self.cur_lap]] \
            + self.laptimes[self.cur_lap, self.bool_driving[self.cur_lap]]

        # increment current lap and copy positions from last lap
        self.cur_lap += 1
        self.positions[self.cur_lap] = self.positions[self.cur_lap - 1]

        # check for pitstop outlaps
        self._Race__handle_pitstop_outlap()

        # calculate current lap time for all drivers
        self._Race__calc_laptimes()

        # handle fcy phases -> increase lap times, forbid overtaking etc. if driver is within a FCY phase
        self._Race__handle_fcy()

        # increase car age (i.e. consider fuel mass loss and tire degradation)
        self._Race__increase_car_age()

        # check for driver retirements (must be done after calculating the lap times to obtain a valid race time
        # estimation)
        self._Race__handle_driver_retirements()

        # check overtaking and modify positions and laptimes according to overtaking time losses
        self._Race__handle_overtaking_track()

        # if VSE (virtual strategy engineer) is used it has to take the strategy decisions here
        self._Race__handle_vse()

    def __simulate_lap_end(self) -> None:
        # check for pitstop inlaps
        self._Race__handle_pitstop_inlap()

        # perform some actions related to FCY phases after final lap times are known for current lap
        self._Race__fcy_phase_checks_aft_final_laptimes()

        # calculate final racetimes at the end of the current lap
        self.racetimes[self.cur_lap, self.bool_driving[self.cur_lap]] = \
            self.racetimes[self.cur_lap - 1, self.bool_driving[self.cur_lap]] \
            + self.laptimes[self.cur_lap, self.bool_driving[self.cur_lap]]

        # retirements were converted from progress to race time during the simulation -> assure this is set
        self.retire_data["domain"] = 'time'

        # set result status to result available
        if self.result_status == -1:
            self.result_status = 0

        # when race is finished drivers are allowed to finish current lap -> laped drivers will not complete all laps
        self._Race__reset_invalid_laps_aft_race()

        # check plausibility of result
        self._Race__check_plausibility()

    def execute_presim_average_laptimes(self) -> list:
        """
        This method returns the fastest possible average lap times on a free race track for each driver. The average lap
        times are calculated in a pre-simulation that simulates the race without any opponents on the race track and
        with the basic race strategy (fastest possible race strategy) for each driver. FCY phases are deactivated as
        well. The first lap of the pre-simulated race is not considered for the average lap time, because the agent
        cannot influence it. The resulting information can be used for calculating a reward during the training of the
        reinforcement agent.
        """

        average_laptimes = []

        for presim_driver in self.drivers_list:
            # read base strategy
            strategy_info_tmp = self.vse.vse_pars['base_strategy'][presim_driver.initials]

            # perform pre simulation for the driver
            t_race_lapwise = racesim_basic.src.calc_racetimes_basic.\
                calc_racetimes_basic(t_base=(self.track.t_q + self.track.t_gap_racepace + presim_driver.t_driver
                                             + presim_driver.car.t_car),
                                     tot_no_laps=self.race_pars["tot_no_laps"],
                                     t_lap_sens_mass=self.track.t_lap_sens_mass,
                                     t_pitdrive_inlap=self.track.t_pitdrive_inlap,
                                     t_pitdrive_outlap=self.track.t_pitdrive_outlap,
                                     t_pitdrive_inlap_fcy=self.track.t_pitdrive_inlap_fcy,
                                     t_pitdrive_outlap_fcy=self.track.t_pitdrive_outlap_fcy,
                                     t_pitdrive_inlap_sc=self.track.t_pitdrive_inlap_sc,
                                     t_pitdrive_outlap_sc=self.track.t_pitdrive_outlap_sc,
                                     t_pit_tirechange=(self.track.t_pit_tirechange_min
                                     + presim_driver.car.t_pit_tirechange_add),
                                     pits_aft_finishline=self.track.pits_aft_finishline,
                                     tire_pars=presim_driver.tireset_pars,
                                     p_grid=presim_driver.p_grid,
                                     t_loss_pergridpos=self.track.t_loss_pergridpos,
                                     t_loss_firstlap=self.track.t_loss_firstlap,
                                     strategy=strategy_info_tmp,
                                     drivetype=presim_driver.car.drivetype,
                                     m_fuel_init=presim_driver.car.m_fuel,
                                     b_fuel_perlap=presim_driver.car.b_fuel_perlap,
                                     t_pit_refuel_perkg=presim_driver.car.t_pit_refuel_perkg,
                                     t_pit_charge_perkwh=presim_driver.car.t_pit_charge_perkwh,
                                     fcy_phases=[],
                                     t_lap_sc=self.track.t_lap_sc,
                                     t_lap_fcy=self.track.t_lap_fcy)[0]

            # get lap times and save average laptime (excluding the first lap)
            laptimes_aft_first_lap = np.diff(t_race_lapwise)
            average_laptimes.append(float(np.mean(laptimes_aft_first_lap)))

        return average_laptimes
