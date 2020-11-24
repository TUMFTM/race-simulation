import numpy as np
import random
import math
import racesim_basic.src.calc_racetimes_basic


class MonteCarlo(object):
    """
    author:
    Alexander Heilmeier

    date:
    11.06.2019

    .. description::
    Dummy class for outsourced methods related to monte carlo simulations.

    The function in this class determines which random events take place during the race. It is called once before the
    actual race is simulated. Random events are car failures and accidents on the driver side and full course yellow
    phases (SC or VSC) on the race side.

    Assumptions:
    1) Accidents always lead to an SC
    2) Failures always lead to a VSC or NOEVENT

    Procedure:
    1) Determine SC phases.
    2) Chose a driver having an accident for every SC phase.
    3) Determine failures among the drivers.
    4) Set VSC phases with a specific chance after every occuring failure.

    Hint:
    The race progress is within the range [0.0, tot_no_laps], where 0.0 corresponds to the start of the first lap.
    """

    def create_random_events(self) -> tuple:

        # initialization
        fcy_data = {"phases": [],
                    "domain": 'progress'}

        retire_data = {"retirements": [None] * self.no_drivers,
                       "domain": 'progress'}

        # --------------------------------------------------------------------------------------------------------------
        # DETERMINE SC PHASES ------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # determine number of SC phases (0 - 3) for the race
        no_sc = random.choices(list(range(0, len(self.monte_carlo_pars["p_sc_quant"]))),
                               self.monte_carlo_pars["p_sc_quant"])[0]

        if no_sc > 0:
            # p_sc_start is a list with 6 individual probabilities: [p_firstlap, p<20%, p<40%, p<60%, p<80%, p<100%]
            probs = np.zeros(self.race_pars["tot_no_laps"])
            no_individual_phases = len(self.monte_carlo_pars["p_sc_start"]) - 1  # -1 to remove start lap probability

            for idx, cur_p in enumerate(self.monte_carlo_pars["p_sc_start"]):
                if idx == 0:
                    # CASE 1: first lap (has its own probability)
                    probs[0] = cur_p
                else:
                    # CASE 2: all other laps (covered by phases in 20% steps)
                    # idx - 1 required because of extra first lap probability
                    cur_start_prog = round(1.0 / no_individual_phases * (idx - 1) * self.race_pars["tot_no_laps"])
                    cur_start_prog = max(cur_start_prog, 1)  # assure start progress is at least second lap
                    cur_end_prog = round(1.0 / no_individual_phases * idx * self.race_pars["tot_no_laps"])
                    cur_duration = cur_end_prog - cur_start_prog  # end prog not included
                    probs[cur_start_prog:cur_end_prog] = cur_p / cur_duration  # distribute SC probability among laps

            # create lists
            choices = list(range(0, self.race_pars["tot_no_laps"]))
            probs = list(probs)

            # determine start and stop race progress for every SC phase
            while len(fcy_data["phases"]) < no_sc:
                # start race progress (including random begin within start lap)
                prog_start = random.choices(choices, probs)[0] + random.random()

                # assure that SC phase is started more than a lap before the end of the race -> after that it makes no
                # more sense to send it on the track since drivers will not catch up until the end
                if prog_start > self.race_pars["tot_no_laps"] - 1.0:
                    continue

                # determine duration of SC phase
                sc_dur = float(random.choices(list(range(1, len(self.monte_carlo_pars["p_sc_duration"]) + 1)),
                                              self.monte_carlo_pars["p_sc_duration"])[0])

                # set stop race progress to a full lap for the SC -> floor since the durations are always over-estimated
                prog_stop = float(math.floor(prog_start + sc_dur))

                if prog_stop > self.race_pars["tot_no_laps"]:
                    prog_stop = float(self.race_pars["tot_no_laps"])

                # append created phase temporarily and remove it again if it intersects another one
                fcy_data["phases"].append([prog_start, prog_stop, 'SC', None, None])

                if self.check_fcyphase_intersection(fcy_data=fcy_data):
                    del fcy_data["phases"][-1]

        # --------------------------------------------------------------------------------------------------------------
        # DETERMINE DRIVER ACCIDENTS -----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if no_sc > 0:
            # get driver idxs and according accident probabilities
            choices = []
            probs = []

            for idx, cur_driver in enumerate(self.drivers_list):
                choices.append(idx)
                probs.append(cur_driver.p_accident)

            # determine one driver per SC phase involved into the accident
            for cur_phase in fcy_data["phases"]:
                # check if there are drivers without an retirement left for the current phase (if we simulate only a
                # small amount of drivers) and break otherwise
                if all(True if x is not None else False for x in retire_data["retirements"]):
                    break

                # chose driver until a "free" driver is selected (who was not already selected for another phase)
                idx_tmp = None

                while idx_tmp is None or retire_data["retirements"][idx_tmp] is not None:
                    idx_tmp = random.choices(choices, probs)[0]

                # save retirement information for selected driver
                retire_data["retirements"][idx_tmp] = cur_phase[0]

        # --------------------------------------------------------------------------------------------------------------
        # DETERMINE DRIVER FAILURES AND VSC PHASES ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        """Driver failures and VSC phases should be determined together. This avoids the situation when a driver failure
        happens during an already existing SC phase and should induce a VSC phase."""

        for idx, cur_driver in enumerate(self.drivers_list):

            # if current driver is already involved in an accident continue to next driver
            if retire_data["retirements"][idx] is not None:
                continue

            # determine failure ----------------------------------------------------------------------------------------
            probs = [1.0 - cur_driver.car.p_failure,
                     cur_driver.car.p_failure]

            failure = random.choices([False, True], probs)[0]

            if failure:
                # determine if VSC appears for current failure ---------------------------------------------------------
                probs = [1.0 - self.monte_carlo_pars["p_vsc_aft_failure"],
                         self.monte_carlo_pars["p_vsc_aft_failure"]]

                vsc = random.choices([False, True], probs)[0]

                # if VSC phase was induced determine according start and stop lap --------------------------------------
                if vsc:

                    # loop until valid VSC phase was found
                    while True:

                        # start race progress determined by a uniform distribution
                        prog_start = random.random() * self.race_pars["tot_no_laps"]

                        # assure that VSC phase is started more than half a lap before the end of the race
                        if prog_start >= self.race_pars["tot_no_laps"] - 0.5:
                            continue

                        # determine duration of VSC phase and stop race progress
                        vsc_dur = random.choices(list(range(1, len(self.monte_carlo_pars["p_vsc_duration"]) + 1)),
                                                 self.monte_carlo_pars["p_vsc_duration"])[0] + random.random()
                        prog_stop = prog_start + vsc_dur

                        if prog_stop > self.race_pars["tot_no_laps"]:
                            prog_stop = float(self.race_pars["tot_no_laps"])

                        # append created phase temporarily and remove it again if it intersects another one
                        fcy_data["phases"].append([prog_start, prog_stop, 'VSC', None, None])

                        if self.check_fcyphase_intersection(fcy_data=fcy_data):
                            del fcy_data["phases"][-1]
                        else:
                            break

                # if VSC was not induced determine failure race progress of the driver ---------------------------------
                else:
                    # start race progress determined by a uniform distribution
                    prog_start = random.random() * self.race_pars["tot_no_laps"]

                # save retirement information for selected driver ------------------------------------------------------
                retire_data["retirements"][idx] = prog_start

        # sort FCY phase list by start race progress when finished
        fcy_data["phases"].sort(key=lambda x: x[0])

        return fcy_data, retire_data

    def check_fcyphase_intersection(self, fcy_data: dict) -> bool:
        """This function checks the inserted FCY data for intersections. In case of 'progress' it checks if the inserted
         phases keep the minimum distance defined in the Monte Carlo parameters. In case of 'time' it checks if the
         inserted phases keep a gap calculate on the basis of the base lap time and the minimum distance defined in the
         Monte Carlo parameters. The function returns True if an intersection exists."""

        # check for intersections (works also for no phase (i.e. empty list) and one phase)
        for idx_1, cur_phase_1 in enumerate(fcy_data["phases"]):
            # set minimum distance based on domain and FCY phase type
            if fcy_data["domain"] == 'progress':
                if cur_phase_1[2] == 'VSC':
                    cur_min_dist = self.monte_carlo_pars["min_dist_vsc"]
                elif cur_phase_1[2] == 'SC':
                    cur_min_dist = self.monte_carlo_pars["min_dist_sc"]
                else:
                    raise RuntimeError("Unknown FCY phase type!")
            elif fcy_data["domain"] == 'time':
                if cur_phase_1[2] == 'VSC':
                    cur_min_dist = self.monte_carlo_pars["min_dist_vsc"] * (self.track.t_q + self.track.t_gap_racepace)
                elif cur_phase_1[2] == 'SC':
                    cur_min_dist = self.monte_carlo_pars["min_dist_sc"] * (self.track.t_q + self.track.t_gap_racepace)
                else:
                    raise RuntimeError("Unknown FCY phase type!")
            else:
                raise RuntimeError("Unknown domain type!")

            # check for intersections with phases coming after the current one in the list
            for cur_phase_2 in fcy_data["phases"][idx_1 + 1:]:
                if cur_phase_2[0] <= cur_phase_1[1] + cur_min_dist \
                        and cur_phase_1[0] - cur_min_dist <= cur_phase_2[1]:
                    # if intersection was found return True
                    return True

        return False

    def convert_raceprog_to_racetimes(self) -> tuple:
        """Perform a pre-simulation to determine approximate FCY (and retirement) race times based on the reference
        driver."""

        # find pre simulation driver in drivers_list, set None if not found
        presim_driver = next((cur_driver for cur_driver in self.drivers_list
                              if cur_driver.initials == self.monte_carlo_pars['ref_driver']), None)

        if presim_driver is None:
            raise RuntimeError('Reference driver was not found, check driver initials!')

        # determine basic strategy if VSE is used
        if self.vse is not None:
            strategy_info_tmp = \
                self.vse.determine_basic_strategy(driver=presim_driver,
                                                  tot_no_laps=self.race_pars["tot_no_laps"],
                                                  fcy_phases=self.fcy_data["phases"],
                                                  location=self.track.name,
                                                  t_pit_tirechange_min=self.track.t_pit_tirechange_min,
                                                  t_pitdrive_inlap=self.track.t_pitdrive_inlap,
                                                  t_pitdrive_outlap=self.track.t_pitdrive_outlap,
                                                  t_pitdrive_inlap_fcy=self.track.t_pitdrive_inlap_fcy,
                                                  t_pitdrive_outlap_fcy=self.track.t_pitdrive_outlap_fcy,
                                                  t_pitdrive_inlap_sc=self.track.t_pitdrive_inlap_sc,
                                                  t_pitdrive_outlap_sc=self.track.t_pitdrive_outlap_sc,
                                                  mult_tiredeg_fcy=presim_driver.tireset_pars["mult_tiredeg_fcy"],
                                                  mult_tiredeg_sc=presim_driver.tireset_pars["mult_tiredeg_sc"])
        else:
            strategy_info_tmp = presim_driver.strategy_info

        # perform pre simulation
        t_race_lapwise_tmp, fcy_phases_tmp = racesim_basic.src.calc_racetimes_basic.\
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
                                 fcy_phases=self.fcy_data["phases"],
                                 t_lap_sc=self.track.t_lap_sc,
                                 t_lap_fcy=self.track.t_lap_fcy)

        t_race_lapwise_tmp = np.insert(t_race_lapwise_tmp, 0, 0.0)  # add race time 0.0s for lap 0

        # replace retirements race progress information by approximate race times
        if self.retire_data["domain"] == 'progress':
            for idx_retirement in range(len(self.retire_data["retirements"])):
                # continue to next if current driver's retirement is None
                if self.retire_data["retirements"][idx_retirement] is None:
                    continue

                # check if current retirement race progress matches any of the FCY phases
                idx_fcyphase = next((i for i, x in enumerate(self.fcy_data["phases"])
                                     if math.isclose(x[0], self.retire_data["retirements"][idx_retirement])), None)

                if idx_fcyphase is not None:
                    # CASE 1: retirement matches a FCY phase -> replace race progress by according race time
                    self.retire_data["retirements"][idx_retirement] = fcy_phases_tmp[idx_fcyphase][0]

                else:
                    # CASE 2: retirement was created without a FCY phase -> interpolate according race time
                    frac_tmp, lap_tmp = math.modf(self.retire_data["retirements"][idx_retirement])
                    lap_tmp = int(lap_tmp)

                    self.retire_data["retirements"][idx_retirement] = \
                        (t_race_lapwise_tmp[lap_tmp]
                         + frac_tmp * (t_race_lapwise_tmp[lap_tmp + 1] - t_race_lapwise_tmp[lap_tmp]))

            self.retire_data["domain"] = 'time'

        # replace FCY phase race progress information by approximate race times
        self.fcy_data["phases"] = fcy_phases_tmp
        self.fcy_data["domain"] = 'time'

        return t_race_lapwise_tmp[-1], strategy_info_tmp

    def check_fcyphase_activation(self, idx_driver: int) -> None:
        # check for the activation of a FCY phase is only required if no FCY phase is active and if there is a FCY phase
        # remaining
        if self.fcy_handling["idxs_act_phase"][idx_driver] is None \
                and self.fcy_handling["idxs_next_phase"][idx_driver] < len(self.fcy_data["phases"]):

            # get required times
            racetime_prev_lap = self.racetimes[self.cur_lap - 1, idx_driver]
            laptime_cur_lap = self.laptimes[self.cur_lap, idx_driver]

            # assure that this function can skip phases that were already passed without activation (e.g. because of a
            # extremly long lap time) which would otherwise block new activations
            while self.fcy_data["phases"][self.fcy_handling["idxs_next_phase"][idx_driver]][1] <= racetime_prev_lap:
                self.fcy_handling["idxs_next_phase"][idx_driver] += 1

                if self.fcy_handling["idxs_next_phase"][idx_driver] >= len(self.fcy_data["phases"]):
                    return

            # check if the FCY phase ends after the start of the current lap and starts before the end of the current
            # lap, i.e. if it somehow affects the current lap
            check_phase = self.fcy_data["phases"][self.fcy_handling["idxs_next_phase"][idx_driver]]

            if racetime_prev_lap < check_phase[1] and check_phase[0] < racetime_prev_lap + laptime_cur_lap:
                # successful activation -> update phase indices
                self.fcy_handling["idxs_act_phase"][idx_driver] = self.fcy_handling["idxs_next_phase"][idx_driver]
                self.fcy_handling["idxs_next_phase"][idx_driver] += 1

    def check_fcyphase_reset(self, idx_driver: int) -> None:
        # check for the reset of a FCY phase if it was active during this lap
        if self.fcy_handling["idxs_act_phase"][idx_driver] is not None:
            cur_phase = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx_driver]]

            # active FCY phase is reseted if it ends until the end of this lap or if it is an SC phase and its duration
            # is reached
            if cur_phase[1] <= self.racetimes[self.cur_lap - 1, idx_driver] + self.laptimes[self.cur_lap, idx_driver] \
                    or (self.fcy_handling["sc_ghost_laps"][idx_driver] is not None
                        and self.fcy_handling["sc_ghost_laps"][idx_driver] >= cur_phase[4]):

                # save actual SC end time for post-processing (end of SC phases differs from pre-calculation if the
                # phase is aborted by the SC duration instead of the end race time or if the leader did not run up to
                # the SC until it ends)
                if cur_phase[2] == 'SC' and self.positions[self.cur_lap, idx_driver] == 1:
                    t_race_sc_end_sim = (self.racetimes[self.cur_lap - 1, idx_driver]
                                         + self.laptimes[self.cur_lap, idx_driver] - self.race_pars['min_t_dist_sc'])
                    self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx_driver]].append(t_race_sc_end_sim)

                # reset FCY phase
                self.fcy_handling["idxs_act_phase"][idx_driver] = None
                self.fcy_handling["sc_ghost_racetimes"][idx_driver] = None
                self.fcy_handling["sc_ghost_laps"][idx_driver] = None
                self.fcy_handling["start_end_prog"][idx_driver] = [None, None]

            # save actual SC end time for post-processing also in the case that the phase will not be reseted because it
            # runs until the end of the race
            elif self.cur_lap == self.race_pars["tot_no_laps"] \
                    and cur_phase[2] == 'SC' \
                    and self.positions[self.cur_lap, idx_driver] == 1 \
                    and math.isinf(cur_phase[1]):
                self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx_driver]].append(math.inf)

    def calc_lapfracs_fcyphase(self, idx_driver: int) -> tuple:
        """
        This method determines the lap fraction driven with normal speed considering FCY phases if active. The method
        check_fcyphase_activation should have been executed within this lap before performing this method!
        """

        # calculate fractions if a phase is active
        if self.fcy_handling["idxs_act_phase"][idx_driver] is not None:
            # get currently active phase
            cur_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx_driver]]

            # get required times
            racetime_prev_lap = self.racetimes[self.cur_lap - 1, idx_driver]
            laptime_cur_lap = self.laptimes[self.cur_lap, idx_driver]

            # get lap fraction driven normally before FCY phase as well as remaining FCY duration
            if racetime_prev_lap <= cur_fcy_phase[0] < racetime_prev_lap + laptime_cur_lap:
                # CASE 1: phase starts within this lap
                lap_frac_normal_bef = (cur_fcy_phase[0] - racetime_prev_lap) / laptime_cur_lap
                remain_dur_fcy = cur_fcy_phase[1] - cur_fcy_phase[0]
            else:
                # CASE 2: phase started already before this lap
                lap_frac_normal_bef = 0.0
                remain_dur_fcy = cur_fcy_phase[1] - racetime_prev_lap

            # calculate remaining FCY lap fraction (can be > 1.0)
            if cur_fcy_phase[2] == 'SC':
                # SC always runs until the end of a lap
                lap_frac_normal_aft = 0.0
            else:
                lap_frac_fcy = remain_dur_fcy / self.track.t_lap_fcy

                # get lap fraction driven normally after a FCY phase
                if lap_frac_normal_bef + lap_frac_fcy >= 1.0:
                    # CASE 1: phase lasts exactly until the end of the current lap or longer
                    lap_frac_normal_aft = 0.0
                else:
                    # CASE 2: phase ends in current lap
                    lap_frac_normal_aft = 1.0 - lap_frac_normal_bef - lap_frac_fcy

            # get total fractions
            lap_frac_normal = lap_frac_normal_bef + lap_frac_normal_aft

            # check lap_frac_normal
            if not 0.0 <= lap_frac_normal <= 1.0:
                raise RuntimeError("lap_frac_normal is not within the range[0,1]!")

        else:
            lap_frac_normal = 1.0
            lap_frac_normal_bef = 1.0

        return lap_frac_normal, lap_frac_normal_bef
