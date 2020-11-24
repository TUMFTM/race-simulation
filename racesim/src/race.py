# import own modules
from racesim.src.track import Track
from racesim.src.driver import Driver
from racesim.src.vse import VSE

# import general Python modules
import numpy as np
import random
from typing import List, Iterator
import copy
import math

# import method classes that are outsourced to extra files
from racesim.src._race_montecarlo import MonteCarlo
from racesim.src._race_raceanalysis import RaceAnalysis


class Race(MonteCarlo, RaceAnalysis):
    """
    author:
    Alexander Heilmeier

    date:
    01.11.2017

    .. description::
    This class contains all the necessary variables and methods to simulate a race. The central arrays are laptimes,
    racetimes and positions. They all have the form [lap, driver] (lap + 1 because of lap zero = starting grid).

    result_status is an integer indicating if the result is valid or not:
    -1  -> result not available
    0   -> result valid
    1   -> all drivers retired before the end of the race
    10  -> infinite loop in __handle_overtaking_track()
    11  -> summed lap times do not equal the race times
    12  -> positions not plausible
    13  -> minimum distances not kept
    14  -> FCY phases were set but do not appear in the race
    15  -> driver did not use two different compounds during the race
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__cur_lap",               # contains current lap (used as discretization variable)
                 # objects ---------------------------------------------------------------------------------------------
                 "__drivers_list",          # [driver1, driver2, ...] -> list with driver objects
                 "__no_drivers",            # number of drivers attending the race
                 "__track",                 # track object
                 # general parameters/options --------------------------------------------------------------------------
                 "__use_prob_infl",         # boolean to set if probabilistic influences should be activated
                 "__race_pars",             # contains race parameters such as t_overtake, drs_window, ...
                 "__monte_carlo_pars",      # parameters used for monte carlo method
                 "__pit_driver_idxs",       # create list for pitting drivers (set by checking their inlaps)
                 "__pit_outlap_losses",     # create array to save time losses due to pit stop (outlap) for DRS checks
                 # race state ------------------------------------------------------------------------------------------
                 "__laptimes",              # array with laptimes
                 "__racetimes",             # array with racetimes
                 "__positions",             # array with positions
                 "__bool_driving",          # bool array containining which drivers are driving (did not retire)
                 "__progress",              # array with race progress in laps for every driver
                 # fcy related -----------------------------------------------------------------------------------------
                 "__fcy_data",              # dict with FCY data -> {"phases": [[start, end, type, SC delay (SC only),
                                            #                                    SC duration (SC only),
                                            #                                    real end (SC only, added during sim.)],
                                            #                                   [], ...],
                                            #                        "domain": 'time' or 'progress' (for start and end)}
                 "__retire_data",           # dict with retirement data -> {"retirements": [start driver 1, ...],
                                            #                               "domain": 'time' or 'progress' (for start)}
                 "__fcy_handling",          # dict containing everything required to handle the FCY phases correctly
                 "__overtake_allowed",      # bool array containing which drivers are allowed to overtake / be overtaken
                 "__presim_info",           # saves information from the pre-simulation (e.g. race duration)
                 # virtual strategy engineer ---------------------------------------------------------------------------
                 "__vse",                   # ML model handling pit stop decisions
                 # result arrays ---------------------------------------------------------------------------------------
                 "__flagstates",            # list with flag states of the race (with regard to the leader's lap)
                 "__result_status")         # integer indicating if the result is valid or not (and why)

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

        # --------------------------------------------------------------------------------------------------------------
        # CREATE OTHER REQUIRED OBJECTS --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create driver list
        self.drivers_list = []

        for initials in race_pars["participants"]:
            self.drivers_list.append(Driver(driver_pars=driver_pars[initials],
                                            car_pars=car_pars[driver_pars[initials]["team"]],
                                            tireset_pars=tireset_pars[initials]))

        self.drivers_list.sort(key=lambda driver: driver.carno)
        self.no_drivers = len(self.drivers_list)

        # create track object
        self.track = Track(track_pars=track_pars)

        # create VSE (virtual strategy engineer) if indicated
        if vse_paths is None:
            self.vse = None

        else:
            self.vse = VSE(vse_paths=vse_paths,
                           vse_pars=vse_pars)

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZE RACE OBJECT ---------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialize base class objects
        MonteCarlo.__init__(self)
        RaceAnalysis.__init__(self)

        # initialize discretization variable
        self.cur_lap = 0

        # set general parameters
        self.use_prob_infl = use_prob_infl
        self.race_pars = race_pars
        self.race_pars['drs_act_lap'] = [self.race_pars["drs_allow_lap"]] * self.no_drivers
        self.monte_carlo_pars = monte_carlo_pars
        self.pit_driver_idxs = []
        self.pit_outlap_losses = np.zeros(self.no_drivers)

        # create race state arrays (tot_no_laps + 1 is set to include lap 0)
        self.laptimes = np.zeros((self.race_pars["tot_no_laps"] + 1, self.no_drivers))
        self.racetimes = np.zeros((self.race_pars["tot_no_laps"] + 1, self.no_drivers))
        self.positions = np.zeros((self.race_pars["tot_no_laps"] + 1, self.no_drivers), dtype=np.int32)
        self.bool_driving = np.full((self.race_pars["tot_no_laps"] + 1, self.no_drivers), True)
        self.progress = np.zeros(self.no_drivers)

        # create FCY related arrays
        self.fcy_data = copy.deepcopy(event_pars["fcy_data"])  # create copy to avoid changing the original dict
        self.retire_data = copy.deepcopy(event_pars["retire_data"])  # create copy to avoid changing the original dict
        self.fcy_handling = {"sc_ghost_racetimes": [None] * self.no_drivers,
                             "sc_ghost_laps": [None] * self.no_drivers,
                             "idxs_act_phase": [None] * self.no_drivers,
                             "idxs_next_phase": [0] * self.no_drivers,
                             "start_end_prog": [[None, None] for _ in range(self.no_drivers)]}
        self.overtake_allowed = np.full((self.race_pars["tot_no_laps"] + 1, self.no_drivers), True)
        self.presim_info = {"fcy_phases_progress": [],
                            "race_duration": None,
                            "base_strategy_vse": None}

        # create result arrays/lists
        self.flagstates = ["G"] * (self.race_pars["tot_no_laps"] + 1)
        self.result_status = -1  # initialize -1 (result not available)

        # --------------------------------------------------------------------------------------------------------------
        # SET INITIAL CONDITIONS ---------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # set retirements per driver (if set in the parameter file, i.e. a filled list was inserted)
        if self.retire_data["retirements"] is not None and self.retire_data["retirements"]:
            retirements_per_driver = [None] * self.no_drivers

            for cur_retirement in self.retire_data["retirements"]:
                # find current driver index
                idx_driver = next((idx for idx, driver in enumerate(self.drivers_list)
                                   if driver.initials == cur_retirement[0]), None)

                # set according retirement value
                retirements_per_driver[idx_driver] = cur_retirement[1]

            self.retire_data["retirements"] = retirements_per_driver

        # set positions for lap 0 according to starting grid (use sorted indices to handle the case if not all grid
        # positions are set)
        grid_positions = [cur_driver.p_grid for cur_driver in self.drivers_list]
        idxs_sorted = sorted(range(self.no_drivers), key=lambda idx: grid_positions[idx])
        self.positions[0, idxs_sorted] = np.arange(1, self.no_drivers + 1)

        # reset strategy info for every driver if VSE is used -> keep only the start information
        if self.vse is not None:
            for driver in self.drivers_list:
                # adjust start compound basestrategy VSE is chosen for current driver
                if self.vse.vse_pars['vse_type'][driver.initials] == 'basestrategy':
                    driver.strategy_info = [self.vse.vse_pars['base_strategy'][driver.initials][0]]

                # adjust start compound realstrategy VSE is chosen for current driver
                elif self.vse.vse_pars['vse_type'][driver.initials] == 'realstrategy':
                    driver.strategy_info = [self.vse.vse_pars['real_strategy'][driver.initials][0]]

                # use start compound that is defined in the strategy info section (NN VSE)
                else:
                    driver.strategy_info = [driver.strategy_info[0]]

        # --------------------------------------------------------------------------------------------------------------
        # PREPARE FCY PHASES AND RETIREMENTS ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # check for possible intersections between manually inserted FCY phases
        if self.fcy_data["phases"] and self.check_fcyphase_intersection(fcy_data=self.fcy_data):
            raise RuntimeError("Manually inserted FCY phases either intersect or lie too close together (the race"
                               " simulation can only handle one active phase per lap, therefore a minimum distance of"
                               " %.1f laps (SC) and %.1f laps (VSC) is enforced between two phases)!"
                               % (self.monte_carlo_pars["min_dist_sc"], self.monte_carlo_pars["min_dist_vsc"]))

        # create random events such as accidents or car failures if create_rand_events is True and empty lists were
        # given in the parameter file (such events must be determined in front of the actual race simulation)
        if create_rand_events and type(self.fcy_data["phases"]) is list and len(self.fcy_data["phases"]) == 0:
            create_fcyphases = True
        else:
            create_fcyphases = False

        if create_rand_events \
                and type(self.retire_data["retirements"]) is list and len(self.retire_data["retirements"]) == 0:
            create_retirements = True
        else:
            create_retirements = False

        if create_fcyphases or create_retirements:
            fcy_data_tmp, retire_data_tmp = self.create_random_events()

            if create_fcyphases:
                self.fcy_data = fcy_data_tmp
            if create_retirements:
                self.retire_data = retire_data_tmp

        # make sure that fcy_data and retire_data have the correct form if they were not determined in the previous step
        if self.fcy_data["phases"] is None:
            self.fcy_data["phases"] = []

        if self.retire_data["retirements"] is None or not self.retire_data["retirements"]:
            self.retire_data["retirements"] = [None] * self.no_drivers

        """
        If FCY phases are given and their domain is race progress we have to convert the progress information into the
        time domain. This is done using a pre-simulation. In that case, retirements are also converted into the time
        domain if they are given in the progress domain such that they can appear at the same point as an according FCY
        phase. However, if the FCY phases are already given in the time domain the pre-simulation cannot handle them. In
        this case (or if there are no FCY phases given at all) the retirements are not converted into the time domain
        since there is no necessity because they can also be handled in the progress domain.
        """

        if self.fcy_data["domain"] == 'progress' and self.fcy_data["phases"]:
            # save progress information for VSE (required for pre simulation within reinforcement training)
            self.presim_info["fcy_phases_progress"] = copy.deepcopy(self.fcy_data["phases"])

            # convert race progress to race time using a pre simulation
            presim_info_tmp = self.convert_raceprog_to_racetimes()

            # save pre-simulation race duration and base strategy (in case of VSE) for postprocessing
            self.presim_info["race_duration"] = presim_info_tmp[0]
            if self.vse is not None:
                self.presim_info["base_strategy_vse"] = presim_info_tmp[1]

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_cur_lap(self) -> int: return self.__cur_lap

    def __set_cur_lap(self, x: int) -> None:
        if not 0 <= x < 200:
            raise RuntimeError("Unreasonable value!", x)
        self.__cur_lap = x
    cur_lap = property(__get_cur_lap, __set_cur_lap)

    def __get_drivers_list(self) -> List[Driver]: return self.__drivers_list
    def __set_drivers_list(self, x: List[Driver]) -> None: self.__drivers_list = x
    drivers_list = property(__get_drivers_list, __set_drivers_list)

    def __get_no_drivers(self) -> int: return self.__no_drivers

    def __set_no_drivers(self, x: int) -> None:
        if not 0 < x < 30:
            raise RuntimeError("Unreasonable value!", x)
        self.__no_drivers = x
    no_drivers = property(__get_no_drivers, __set_no_drivers)

    def __get_track(self) -> Track: return self.__track
    def __set_track(self, x: Track) -> None: self.__track = x
    track = property(__get_track, __set_track)

    def __get_use_prob_infl(self) -> bool: return self.__use_prob_infl
    def __set_use_prob_infl(self, x: bool) -> None: self.__use_prob_infl = x
    use_prob_infl = property(__get_use_prob_infl, __set_use_prob_infl)

    def __get_race_pars(self) -> dict: return self.__race_pars
    def __set_race_pars(self, x: dict) -> None: self.__race_pars = x
    race_pars = property(__get_race_pars, __set_race_pars)

    def __get_monte_carlo_pars(self) -> dict: return self.__monte_carlo_pars
    def __set_monte_carlo_pars(self, x: dict) -> None: self.__monte_carlo_pars = x
    monte_carlo_pars = property(__get_monte_carlo_pars, __set_monte_carlo_pars)

    def __get_pit_driver_idxs(self) -> List[int]: return self.__pit_driver_idxs
    def __set_pit_driver_idxs(self, x: List[int]) -> None: self.__pit_driver_idxs = x
    pit_driver_idxs = property(__get_pit_driver_idxs, __set_pit_driver_idxs)

    def __get_pit_outlap_losses(self) -> np.ndarray: return self.__pit_outlap_losses
    def __set_pit_outlap_losses(self, x: np.ndarray) -> None: self.__pit_outlap_losses = x
    pit_outlap_losses = property(__get_pit_outlap_losses, __set_pit_outlap_losses)

    def __get_laptimes(self) -> np.ndarray: return self.__laptimes
    def __set_laptimes(self, x: np.ndarray) -> None: self.__laptimes = x
    laptimes = property(__get_laptimes, __set_laptimes)

    def __get_racetimes(self) -> np.ndarray: return self.__racetimes
    def __set_racetimes(self, x: np.ndarray) -> None: self.__racetimes = x
    racetimes = property(__get_racetimes, __set_racetimes)

    def __get_positions(self) -> np.ndarray: return self.__positions
    def __set_positions(self, x: np.ndarray) -> None: self.__positions = x
    positions = property(__get_positions, __set_positions)

    def __get_bool_driving(self) -> np.ndarray: return self.__bool_driving
    def __set_bool_driving(self, x: np.ndarray) -> None: self.__bool_driving = x
    bool_driving = property(__get_bool_driving, __set_bool_driving)

    def __get_progress(self) -> np.ndarray: return self.__progress
    def __set_progress(self, x: np.ndarray) -> None: self.__progress = x
    progress = property(__get_progress, __set_progress)

    def __get_fcy_data(self) -> dict: return self.__fcy_data
    def __set_fcy_data(self, x: dict) -> None: self.__fcy_data = x
    fcy_data = property(__get_fcy_data, __set_fcy_data)

    def __get_retire_data(self) -> dict: return self.__retire_data
    def __set_retire_data(self, x: dict) -> None: self.__retire_data = x
    retire_data = property(__get_retire_data, __set_retire_data)

    def __get_fcy_handling(self) -> dict: return self.__fcy_handling
    def __set_fcy_handling(self, x: dict) -> None: self.__fcy_handling = x
    fcy_handling = property(__get_fcy_handling, __set_fcy_handling)

    def __get_overtake_allowed(self) -> np.ndarray: return self.__overtake_allowed
    def __set_overtake_allowed(self, x: np.ndarray) -> None: self.__overtake_allowed = x
    overtake_allowed = property(__get_overtake_allowed, __set_overtake_allowed)

    def __get_presim_info(self) -> dict: return self.__presim_info
    def __set_presim_info(self, x: dict) -> None: self.__presim_info = x
    presim_info = property(__get_presim_info, __set_presim_info)

    def __get_vse(self) -> VSE: return self.__vse
    def __set_vse(self, x: VSE) -> None: self.__vse = x
    vse = property(__get_vse, __set_vse)

    def __get_flagstates(self) -> List[str]: return self.__flagstates

    def __set_flagstates(self, x: List[str]) -> None:
        for entry in x:
            if entry not in ["G", "Y", "VSC", "SC", "C"]:
                raise RuntimeError("Unknown flagstate %s!" % entry)
        self.__flagstates = x
    flagstates = property(__get_flagstates, __set_flagstates)

    def __get_result_status(self) -> int: return self.__result_status
    def __set_result_status(self, x: int) -> None: self.__result_status = x
    result_status = property(__get_result_status, __set_result_status)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (MAIN METHODS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def simulate_race(self) -> None:
        """
        This is the main method than can be called from outside to simulate a race.
        """

        # --------------------------------------------------------------------------------------------------------------
        # DURING THE RACE ----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # simulate race lap by lap
        while self.cur_lap < self.race_pars["tot_no_laps"]:
            self.__simulate_lap()

        # retirements were converted from progress to race time during the simulation -> assure this is set
        self.retire_data["domain"] = 'time'

        # set result status to result available
        if self.result_status == -1:
            self.result_status = 0

        # --------------------------------------------------------------------------------------------------------------
        # AFTER THE RACE -----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # when race is finished drivers are allowed to finish current lap -> laped drivers will not complete all laps
        self.__reset_invalid_laps_aft_race()

        # check plausibility of result
        self.__check_plausibility()

    def __simulate_lap(self) -> None:
        """
        This method is the mainly used method within the race class. It performs all the necessary steps to simulate
        a lap. At first the positions are copied from last lap and the base laptime (= best possible laptime) is set for
        every driver. Afterwards it is checked, if pitstops occur between this lap and last lap. Before calculating the
        basic timeloss without taking into account the race situation (due to tire degradation etc.) for every driver,
        the age of tires and the fuel consumption is updated by one lap. Subsequently the timeloss due to the race-
        situation is calculated, e.g. due to yellow flags. Finally, the calculated laptime is added to get the total
        racetime. It is used as a basis for the overtaking method.
        """

        # increment current lap and copy positions from last lap
        self.cur_lap += 1
        self.positions[self.cur_lap] = self.positions[self.cur_lap - 1]

        # check for pitstop outlaps
        self.__handle_pitstop_outlap()

        # calculate current lap time for all drivers
        self.__calc_laptimes()

        # handle fcy phases -> increase lap times, forbid overtaking etc. if driver is within a FCY phase
        self.__handle_fcy()

        # increase car age (i.e. consider fuel mass loss and tire degradation)
        self.__increase_car_age()

        # check for driver retirements (must be done after calculating the lap times to obtain a valid race time
        # estimation)
        self.__handle_driver_retirements()

        # check overtaking and modify positions and laptimes according to overtaking time losses
        self.__handle_overtaking_track()

        # if VSE (virtual strategy engineer) is used it has to take the strategy decisions here
        self.__handle_vse()

        # check for pitstop inlaps
        self.__handle_pitstop_inlap()

        # perform some actions related to FCY phases after final lap times are known for current lap
        self.__fcy_phase_checks_aft_final_laptimes()

        # calculate final racetimes at the end of the current lap
        self.racetimes[self.cur_lap, self.bool_driving[self.cur_lap]] = \
            self.racetimes[self.cur_lap - 1, self.bool_driving[self.cur_lap]] \
            + self.laptimes[self.cur_lap, self.bool_driving[self.cur_lap]]

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (RACE SIMULATION PARTS) ----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __handle_pitstop_outlap(self) -> None:
        """
        The pitstop method at first loops through all the pitting drivers, calls the change tires method and adds the
        pit timeloss to the current laptime. A virtual racetime consisting of the total racetimes after last lap and the
        timelosses due to pit is used to reorder the driver positions afterwards. Therefore no timeloss due to
        overtaking is added.

        It is assumed that drivers always complete their pitstop if they were driving in the last lap and retire in the
        current lap. Therefore, position changes and minimum distance check after the pitstops are performed based on a
        temporary racetimes array.
        """

        # reset pit outlap time losses
        self.pit_outlap_losses = np.zeros(self.no_drivers)

        # continue only if there are drivers that drove into the pit last lap (this also prevents the function from
        # being executed in the first lap)
        if self.pit_driver_idxs:
            # create temporary array containing the pit timelosses of the drivers
            timelosses_pit = np.zeros(self.no_drivers)

            # loop through all pitting drivers
            for idx in self.pit_driver_idxs:
                # update lap_influences dict of affected driver
                self.drivers_list[idx].update_lap_influences(cur_lap=self.cur_lap, influence_type='pitoutlap')

                # if pits are located after the finish line add standstill time loss to outlap here
                if self.track.pits_aft_finishline:
                    timelosses_pit[idx] += self.__perform_pitstop_standstill(idx_driver=idx, inlap=self.cur_lap - 1)

                # determine pit driving outlap time loss -> modified time loss if it happens under an active FCY phase
                t_pitdrive_outlap_tmp = self.track.t_pitdrive_outlap

                if self.fcy_handling["idxs_act_phase"][idx] is not None:
                    check_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]]
                elif self.fcy_handling["idxs_next_phase"][idx] < len(self.fcy_data["phases"]):
                    check_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_next_phase"][idx]]
                else:
                    check_fcy_phase = None

                if check_fcy_phase is not None:
                    if check_fcy_phase[2] == 'VSC' \
                            and (check_fcy_phase[0] <= self.racetimes[self.cur_lap - 1, idx]) \
                            and (self.racetimes[self.cur_lap - 1, idx] + timelosses_pit[idx]
                                 + self.track.t_pitdrive_outlap_fcy < check_fcy_phase[1]):
                        # CASE 1: VSC phase (considered if the FCY phase fully covers the pit stop outlap part)
                        t_pitdrive_outlap_tmp = self.track.t_pitdrive_outlap_fcy

                    elif check_fcy_phase[2] == 'SC' \
                            and check_fcy_phase[0] <= self.racetimes[self.cur_lap - 1, idx] < check_fcy_phase[1]:
                        # CASE 2: SC phase (considered if the FCY phase starts before or at the beginning of the current
                        # lap and if it ends after the start of the current lap, i.e. if it somehow reaches into the
                        # current lap (since SC stays until the end of the lap no matter if the FCY phase ends earlier))

                        if check_fcy_phase[0] < self.racetimes[self.cur_lap - 2, idx]:
                            # CASE 2a: SC phase started before the last lap already -> drivers should have run up to it
                            t_pitdrive_outlap_tmp = self.track.t_pitdrive_outlap_sc
                        else:
                            # CASE 2b: SC phase started within the pit inlap -> no driver ran up to the SC already
                            # -> time loss is equal to a FCY phase instead
                            t_pitdrive_outlap_tmp = self.track.t_pitdrive_outlap_fcy

                # save t_pitdrive part
                timelosses_pit[idx] += t_pitdrive_outlap_tmp

            # add timelosses to current laptimes
            self.laptimes[self.cur_lap] += timelosses_pit

            # check for position changes (without overtaking timeloss) to sort the driver positions and assure minimum
            # distances after pitstops (both using the timelosses that were determined before)
            self.__check_pos_changes_wo_timeloss(t_lap_tmp=timelosses_pit)
            self.__assure_min_dists(t_lap_tmp=timelosses_pit)

            # save pit outlap time losses (required for DRS checks later)
            self.pit_outlap_losses = np.copy(self.laptimes[self.cur_lap])

    def __perform_pitstop_standstill(self, idx_driver: int, inlap: int) -> float:
        """This method returns the standstill time loss during a pit stop that is caused by tire change and refueling.
        The time loss must be added to the inlap or outlap lap time based on the location of the finish line."""

        # get relevant pitstop of currently pitting driver
        rel_pitstop = next((pitstop for pitstop in self.drivers_list[idx_driver].strategy_info if inlap == pitstop[0]),
                           None)

        # perform pit stop (depending on the drivetype)
        if self.drivers_list[idx_driver].car.drivetype == 'combustion':
            # set new tire with correct compound and age
            self.drivers_list[idx_driver].car.change_tires(tireset_compound=rel_pitstop[1],
                                                           tireset_age=rel_pitstop[2],
                                                           tireset_pars=self.drivers_list[idx_driver].tireset_pars)

            # refuel
            self.drivers_list[idx_driver].car.refuel(m_fuel_add=rel_pitstop[3])

            # calculate timeloss due to pitstop
            timeloss_standstill = self.drivers_list[idx_driver].car. \
                t_add_pit_standstill(use_prob_infl=self.use_prob_infl,
                                     m_fuel_add=rel_pitstop[3],
                                     t_pit_tirechange_min=self.track.t_pit_tirechange_min)

        elif self.drivers_list[idx_driver].car.drivetype == 'electric':
            # set new tire with correct compound and age
            self.drivers_list[idx_driver].car.change_tires(tireset_compound=rel_pitstop[1],
                                                           tireset_age=rel_pitstop[2],
                                                           tireset_pars=self.drivers_list[idx_driver].tireset_pars)

            # recharge
            self.drivers_list[idx_driver].car.refuel(energy_add=rel_pitstop[3])

            # calculate timeloss due to pitstop
            timeloss_standstill = self.drivers_list[idx_driver].car. \
                t_add_pit_standstill(use_prob_infl=self.use_prob_infl,
                                     energy_add=rel_pitstop[3],
                                     t_pit_tirechange_min=self.track.t_pit_tirechange_min)

        else:
            raise RuntimeError("Unknown drivetype!")

        return timeloss_standstill

    def __calc_laptimes(self) -> None:
        """
        This methods update the lap time of every driver taking into account tire degradation, fuel mass loss, the
        current race situation (DRS, flags, ...) etc.
        """

        # --------------------------------------------------------------------------------------------------------------
        # SET BASE LAP TIME --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.laptimes[self.cur_lap, self.bool_driving[self.cur_lap]] += self.track.t_q + self.track.t_gap_racepace

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER DRIVER SPECIFIC TIME DELTAS -------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        for idx in self.__get_driver_iter():

            # ----------------------------------------------------------------------------------------------------------
            # BASIC TIME LOSSES (INCLUDING CAR AND DRIVER CAPABILITIES AS WELL AS FUEL MASS LOSS AND TIRE DEGRADATION) -
            # ----------------------------------------------------------------------------------------------------------

            self.laptimes[self.cur_lap, idx] += \
                self.drivers_list[idx].calc_basic_timeloss(use_prob_infl=self.use_prob_infl,
                                                           t_lap_sens_mass=self.track.t_lap_sens_mass)

            # ----------------------------------------------------------------------------------------------------------
            # STARTING GRID AND START TIME LOSS ------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            if self.cur_lap == 1:

                # timeloss at race start due to start from standstill
                self.laptimes[self.cur_lap, idx] += self.track.t_loss_firstlap

                # timeloss due to grid position (- 1 because first grid position has no time loss)
                self.laptimes[self.cur_lap, idx] += (self.drivers_list[idx].p_grid - 1) * self.track.t_loss_pergridpos

                # add a random part for race start
                if self.use_prob_infl:
                    self.laptimes[self.cur_lap, idx] += \
                        random.gauss(self.drivers_list[idx].t_startperf["mean"],
                                     self.drivers_list[idx].t_startperf["sigma"])

            # ----------------------------------------------------------------------------------------------------------
            # DRS ------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            if self.race_pars["use_drs"] and self.cur_lap >= self.race_pars["drs_act_lap"][idx]:
                # we assume that the first half of the DRS gain happens on the start finish straight, which is why we
                # base the check on the race state before the pit stop outlap handling
                if self.positions[self.cur_lap - 1, idx] > 1:
                    # get bool array with driver in front of current driver at the end of the previous lap
                    pos_front_b = self.positions[self.cur_lap - 1] == self.positions[self.cur_lap - 1, idx] - 1

                    # skip DRS check if driver in front was in the pit lane
                    pos_front_idx = np.argmax(pos_front_b)
                    if pos_front_idx in self.pit_driver_idxs:
                        continue

                    # activate DRS (half effect) if driver was within the DRS window at the end of the previous lap
                    if self.racetimes[self.cur_lap - 1, idx] - self.racetimes[self.cur_lap - 1, pos_front_b] \
                            <= self.race_pars["drs_window"]:
                        # t_drseffect is negative
                        self.laptimes[self.cur_lap, idx] += self.track.t_drseffect / 2.0

                # we assume that the second half of the DRS gain happens elsewhere on the track, which is why we base
                # the check on the race state after the pit stop outlap handling
                if self.positions[self.cur_lap, idx] > 1:
                    # get bool array with driver in front of current driver after the pit stop outlap handling
                    pos_front_b = self.positions[self.cur_lap] == self.positions[self.cur_lap, idx] - 1

                    # activate DRS (half effect) if driver is within the DRS window after the pit stop outlap handling
                    if (self.racetimes[self.cur_lap - 1, idx] + self.pit_outlap_losses[idx]) \
                            - (self.racetimes[self.cur_lap - 1, pos_front_b] + self.pit_outlap_losses[pos_front_b]) \
                            <= self.race_pars["drs_window"]:
                        # t_drseffect is negative
                        self.laptimes[self.cur_lap, idx] += self.track.t_drseffect / 2.0

    def __handle_fcy(self) -> None:
        """
        This method is used to handle FCY phases correctly within the lap discretized race simulation.

        The main problem with FCY phases is that they appear at a distinct point in time while we simulate lap per lap
        and therefore some slower drivers might still be in already simulated laps when the FCY appears in a given
        leader lap. The approach to  overcome this is to calculate the racetimes at which every FCY phase starts and
        ends instead of using distinct laps. These race times can be considered within the lap discretized simulation.

        In case of a FCY the lap times of the drivers increase to about 140% of a normal lap. However, when driving
        directly behind an SC the lap time is about 160% of a normal lap. To be able to simulate the race cars running
        up to an SC, ghost cars are used. These cars are considered blocking (basically like other drivers when
        overtaking is not allowed) and are set individually for every driver. This concept prevents a lapped driver
        from passing the leading drivers during SC phases (which would be possible otherwise because the leading
        drivers drive with 160% and the lapped drivers with 140% of a normal lap time and overtaking is only blocked
        for the opponents directly in front).
        """

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER FCY INDUCED LAP TIME INCREASE AND SAFETY CAR (CONSIDERED AS GHOST CAR) ------------------------------
        # --------------------------------------------------------------------------------------------------------------

        for idx in self.__get_driver_iter():
            # perform FCY phase activation
            self.check_fcyphase_activation(idx_driver=idx)

            # in case of an active FCY phase perform the further steps
            if self.fcy_handling["idxs_act_phase"][idx] is not None:
                # ------------------------------------------------------------------------------------------------------
                # FCY PHASE PART ---------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                # get current phase information
                cur_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]]

                # update lap_influences dict of affected driver
                self.drivers_list[idx].update_lap_influences(cur_lap=self.cur_lap,
                                                             influence_type=cur_fcy_phase[2].lower())

                # determine lap fraction driven normally
                lap_frac_normal, lap_frac_normal_bef = self.calc_lapfracs_fcyphase(idx_driver=idx)

                # set FCY lap time for current driver
                laptime_tmp = (lap_frac_normal * self.laptimes[self.cur_lap, idx]
                               + (1.0 - lap_frac_normal) * self.track.t_lap_fcy)

                if self.laptimes[self.cur_lap, idx] < laptime_tmp:
                    self.laptimes[self.cur_lap, idx] = laptime_tmp

                # determine if overtaking is allowed in current lap for current driver
                if lap_frac_normal < 0.5 or cur_fcy_phase[2] == 'SC':
                    # an SC always leads to forbidden overtaking for the whole lap (this would cause problems since
                    # the safety car is handled before the overtaking and therefore positions changes could lead to
                    # wrong minimum distances etc.), for VSC it is forbidden if at least half of the lap is affected
                    self.overtake_allowed[self.cur_lap, idx] = False

                # set start and end progress information (used for VSE)
                if self.fcy_handling["start_end_prog"][idx][0] is None:
                    # start progress must only be set if phase was newly started
                    self.fcy_handling["start_end_prog"][idx][0] = self.cur_lap - 1.0 + lap_frac_normal_bef

                self.fcy_handling["start_end_prog"][idx][1] = self.cur_lap - (lap_frac_normal - lap_frac_normal_bef)

                # ------------------------------------------------------------------------------------------------------
                # SAFETY CAR PART --------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                if cur_fcy_phase[2] == 'SC':
                    # set ghost car race time (the same for all drivers in the same lap, minimum distance between
                    # positions is considered afterwards) and lap counter
                    if self.fcy_handling["sc_ghost_racetimes"][idx] is None:
                        self.fcy_handling["sc_ghost_racetimes"][idx] = cur_fcy_phase[0]
                        self.fcy_handling["sc_ghost_laps"][idx] = 0  # 0 equals start lap
                    else:
                        self.fcy_handling["sc_ghost_racetimes"][idx] += self.track.t_lap_sc
                        self.fcy_handling["sc_ghost_laps"][idx] += 1

                        # the SC ghost lap time is increased in its first lap such that the driver field runs up quicker
                        # the according information was stored in the FCY phase after the pre simulation (SC delay)
                        if self.fcy_handling["sc_ghost_laps"][idx] == 1:
                            self.fcy_handling["sc_ghost_racetimes"][idx] += cur_fcy_phase[3]

                    # increase the laptime if a driver would pass his designated position behind the ghost car (minimum
                    # distance to ghost car / front driver is considered on the basis of the current position)
                    racetime_incl_fcy = self.racetimes[self.cur_lap - 1, idx] + self.laptimes[self.cur_lap, idx]
                    sc_ghost_incl_min_dist = (self.fcy_handling["sc_ghost_racetimes"][idx]
                                              + self.positions[self.cur_lap, idx] * self.race_pars['min_t_dist_sc'])

                    if racetime_incl_fcy < sc_ghost_incl_min_dist:
                        self.laptimes[self.cur_lap, idx] = (sc_ghost_incl_min_dist
                                                            - self.racetimes[self.cur_lap - 1, idx])

                    # DRS usage is not allowed directly after an SC phase (+ 1 because current lap is still SC)
                    if self.race_pars["use_drs"]:
                        self.race_pars["drs_act_lap"][idx] = self.cur_lap + 1 + self.race_pars["drs_sc_delay"]

    def __increase_car_age(self) -> None:
        """
        Increase car age (fuel mass loss and tire degradation). Call this method only once after timeloss calculation.
        This must be done after FCY phase handling since fuel consumption and tire degradation are reduced during a FCY
        phase.
        """

        for idx in self.__get_driver_iter():
            if self.fcy_handling["idxs_act_phase"][idx] is None:
                cur_fcy_type = None
                lap_frac_normal = 1.0
            else:
                cur_fcy_type = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]][2]
                lap_frac_normal = self.calc_lapfracs_fcyphase(idx_driver=idx)[0]

            # remaining_laps must be inserted on the state before this lap to calculate the correct fuel consumption
            # (if the automatic consumption adjustment is activated)
            self.drivers_list[idx].car.drive_lap(cur_fcy_type=cur_fcy_type,
                                                 lap_frac_normal=lap_frac_normal,
                                                 remaining_laps=self.race_pars["tot_no_laps"] - (self.cur_lap - 1))

    def __handle_driver_retirements(self) -> None:
        """
        This method checks for driver retirements in the current lap. If a driver retires he must be sorted to the end
        using check_pos_changes_wo_timeloss.
        """

        # calculate racetimes based on the temporary laptimes (only required if retirements are in the time domain)
        if self.retire_data["domain"] == 'time':
            racetimes_tmp = self.racetimes[self.cur_lap - 1] + self.laptimes[self.cur_lap]
        else:
            racetimes_tmp = None

        # if driver retires update bool_driving and set estimated lap progress, else set lap progress until end of lap
        for idx in self.__get_driver_iter():
            if self.retire_data["retirements"][idx] is not None \
                    and ((self.retire_data["domain"] == 'time'
                          and racetimes_tmp[idx] > self.retire_data["retirements"][idx])
                         or (self.retire_data["domain"] == 'progress'
                             and self.cur_lap > self.retire_data["retirements"][idx])):

                # update bool_driving
                self.bool_driving[self.cur_lap:, idx] = False

                # update lap_influences dict of affected driver
                self.drivers_list[idx].update_lap_influences(cur_lap=self.cur_lap, influence_type='retiring')

                # set estimated lap progress until retirement based on the currently estimated lap time or the progress
                if self.retire_data["domain"] == 'time':
                    progress_tmp = (self.retire_data["retirements"][idx]
                                    - self.racetimes[self.cur_lap - 1, idx]) / self.laptimes[self.cur_lap, idx]
                else:
                    progress_tmp = math.modf(self.retire_data["retirements"][idx])[0]

                    # convert retirement data domain to race time
                    self.retire_data["retirements"][idx] = (self.racetimes[self.cur_lap - 1, idx]
                                                            + progress_tmp * self.laptimes[self.cur_lap, idx])

                # if final race time of the previous lap changed after the execution of this function it could happen
                # that a driver should have retired in the last lap already leading to a negative progress calculation
                self.progress[idx] += max(progress_tmp, 0.0)

            else:
                # driver will finish the current lap -> set according lap progress
                self.progress[idx] = self.cur_lap

        # set correct position for every retired driver, update racetimes and laptimes arrays and sort drivers
        bool_retire_new = self.bool_driving[self.cur_lap] != self.bool_driving[self.cur_lap - 1]

        if np.any(bool_retire_new):
            # get indices of newly retired drivers
            idxs_retired = np.flatnonzero(bool_retire_new)

            # sort idxs_retired by ascending progress and descending positions (i.e. worst driver first) such that
            # positions can be set in the correct order afterwards when going from back to front
            sorted_idxs = np.lexsort((-self.positions[self.cur_lap, bool_retire_new], self.progress[bool_retire_new]))
            idxs_retired_sorted = idxs_retired[sorted_idxs]

            for idx in idxs_retired_sorted:
                # determine last position of all drivers that were driving until now (driving_bool used instead
                # self.bool_driving to also work if more than one driver retires in the same lap)
                driving_bool = np.invert(np.isnan(self.racetimes[self.cur_lap]))
                last_pos_driving = np.sum(driving_bool)

                # set last position for retiring driver and nan for the lap times and race times
                self.positions[self.cur_lap:, idx] = last_pos_driving
                self.laptimes[self.cur_lap:, idx] = np.nan
                self.racetimes[self.cur_lap:, idx] = np.nan

            # if a new driver retired sort him to the end and sort actually driving drivers to the front using
            # check_pos_changes_wo_timeloss. check_min_dist is not performed because the time deltas in between the cars
            # do not change due to a retirement.
            self.__check_pos_changes_wo_timeloss()

    def __handle_overtaking_track(self) -> None:
        """
        This method checks if overtaking maneuvers will happen in the current lap, i.e. if a rear driver is close enough
        to a front driver and fast enough to overtake him.

        t_duel is applied if rear driver overtakes or is fast enough to drive into the minimum distance min_t_dist.
        """

        # calculate racetimes based on the temporary laptimes
        racetimes_tmp = self.racetimes[self.cur_lap - 1] + self.laptimes[self.cur_lap]

        # loop through overtaking check as long as there are position changes to allow multiple overtaking within a lap
        # (e.g. if a driver is so fast that he overtakes two drivers in front)
        pos_change_occured_b = True
        t_duel_applied = np.full(self.no_drivers, False)  # array to keep track if t_duel was already applied to driver
        ctr = 0  # loop counter to detect and break posible infinite loop

        while pos_change_occured_b:
            ctr += 1  # increase loop counter
            pos_change_occured_b = False  # terminate if there is no position change in current iteration

            # detect and break possible infinite loop -> under certain circumstances (e.g. t_overtake_loser very small)
            # it could happen that drivers will overtake each other again and again leading to an infinite loop
            if ctr > 20:
                print("WARNING: Possible infinite loop detected in __handle_overtaking_track(). Will break loop and"
                      " mark result as invalid!")
                self.result_status = 10
                break

            # go through positions from front to back and check if driver is within overtake window
            for pos_iter in range(1, self.no_drivers):
                # using bool vectors instead of indices as in __assure_min_dists since positions can change during the
                # iterations
                pos_cur_b = self.positions[self.cur_lap] == pos_iter          # bool vector current position
                pos_back_b = self.positions[self.cur_lap] == pos_iter + 1     # bool vector backman position

                # continue only if driver behind is still driving
                if not self.bool_driving[self.cur_lap, pos_back_b]:
                    break

                # ------------------------------------------------------------------------------------------------------
                # DETERMINE IF OVERTAKING IS ALLOWED -------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                overtake_allowed = \
                    self.overtake_allowed[self.cur_lap, pos_cur_b] and self.overtake_allowed[self.cur_lap, pos_back_b]

                # ------------------------------------------------------------------------------------------------------
                # APPLY DUEL TIME LOSS ---------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                # apply t_duel if overtaking is allowed and drivers are in short distance -> once per driver and lap
                if overtake_allowed \
                        and racetimes_tmp[pos_cur_b] - racetimes_tmp[pos_back_b] > -self.race_pars["min_t_dist"]:
                    if not t_duel_applied[pos_cur_b]:
                        self.laptimes[self.cur_lap, pos_cur_b] += self.race_pars["t_duel"]
                        racetimes_tmp[pos_cur_b] += self.race_pars["t_duel"]
                        t_duel_applied[pos_cur_b] = True

                    if not t_duel_applied[pos_back_b]:
                        self.laptimes[self.cur_lap, pos_back_b] += self.race_pars["t_duel"]
                        racetimes_tmp[pos_back_b] += self.race_pars["t_duel"]
                        t_duel_applied[pos_back_b] = True

                # ------------------------------------------------------------------------------------------------------
                # CALCULATE T_GAP_OVERTAKE MODIFIERS -------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                if overtake_allowed:
                    # get t_gap_overtake modifier due to teamorder
                    idx_driver_cur = int(np.argmax(pos_cur_b))
                    idx_driver_back = int(np.argmax(pos_back_b))

                    if self.drivers_list[idx_driver_cur].team == self.drivers_list[idx_driver_back].team:
                        t_gap_overtake_mod_teamorder = self.drivers_list[idx_driver_back].t_teamorder
                    else:
                        t_gap_overtake_mod_teamorder = 0.0

                    # calculate velocity-delta dependent modificiation of t_gap_overtake
                    vel_delta = self.drivers_list[idx_driver_back].vel_max - self.drivers_list[
                        idx_driver_cur].vel_max
                    t_gap_overtake_mod_vel = self.track.t_gap_overtake_vel * vel_delta

                    # get final required overtaking time advantage
                    t_gap_overtake_tot = (self.track.t_gap_overtake + t_gap_overtake_mod_vel
                                          + t_gap_overtake_mod_teamorder)

                else:
                    t_gap_overtake_tot = None

                # ------------------------------------------------------------------------------------------------------
                # CHECK FOR OVERTAKE MANEUVER AND SET MINIMUM DISTANCE OTHERWISE ---------------------------------------
                # ------------------------------------------------------------------------------------------------------

                # overtake if driver behind is fast enough and overtaking is allowed due to flags
                if overtake_allowed and racetimes_tmp[pos_cur_b] - racetimes_tmp[pos_back_b] >= t_gap_overtake_tot:

                    # update positions
                    self.positions[self.cur_lap, pos_cur_b] += 1
                    self.positions[self.cur_lap, pos_back_b] -= 1

                    # increase lap time of overtaken driver because he usually could not drive on the raceline if he was
                    # overtaken -> eases multiple overtaking
                    self.laptimes[self.cur_lap, pos_cur_b] += self.race_pars["t_overtake_loser"]
                    racetimes_tmp[pos_cur_b] += self.race_pars["t_overtake_loser"]

                    """Depending on the race time gaps between the drivers it can temporarily happen that the minimum
                    temporal distance is not kept within this for-loop. Since we loop at least once more after an
                    overtaking maneuver happened, the minimum distance will finally be assured."""

                    # reset termination criteria
                    pos_change_occured_b = True

                # if driver behind is faster than frontman, but not fast enough or if he is in a too short distance to
                # him or if overtaking is not allowed due to flags: set laptime and racetime according to minimum
                # distance min_t_dist
                elif racetimes_tmp[pos_cur_b] - racetimes_tmp[pos_back_b] > -self.race_pars["min_t_dist"]:

                    # keep minimum distance between front and rear driver
                    self.laptimes[self.cur_lap, pos_back_b] += \
                        racetimes_tmp[pos_cur_b] + self.race_pars["min_t_dist"] - racetimes_tmp[pos_back_b]
                    racetimes_tmp[pos_back_b] = racetimes_tmp[pos_cur_b] + self.race_pars["min_t_dist"]

    def __handle_vse(self) -> None:
        """This method handles the VSE (virtual strategy engineer) which is used to take race strategy related decisions
        on the basis of the current race situation."""

        if self.vse is not None:
            # take tirechange decisions (are set None for retired drivers) (important: the decisions are taken based on
            # the data at the end of the previous lap (with some exceptions, e.g. FCY status))
            next_compound = self.vse.\
                decide_pitstop(driver_initials=[driver.initials for driver in self.drivers_list],
                               cur_compounds=[driver.car.tireset.compound for driver in self.drivers_list],
                               no_past_tirechanges=[len(driver.strategy_info) - 1 for driver in self.drivers_list],
                               tire_ages=[driver.car.tireset.age_degr for driver in self.drivers_list],
                               positions_prevlap=self.positions[self.cur_lap - 1],
                               pit_prevlap=[True if idx in self.pit_driver_idxs else False
                                            for idx in range(self.no_drivers)],
                               cur_lap=self.cur_lap,
                               tot_no_laps=self.race_pars["tot_no_laps"],
                               fcy_types=[self.fcy_data["phases"][idx_phase][2] if idx_phase is not None
                                          else None for idx_phase in self.fcy_handling["idxs_act_phase"]],
                               fcy_start_end_progs=self.fcy_handling["start_end_prog"],
                               bool_driving=self.bool_driving[self.cur_lap],
                               bool_driving_prevlap=self.bool_driving[self.cur_lap - 1],
                               racetimes_prevlap=self.racetimes[self.cur_lap - 1],
                               location=self.track.name,
                               used_2compounds=[True if len({x[1] for x in driver.strategy_info}) > 1 else False
                                                for driver in self.drivers_list],
                               cur_positions=self.positions[self.cur_lap],
                               cur_racetimes_tmp=self.racetimes[self.cur_lap - 1] + self.laptimes[self.cur_lap],
                               t_pit_tirechange_min=self.track.t_pit_tirechange_min,
                               t_pit_tirechange_adds=[driver.car.t_pit_tirechange_add for driver in self.drivers_list],
                               t_pitdrive_inlap=self.track.t_pitdrive_inlap,
                               t_pitdrive_outlap=self.track.t_pitdrive_outlap,
                               t_pitdrive_inlap_fcy=self.track.t_pitdrive_inlap_fcy,
                               t_pitdrive_outlap_fcy=self.track.t_pitdrive_outlap_fcy,
                               t_pitdrive_inlap_sc=self.track.t_pitdrive_inlap_sc,
                               t_pitdrive_outlap_sc=self.track.t_pitdrive_outlap_sc)

            # update strategy info for affected drivers
            for idx, compound in enumerate(next_compound):
                if compound is not None:
                    self.drivers_list[idx].strategy_info.append([self.cur_lap, compound, 0, 0.0])

    def __handle_pitstop_inlap(self) -> None:
        """
        Check for drivers doing a pitstop and save their indices (used again for pitstop_outlap afterwards). Consider
        that drivers could have retired before they reach their pitstop lap.
        """

        # reset pitstop information
        self.pit_driver_idxs = []

        # fill array with the indices belonging to the drivers that have an inlap this lap
        for idx in self.__get_driver_iter():
            # get boolean list if any of the pitstop inlaps matches the current lap
            pit_b = [True if self.cur_lap == pitstop[0] else False for pitstop in self.drivers_list[idx].strategy_info]

            if any(pit_b):
                # save index of pitting driver
                self.pit_driver_idxs.append(idx)

                # update lap_influences dict of affected driver
                self.drivers_list[idx].update_lap_influences(cur_lap=self.cur_lap, influence_type='pitinlap')

                # create variable to save pit time loss
                timeloss_pit = 0.0

                # if pits are located in front of the finish line add standstill time loss to inlap here
                if not self.track.pits_aft_finishline:
                    timeloss_pit += self.__perform_pitstop_standstill(idx_driver=idx, inlap=self.cur_lap)

                # determine pit driving inlap time loss -> modified time loss if it happens under an active FCY phase
                t_pitdrive_inlap_tmp = self.track.t_pitdrive_inlap

                if self.fcy_handling["idxs_act_phase"][idx] is not None:
                    check_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]]
                elif self.fcy_handling["idxs_next_phase"][idx] < len(self.fcy_data["phases"]):
                    check_fcy_phase = self.fcy_data["phases"][self.fcy_handling["idxs_next_phase"][idx]]
                else:
                    check_fcy_phase = None

                if check_fcy_phase is not None:
                    if check_fcy_phase[2] == 'VSC' \
                            and (check_fcy_phase[0]
                                 <= self.racetimes[self.cur_lap - 1, idx] + self.laptimes[self.cur_lap, idx]) \
                            and (self.racetimes[self.cur_lap - 1, idx] + self.laptimes[self.cur_lap, idx]
                                 + self.track.t_pitdrive_inlap_fcy < check_fcy_phase[1]):
                        # CASE 1: VSC phase (considered if the FCY phase fully covers the pit stop inlap part)
                        t_pitdrive_inlap_tmp = self.track.t_pitdrive_inlap_fcy

                    elif check_fcy_phase[2] == 'SC' \
                            and (check_fcy_phase[0]
                                 <= self.racetimes[self.cur_lap - 1, idx] + self.laptimes[self.cur_lap, idx]) \
                            and self.racetimes[self.cur_lap - 1, idx] < check_fcy_phase[1]:
                        # CASE 2: SC phase (considered if the FCY phase starts before entering the pit and if it ends
                        # after the start of the current lap, i.e. if it somehow reaches into the current lap (since SC
                        # stays until the end of the lap no matter if the FCY phase ends earlier))

                        if check_fcy_phase[0] < self.racetimes[self.cur_lap - 1, idx]:
                            # CASE 2a: SC phase started before this lap already -> drivers should have run up to it
                            t_pitdrive_inlap_tmp = self.track.t_pitdrive_inlap_sc
                        else:
                            # CASE 2b: SC phase started within this pit inlap -> no driver ran up to the SC already
                            # -> time loss is equal to a FCY phase instead
                            t_pitdrive_inlap_tmp = self.track.t_pitdrive_inlap_fcy

                # save t_pitdrive part
                timeloss_pit += t_pitdrive_inlap_tmp

                # add timeloss to current laptime
                self.laptimes[self.cur_lap, idx] += timeloss_pit

        # check for position changes (without overtaking timeloss) only if there are pit drivers
        if self.pit_driver_idxs:
            self.__check_pos_changes_wo_timeloss()

    def __fcy_phase_checks_aft_final_laptimes(self) -> None:
        """
        This method performs actions related to FCY phases that require to know the final lap (and race) times. This
        includes setting the flag state of the race (used during plotting of the race results mostly) as well as
        resetting the active FCY phase if it ended within the current lap. Additionally, we check if a FCY phase should
        have started within the current lap knowing the final lap times (e.g. because a pitstop inlap was driven after
        the FCY handling in the other method) or if a FCY phase should have started within the current lap after the
        previous phase ended. The latter sometimes happens in reality if a VSC phase is directly followed by an SC
        phase. In one of these cases, the lap times are not touched anymore. However, to handle an SC as correctly as
        possible, the SC ghost must be started within this lap. This behavior does not lead to wrong results since no
        driver catches up to the SC during its first lap.
        """

        for idx in self.__get_driver_iter():
            # update race flag state if current driver is the leader
            if self.positions[self.cur_lap, idx] == 1 and self.fcy_handling["idxs_act_phase"][idx] is not None:
                self.flagstates[self.cur_lap] = self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]][2]

            # reset FCY phases that ended within the current lap
            self.check_fcyphase_reset(idx_driver=idx)

            # check if another phase should have started within the current lap but did not since the simulation can
            # currently only handle one phase per lap -> race flag state is not updated in this case since it should
            # show which phase was really considered within the lap
            self.check_fcyphase_activation(idx_driver=idx)

            # if a new FCY phase was started set start and end progress information (used for VSE)
            if self.fcy_handling["idxs_act_phase"][idx] is not None \
                    and self.fcy_handling["start_end_prog"][idx][0] is None:
                # get lap fractions
                lap_frac_normal, lap_frac_normal_bef = self.calc_lapfracs_fcyphase(idx_driver=idx)

                # set progress information
                self.fcy_handling["start_end_prog"][idx][0] = self.cur_lap - 1.0 + lap_frac_normal_bef
                self.fcy_handling["start_end_prog"][idx][1] = self.cur_lap - (lap_frac_normal - lap_frac_normal_bef)

            # if a new SC phase was started now we have to set the SC ghost data accordingly
            if self.fcy_handling["idxs_act_phase"][idx] is not None \
                    and self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]][2] == 'SC' \
                    and self.fcy_handling["sc_ghost_racetimes"][idx] is None:

                # set SC ghost data
                self.fcy_handling["sc_ghost_racetimes"][idx] = \
                    self.fcy_data["phases"][self.fcy_handling["idxs_act_phase"][idx]][0]
                self.fcy_handling["sc_ghost_laps"][idx] = 0  # 0 equals start lap

                # DRS usage is not allowed directly after an SC phase (+ 1 because current lap is still SC)
                if self.race_pars["use_drs"]:
                    self.race_pars["drs_act_lap"][idx] = self.cur_lap + 1 + self.race_pars["drs_sc_delay"]

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (HELPERS) ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __check_pos_changes_wo_timeloss(self,
                                        t_lap_tmp: np.ndarray = None) -> None:
        """
        Primarily used to check position changes due to pit stops.
        """

        # calculate racetimes based on the temporary laptimes
        if t_lap_tmp is None:
            racetimes_tmp = self.racetimes[self.cur_lap - 1] + self.laptimes[self.cur_lap]
        else:
            racetimes_tmp = self.racetimes[self.cur_lap - 1] + t_lap_tmp

        """Sort racetimes_tmp and set positions accordingly (use previous positions as second sorting criteria). This
        works with np.nan entries in the racetimes_tmp array: they are sorted to the end. Positions for retired
        drivers must be set separately afterwards."""

        # sort positions
        sorted_idxs = np.lexsort((self.positions[self.cur_lap], racetimes_tmp))
        self.positions[self.cur_lap, sorted_idxs] = np.arange(1, self.no_drivers + 1)

    def __assure_min_dists(self,
                           t_lap_tmp: np.ndarray = None) -> None:
        """
        Check distances between cars to keep a minimum distance.
        """

        # calculate racetimes based on the temporary laptimes
        if t_lap_tmp is None:
            racetimes_tmp = self.racetimes[self.cur_lap - 1] + self.laptimes[self.cur_lap]
        else:
            racetimes_tmp = self.racetimes[self.cur_lap - 1] + t_lap_tmp

        # iterate through positions from front to back and check if distance is lower than minimum distance
        # -> using index pairs with driver on current position and on position behind
        idxs_sorted_by_pos = np.argsort(self.positions[self.cur_lap])

        for idx_pair in zip(idxs_sorted_by_pos, idxs_sorted_by_pos[1:]):
            # continue only if driver behind is still driving
            if not self.bool_driving[self.cur_lap, idx_pair[1]]:
                break

            # if driver behind is faster than frontman or in a too short distance to him: set laptime and racetime
            # according to minimum distance min_t_dist
            if racetimes_tmp[idx_pair[0]] - racetimes_tmp[idx_pair[1]] > -self.race_pars["min_t_dist"]:
                # add difference between originally calculated racetime and real racetime to the calculated laptime
                self.laptimes[self.cur_lap, idx_pair[1]] += (racetimes_tmp[idx_pair[0]] + self.race_pars["min_t_dist"]
                                                             - racetimes_tmp[idx_pair[1]])
                racetimes_tmp[idx_pair[1]] = racetimes_tmp[idx_pair[0]] + self.race_pars["min_t_dist"]

    def get_last_compl_lap(self,
                           idx: int) -> int:
        """This method returns the lap number of the last completed (i.e. not NaN) lap of a given driver index. The
        method returns the final lap if driver finished all laps."""

        if not np.isnan(self.racetimes[-1, idx]):
            # CASE 1: driver finished whole race
            last_compl_lap = self.race_pars['tot_no_laps']

        elif np.isnan(self.racetimes[1, idx]):
            # CASE 2: driver retired in the first lap
            last_compl_lap = 0

        else:
            # CASE 3: driver was either lapped or retired somewhere in the race
            last_compl_lap = int(np.argmax(np.isnan(self.racetimes[:, idx]))) - 1

        return last_compl_lap

    def __get_driver_iter(self) -> Iterator:
        """This method returns an iterator for the driver indices of those drivers that did not retire."""

        # get driver indices for drivers that did not retire
        driving_drivers = list(np.arange(self.no_drivers)[self.bool_driving[self.cur_lap]])

        return iter(driving_drivers)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (PROCESS RESULTS) ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __reset_invalid_laps_aft_race(self) -> None:
        """
        The drivers are allowed to finish their current lap as soon as the winner crosses the finish line. This leads
        to the case that laped drivers will not run the last lap(s) of a race. Therefore, these laps should be set to
        nan in the simulation. However, this has some more implications on the positions which must be corrected.

        This method is executed after the race was fully simulated.
        """

        # --------------------------------------------------------------------------------------------------------------
        # REMOVE INVALID LAPS ------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # get race time of winner (catch case that nobody might have finished the race completely)
        idx_winner = int(np.argmax(self.positions[-1] == 1))
        last_compl_lap_winner = self.get_last_compl_lap(idx=idx_winner)
        t_race_winner = self.racetimes[last_compl_lap_winner, idx_winner]

        if last_compl_lap_winner < self.race_pars['tot_no_laps']:
            print("WARNING: Even the race winner did not finish all laps (retired after lap %i), race will be marked as"
                  " invalid!" % last_compl_lap_winner)
            self.result_status = 1  # this seems to be no valid result

        # loop through the drivers
        for idx in range(self.no_drivers):

            # jump over current driver if he is the race winner
            if self.positions[-1, idx] == 1:
                continue

            # get index of first lap that has a race time greater than t_race_winner -> returns 0 if no lap is found
            idx_first_nan = self.get_last_compl_lap(idx=idx) + 1
            racetime_comp_b = self.racetimes[:idx_first_nan, idx] > t_race_winner
            idx_greater_t_race_winner = int(np.argmax(racetime_comp_b))

            # if idx_greater_t_race_winner is not zero and is reached before last lap we have to modify the result
            if 0 < idx_greater_t_race_winner < self.race_pars["tot_no_laps"]:
                self.positions[idx_greater_t_race_winner + 1:, idx] = self.positions[idx_greater_t_race_winner, idx]
                self.laptimes[idx_greater_t_race_winner + 1:, idx] = np.nan
                self.racetimes[idx_greater_t_race_winner + 1:, idx] = np.nan
                self.progress[idx] = idx_greater_t_race_winner

        # --------------------------------------------------------------------------------------------------------------
        # CORRECT POSITIONS ARRAY --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # loop through all laps of the race and set new positions
        for lap_tmp in range(1, self.race_pars['tot_no_laps'] + 1):
            # sort positions by racetimes, progress and original positions -> idea is to sort all driving drivers in
            # front, then to add the drivers who retired during the current lap (progress X.5), then to add drivers
            # who did not start the current lap due to the previous step (progress X.0) and finally to add drivers who
            # retired earlier or did not even start earlier laps. Within every of those steps the original positions
            # should be kept.
            sorted_idxs = np.lexsort((self.positions[lap_tmp], -self.progress, self.racetimes[lap_tmp]))
            self.positions[lap_tmp, sorted_idxs] = np.arange(1, self.no_drivers + 1)

    def __check_plausibility(self):
        """This method performs some simple checks on the simulation result."""

        # check if laptimes sum up to total racetime of every driver ---------------------------------------------------
        for idx in range(self.no_drivers):

            # find last valid racetime of driver
            last_compl_lap = self.get_last_compl_lap(idx=idx)

            if not np.isclose(np.nansum(self.laptimes[:, idx]), self.racetimes[last_compl_lap, idx]):
                print("WARNING: Summed laptime of driver %s does not equal his total racetime, race will be marked as"
                      " invalid!" % self.drivers_list[idx].initials)
                self.result_status = 11

        # check if every position exists only once for every lap -------------------------------------------------------
        ref_positions = np.arange(1, self.no_drivers + 1)

        for cur_lap in range(self.race_pars["tot_no_laps"] + 1):
            positions_sorted = np.sort(self.positions[cur_lap])

            if not np.all(ref_positions == positions_sorted):
                print("WARNING: Positions are not plausible in lap %i, race will be marked as invalid!" % cur_lap)
                self.result_status = 12

        # check if minimum distance between drivers is always kept (consider that pitstop inlaps can lead to "invalid"
        # distances when another driver has an almost equal race time on the track at the start finish line)
        # (strategy_info[1:] is used to avoid considering the start tire information) ----------------------------------
        pit_inlaps = [[pitstop[0] for pitstop in driver.strategy_info[1:]] for driver in self.drivers_list]
        racetimes_sorted = np.sort(self.racetimes[1:], axis=1)

        with np.errstate(invalid='ignore'):  # errors must be ignored here due to warnings from nan comparison
            # get bool array containing invalid distances between drivers
            invalid_dists = np.diff(racetimes_sorted, axis=1) + 1e-7 < self.race_pars["min_t_dist"]

            # loop through all pitstop inlaps and set according entries in the bool array False
            for idx, inlaps in enumerate(pit_inlaps):
                for inlap in inlaps:
                    # get driver position in the inlap
                    pos = self.positions[inlap, idx]

                    # distance to driver in front and driver in the back can both be wrong
                    # inlap - 1 because invalid_dists starts in lap 1
                    # pos - 2 because positions start with 1 and we want the difference to position in front of driver
                    if pos > 1:
                        invalid_dists[inlap - 1, pos - 2] = False
                    if pos < self.no_drivers:
                        invalid_dists[inlap - 1, pos - 1] = False

            # check if there are invalid distances remaining
            if np.any(invalid_dists):
                print("WARNING: Minimum distances between drivers are not plausible, race will be marked as invalid!")
                self.result_status = 13

        # check if possibly set FCY phases appear in the flag states ---------------------------------------------------
        if self.fcy_data["phases"] and not any(True if x in ["VSC", "SC"] else False for x in self.flagstates):
            print("WARNING: FCY phases were set but do not appear in the flag states, race will be marked as invalid!")
            self.result_status = 14

        # check if every driver used at least two different compounds --------------------------------------------------
        for idx_driver, driver in enumerate(self.drivers_list):
            if self.bool_driving[-1, idx_driver]\
                    and not len({x[1] for x in driver.strategy_info}) > 1\
                    and idx_driver not in self.vse.idxs_driver_reinf_training:
                print("WARNING: %s did not use two different compounds during the race, race will be marked as"
                      " invalid!" % driver.initials)
                self.result_status = 15


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
