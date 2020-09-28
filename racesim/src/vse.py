import numpy as np
from racesim.src.driver import Driver
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set logging level such that TF shows only errors
from racesim.src.vse_supervised import VSE_SUPERVISED
from racesim.src.vse_basestrategy import VSE_BASESTRATEGY
from racesim.src.vse_realstrategy import VSE_REALSTRATEGY


class VSE(object):
    """
    author:
    Alexander Heilmeier

    date:
    04.05.2020

    .. description::
    This class handles the various variants of the VSE (Virtual Strategy Engineer). The VSE takes decisions related
    to race strategy, i.e. it determines when a pit stop happens and which compound is used for the next stint.
    Currently, there are four types of VSE: the supervised VSE, a base strategy VSE, and a real strategy VSE. The
    supervised VSE is based on a NN that was trained on real data. The base strategy VSE and the real strategy VSE rely
    on the pre-definied base strategies (optimal strategy on a free race track, i.e. without opponents) and real
    strategies in the race parameter file.

    tc = tire change decision
    cc = compound choice
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__vse_supervised",
                 "__vse_base",
                 "__vse_real",
                 "__idxs_driver_supervised",
                 "__idxs_driver_base",
                 "__idxs_driver_real",
                 "__no_drivers",
                 "__avail_dry_compounds",
                 "__param_dry_compounds",
                 "__vse_pars",
                 "__cache_tireageprogress_corr_prevlap",
                 "__cache_position_preprevlap",
                 "__cache_ahead_preprevlap")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 vse_paths: dict,
                 vse_pars: dict) -> None:

        # initialize known variables (avail_dry_compounds should be solely used for the compound choice NN!)
        self.vse_pars = vse_pars
        self.avail_dry_compounds = [x for x in self.vse_pars["available_compounds"] if x not in ["I", "W"]]
        self.param_dry_compounds = self.vse_pars["param_dry_compounds"]

        # initialize unknown variables (they are set during the first call of the decide_pitstop method)
        self.idxs_driver_supervised = None
        self.idxs_driver_base = None
        self.idxs_driver_real = None
        self.no_drivers = None
        self.cache_tireageprogress_corr_prevlap = None
        self.cache_position_preprevlap = None
        self.cache_ahead_preprevlap = None

        # create supervised VSE (if indicated) -------------------------------------------------------------------------
        if any(True if self.vse_pars["vse_type"][key] == 'supervised' else False for key in self.vse_pars["vse_type"]):
            self.vse_supervised = VSE_SUPERVISED(preprocessor_cc_path=vse_paths["supervised_preprocessor_cc"],
                                                 preprocessor_tc_path=vse_paths["supervised_preprocessor_tc"],
                                                 nnmodel_cc_path=vse_paths["supervised_nnmodel_cc"],
                                                 nnmodel_tc_path=vse_paths["supervised_nnmodel_tc"])
        else:
            self.vse_supervised = None

        # create base strategy VSE (if indicated) ----------------------------------------------------------------------
        if any(True if self.vse_pars["vse_type"][key] == 'basestrategy' else False
               for key in self.vse_pars["vse_type"]):
            self.vse_base = VSE_BASESTRATEGY(base_strategies=self.vse_pars["base_strategy"])
        else:
            self.vse_base = None

        # create real strategy VSE (if indicated) ----------------------------------------------------------------------
        if any(True if self.vse_pars["vse_type"][key] == 'realstrategy' else False
               for key in self.vse_pars["vse_type"]):
            self.vse_real = VSE_REALSTRATEGY(real_strategies=self.vse_pars["real_strategy"])
        else:
            self.vse_real = None

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_vse_supervised(self) -> VSE_SUPERVISED: return self.__vse_supervised
    def __set_vse_supervised(self, x: VSE_SUPERVISED) -> None: self.__vse_supervised = x
    vse_supervised = property(__get_vse_supervised, __set_vse_supervised)

    def __get_vse_base(self) -> VSE_BASESTRATEGY: return self.__vse_base
    def __set_vse_base(self, x: VSE_BASESTRATEGY) -> None: self.__vse_base = x
    vse_base = property(__get_vse_base, __set_vse_base)

    def __get_vse_real(self) -> VSE_REALSTRATEGY: return self.__vse_real
    def __set_vse_real(self, x: VSE_REALSTRATEGY) -> None: self.__vse_real = x
    vse_real = property(__get_vse_real, __set_vse_real)

    def __get_idxs_driver_supervised(self) -> list: return self.__idxs_driver_supervised
    def __set_idxs_driver_supervised(self, x: list) -> None: self.__idxs_driver_supervised = x
    idxs_driver_supervised = property(__get_idxs_driver_supervised, __set_idxs_driver_supervised)

    def __get_idxs_driver_base(self) -> list: return self.__idxs_driver_base
    def __set_idxs_driver_base(self, x: list) -> None: self.__idxs_driver_base = x
    idxs_driver_base = property(__get_idxs_driver_base, __set_idxs_driver_base)

    def __get_idxs_driver_real(self) -> list: return self.__idxs_driver_real
    def __set_idxs_driver_real(self, x: list) -> None: self.__idxs_driver_real = x
    idxs_driver_real = property(__get_idxs_driver_real, __set_idxs_driver_real)

    def __get_no_drivers(self) -> int: return self.__no_drivers
    def __set_no_drivers(self, x: int) -> None: self.__no_drivers = x
    no_drivers = property(__get_no_drivers, __set_no_drivers)

    def __get_avail_dry_compounds(self) -> list: return self.__avail_dry_compounds
    def __set_avail_dry_compounds(self, x: list) -> None: self.__avail_dry_compounds = x
    avail_dry_compounds = property(__get_avail_dry_compounds, __set_avail_dry_compounds)

    def __get_param_dry_compounds(self) -> list: return self.__param_dry_compounds
    def __set_param_dry_compounds(self, x: list) -> None: self.__param_dry_compounds = x
    param_dry_compounds = property(__get_param_dry_compounds, __set_param_dry_compounds)

    def __get_vse_pars(self) -> dict: return self.__vse_pars
    def __set_vse_pars(self, x: dict) -> None: self.__vse_pars = x
    vse_pars = property(__get_vse_pars, __set_vse_pars)

    def __get_cache_tireageprogress_corr_prevlap(self) -> list: return self.__cache_tireageprogress_corr_prevlap
    def __set_cache_tireageprogress_corr_prevlap(self, x: list) -> None: self.__cache_tireageprogress_corr_prevlap = x
    cache_tireageprogress_corr_prevlap = property(__get_cache_tireageprogress_corr_prevlap,
                                                  __set_cache_tireageprogress_corr_prevlap)

    def __get_cache_position_preprevlap(self) -> np.ndarray: return self.__cache_position_preprevlap
    def __set_cache_position_preprevlap(self, x: np.ndarray) -> None: self.__cache_position_preprevlap = x
    cache_position_preprevlap = property(__get_cache_position_preprevlap, __set_cache_position_preprevlap)

    def __get_cache_ahead_preprevlap(self) -> np.ndarray: return self.__cache_ahead_preprevlap
    def __set_cache_ahead_preprevlap(self, x: np.ndarray) -> None: self.__cache_ahead_preprevlap = x
    cache_ahead_preprevlap = property(__get_cache_ahead_preprevlap, __set_cache_ahead_preprevlap)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        # reset VSE such that it can be used to simulate (the same race) again
        self.idxs_driver_supervised = None
        self.idxs_driver_base = None
        self.idxs_driver_real = None
        self.no_drivers = None
        self.cache_tireageprogress_corr_prevlap = None
        self.cache_position_preprevlap = None
        self.cache_ahead_preprevlap = None

        if self.vse_supervised is not None:
            self.vse_supervised.reset()

        if self.vse_base is not None:
            self.vse_base.reset()

        if self.vse_real is not None:
            self.vse_real.reset()

    def decide_pitstop(self,
                       driver_initials: list,
                       cur_compounds: list,
                       no_past_tirechanges: list,
                       tire_ages: list,
                       positions_prevlap: np.ndarray,
                       pit_prevlap: list,
                       cur_lap: int,
                       tot_no_laps: int,
                       fcy_types: list,
                       fcy_start_end_progs: list,
                       bool_driving: np.ndarray,
                       bool_driving_prevlap: np.ndarray,
                       racetimes_prevlap: np.ndarray,
                       location: str,
                       used_2compounds: list) -> list:
        """
        .. inputs::
        :param driver_initials:         List with driver initials (must be in the same order as the remaining inputs)
        :type driver_initials:          list
        :param cur_compounds:           List with current compound for every driver.
        :type cur_compounds:            list
        :param no_past_tirechanges:     List with number of already executed tire changes for every driver.
        :type no_past_tirechanges:      list
        :param tire_ages:               List with tire age (tire_age_degr) for every driver.
        :type tire_ages:                list
        :param positions_prevlap:       Array containing the positions of all drivers at the end of the previous lap.
        :type positions_prevlap:        np.ndarray
        :param pit_prevlap:             List containing True for drivers who have their out-lap this lap and False
                                        otherwise.
        :type pit_prevlap:              list
        :param cur_lap:                 Current lap.
        :type cur_lap:                  int
        :param tot_no_laps:             Total number of laps in the race.
        :type tot_no_laps:              int
        :param fcy_types:               List containing the type of active FCY phases or None elsewise for every driver.
        :type fcy_types:                list
        :param fcy_start_end_progs:     List containing start and end progress of active FCY phases or [None, None]
                                        elsewise for every driver.
        :type fcy_start_end_progs:      list
        :param bool_driving:            Array containing True for all drivers that have not retired from the race.
        :type bool_driving:             np.ndarray
        :param bool_driving_prevlap:    Array containing True for all drivers that have not retired from the race until
                                        the end of the previous lap.
        :type bool_driving_prevlap:     np.ndarray
        :param racetimes_prevlap:       Array containing the race times of all drivers at the end of the previous lap.
        :type racetimes_prevlap:        np.ndarray
        :param location:                Name of current location
        :type location:                 str
        :param used_2compounds:         List containing booleans that indicate if the according driver used two
                                        different compounds in the race
        :type used_2compounds:          list
        """

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZATION (IF CALLED FOR THE FIRST TIME) ----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # save idxs of drivers using supervised VSE and base strategy VSE and real strategy VSE
        if self.idxs_driver_supervised is None:
            self.idxs_driver_supervised = [idx for idx, initials in enumerate(driver_initials)
                                           if self.vse_pars["vse_type"][initials] == 'supervised']

        if self.idxs_driver_base is None:
            self.idxs_driver_base = [idx for idx, initials in enumerate(driver_initials)
                                     if self.vse_pars["vse_type"][initials] == 'basestrategy']

        if self.idxs_driver_real is None:
            self.idxs_driver_real = [idx for idx, initials in enumerate(driver_initials)
                                     if self.vse_pars["vse_type"][initials] == 'realstrategy']

        # save number of drivers
        if self.no_drivers is None:
            self.no_drivers = len(driver_initials)

        # initialize cache for tire age progress data -> contains tire age progress at the end of the previous lap
        if self.cache_tireageprogress_corr_prevlap is None:
            # age at the race start is either in the range (0.0, 1.0) at the end of the first lap, if the driver started
            # on a fresh set, or in the range (2.0, 3.0), if driver started on a used set -> set 2.0 in the second case
            # (-1.0 does not work if there is an FCY phase in the first lap)
            self.cache_tireageprogress_corr_prevlap = [2.0 / tot_no_laps if age > 2.0 else 0.0 for age in tire_ages]

        # initialize cache for position data -> contains positions at the end of the lap before the previous lap
        if self.cache_position_preprevlap is None:
            self.cache_position_preprevlap = np.full(self.no_drivers, np.nan)

        # initialize cache for ahead data -> contains ahead values at the end of the lap before the previous lap
        if self.cache_ahead_preprevlap is None:
            self.cache_ahead_preprevlap = np.full(self.no_drivers, np.nan)

        # --------------------------------------------------------------------------------------------------------------
        # GENERAL PREPROCESSING ----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        """It makes sense to handle the preprocessing here instead within the VSEs since some of the preprocessing
        requires knowledge of the states of all drivers, independently from which VSE type they use, e.g.
        tirechange_pursuer."""

        # raceprogress_prevlap -----------------------------------------------------------------------------------------
        raceprogress_prevlap = (cur_lap - 1) / tot_no_laps

        # rel_compound_num_curlap --------------------------------------------------------------------------------------
        if len(self.avail_dry_compounds) == 2:
            # in 2014 and 2015 there were two relative compounds per race: medium and soft -> add 1 to jump hard
            rel_compound_num_curlap = [self.avail_dry_compounds.index(compound) + 1 for compound in cur_compounds]

        else:
            rel_compound_num_curlap = [self.avail_dry_compounds.index(compound) for compound in cur_compounds]

        # fcy_stat_curlap ----------------------------------------------------------------------------------------------
        fcy_stat_curlap = [0] * self.no_drivers

        for idx_driver in range(self.no_drivers):
            # avoid feature generation for retired drivers
            if not bool_driving[idx_driver]:
                continue

            # a status other than 0 is only set if a FCY phase is active for the current driver
            if fcy_types[idx_driver] is not None:
                # FCY phase is only relevant for the decision if it was active at least 5% before the end of the lap and
                # if it is still active at the end of the lap
                if cur_lap - fcy_start_end_progs[idx_driver][0] >= 0.05 \
                        and math.isclose(fcy_start_end_progs[idx_driver][1], cur_lap):
                    # CASE 1: VSC phase
                    if fcy_types[idx_driver] == 'VSC':
                        # check if phase started within this lap or not and set according status
                        if cur_lap - fcy_start_end_progs[idx_driver][0] <= 1.0:
                            fcy_stat_curlap[idx_driver] = 1
                        else:
                            fcy_stat_curlap[idx_driver] = 2
                    # CASE 2: SC phase
                    else:
                        # check if phase started within this lap or not and set according status
                        if cur_lap - fcy_start_end_progs[idx_driver][0] <= 1.0:
                            fcy_stat_curlap[idx_driver] = 3
                        else:
                            fcy_stat_curlap[idx_driver] = 4

        # remainingtirechanges_curlap ----------------------------------------------------------------------------------
        remainingtirechanges_curlap = [len(self.vse_pars["base_strategy"][initials]) - 1
                                       - no_past_tirechanges[idx_driver]
                                       for idx_driver, initials in enumerate(driver_initials)]

        if any(True if x > 3 else False for x in remainingtirechanges_curlap):
            raise ValueError("The NNs are trained for a maximum of 3 pit stops per race, reduce desired number of"
                             " pit stops!")

        # tirechange_pursuer_prevlap -----------------------------------------------------------------------------------
        tirechange_pursuer_prevlap = [0] * self.no_drivers

        if cur_lap > 1:
            for idx_driver in range(self.no_drivers):
                """The pursuer is determined on the basis of the lap before the previous lap to avoid using the wrong
                pursuer if he drove into the pit."""

                # avoid feature generation for retired drivers
                if not bool_driving[idx_driver]:
                    continue

                # pursuer only available for drivers who are not on the last position
                if self.cache_position_preprevlap[idx_driver] < self.no_drivers:
                    # get idx of pursuer
                    pos_back_b = self.cache_position_preprevlap == self.cache_position_preprevlap[idx_driver] + 1
                    idx_driver_back = int(np.argmax(pos_back_b))

                    # if pursuer was in pit in the previous lap (not the lap before the previous lap) change according
                    # entry to 1
                    if pit_prevlap[idx_driver_back]:
                        tirechange_pursuer_prevlap[idx_driver] = 1

        # close_ahead_preprevlap ---------------------------------------------------------------------------------------
        if cur_lap <= 2:
            # there are no valid ahead values for the lap before the previous lap in first and second lap of the race
            close_ahead_preprevlap = [False] * self.no_drivers
        else:
            close_ahead_preprevlap = list(self.cache_ahead_preprevlap < 1.5)

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZE OUTPUT LIST ---------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create list containing None for drivers who perform no pit stop this lap, otherwise the compound name that is
        # chosen, e.g. [None, "C4", None, None] for four drivers
        next_compounds = [None] * self.no_drivers

        # --------------------------------------------------------------------------------------------------------------
        # MAKE DECISIONS WITH SUPERVISED VSE ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.vse_supervised is not None and self.idxs_driver_supervised:
            # preprocessing
            self.vse_supervised.preprocess_features(
                tireageprogress_corr_zeroinchange=[self.cache_tireageprogress_corr_prevlap[idx]
                                                   for idx in self.idxs_driver_supervised],
                raceprogress=raceprogress_prevlap,
                position=[positions_prevlap[idx] for idx in self.idxs_driver_supervised],
                rel_compound_num_nl=[rel_compound_num_curlap[idx] for idx in self.idxs_driver_supervised],
                fcy_stat_nl=[fcy_stat_curlap[idx] for idx in self.idxs_driver_supervised],
                remainingtirechanges_nl=[remainingtirechanges_curlap[idx] for idx in self.idxs_driver_supervised],
                tirechange_pursuer=[tirechange_pursuer_prevlap[idx] for idx in self.idxs_driver_supervised],
                location_cat=self.vse_pars["location_cat"],
                close_ahead_prevlap=[close_ahead_preprevlap[idx] for idx in self.idxs_driver_supervised],
                location=location,
                used_2compounds_nl=[used_2compounds[idx] for idx in self.idxs_driver_supervised],
                no_avail_dry_compounds=len(self.avail_dry_compounds))

            # decision making (returns a list with an entry for every driver in idxs_driver_supervised)
            next_compounds_tmp = self.vse_supervised.make_decision(
                bool_driving=bool_driving[self.idxs_driver_supervised],
                avail_dry_compounds=self.avail_dry_compounds,
                param_dry_compounds=self.param_dry_compounds,
                remainingtirechanges_curlap=[remainingtirechanges_curlap[idx] for idx in self.idxs_driver_supervised],
                used_2compounds=[used_2compounds[idx] for idx in self.idxs_driver_supervised],
                cur_compounds=[cur_compounds[idx] for idx in self.idxs_driver_supervised],
                raceprogress_prevlap=raceprogress_prevlap)

            for idx_rel, idx_abs in enumerate(self.idxs_driver_supervised):
                if next_compounds_tmp[idx_rel] is not None:
                    next_compounds[idx_abs] = next_compounds_tmp[idx_rel]

        # --------------------------------------------------------------------------------------------------------------
        # MAKE DECISIONS WITH BASE STRATEGY VSE ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.vse_base is not None and self.idxs_driver_base:
            # decision making (returns a list with an entry for every driver in idxs_driver_base)
            next_compounds_tmp = self.vse_base.make_decision(
                cur_lap=cur_lap,
                driver_initials=[driver_initials[idx] for idx in self.idxs_driver_base],
                bool_driving=bool_driving[self.idxs_driver_base],
                param_dry_compounds=self.param_dry_compounds)

            for idx_rel, idx_abs in enumerate(self.idxs_driver_base):
                if next_compounds_tmp[idx_rel] is not None:
                    next_compounds[idx_abs] = next_compounds_tmp[idx_rel]

        # --------------------------------------------------------------------------------------------------------------
        # MAKE DECISIONS WITH REAL STRATEGY VSE ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.vse_real is not None and self.idxs_driver_real:
            # decision making (returns a list with an entry for every driver in idxs_driver_real)
            next_compounds_tmp = self.vse_real.make_decision(
                cur_lap=cur_lap,
                driver_initials=[driver_initials[idx] for idx in self.idxs_driver_real],
                bool_driving=bool_driving[self.idxs_driver_real],
                param_dry_compounds=self.param_dry_compounds)

            for idx_rel, idx_abs in enumerate(self.idxs_driver_real):
                if next_compounds_tmp[idx_rel] is not None:
                    next_compounds[idx_abs] = next_compounds_tmp[idx_rel]

        # --------------------------------------------------------------------------------------------------------------
        # UPDATE CACHES FOR NEXT LAP -----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # prepare ahead values of previous lap (zero for lap 0 evaluated in lap 1) -------------------------------------
        idxs_sorted = list(np.argsort(positions_prevlap))
        aheads_prevlap = np.zeros(self.no_drivers)

        for position_minus1, idx_driver in enumerate(idxs_sorted):
            # continue if driver on current position had retired
            if not bool_driving_prevlap[idx_driver]:
                aheads_prevlap[idx_driver] = np.inf
                continue

            # determine boolean masks
            pos_cur = position_minus1 + 1
            pos_cur_b = positions_prevlap == pos_cur

            if pos_cur < self.no_drivers:
                pos_back_b = positions_prevlap == pos_cur + 1
            else:
                pos_back_b = None

            if pos_back_b is not None and not bool_driving_prevlap[pos_back_b]:
                # pretend there is no driver behind if driver behind retired
                pos_back_b = None

            if pos_back_b is not None:
                aheads_prevlap[idx_driver] = racetimes_prevlap[pos_back_b] - racetimes_prevlap[pos_cur_b]
            else:
                aheads_prevlap[idx_driver] = np.inf

        # cache tireageprogress_corr data of current lap ---------------------------------------------------------------
        self.cache_tireageprogress_corr_prevlap = [age / tot_no_laps for age in tire_ages]

        # set tire age in cache zero for drivers that perform a pit stop this lap
        for idx_driver, compound in enumerate(next_compounds):
            if compound is not None:
                self.cache_tireageprogress_corr_prevlap[idx_driver] = 0.0

        # cache position data of previous lap --------------------------------------------------------------------------
        self.cache_position_preprevlap = np.copy(positions_prevlap)

        # cache ahead data of previous lap (copy not required since aheads_prevlap was created within this method) -----
        self.cache_ahead_preprevlap = aheads_prevlap

        return next_compounds

    def determine_basic_strategy(self,
                                 driver: Driver,
                                 tot_no_laps: int,
                                 fcy_phases: list,
                                 location: str,
                                 mult_tiredeg_fcy: float = 0.5,
                                 mult_tiredeg_sc: float = 0.25) -> list:

        """
        This method is intended to determine a basic strategy with the VSE for the pre-simulation such that it resembles
        the behavior of the VSE in the race. Equally to the pre-simulation, this method only works with FCY phases in
        the progress domain.
        """

        # --------------------------------------------------------------------------------------------------------------
        # SIMULATE RACE ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create list for strategy output starting with the correct information for the race start
        basic_strategy_info = [driver.strategy_info[0]]

        # consider correct start age
        tire_age_tmp = float(basic_strategy_info[0][2])

        for cur_lap in range(1, tot_no_laps + 1):

            # ----------------------------------------------------------------------------------------------------------
            # FEATURE PREPARATION --------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            # tirechange_tmp -------------------------------------------------------------------------------------------
            if cur_lap > 1 and basic_strategy_info[-1][0] == cur_lap - 1:
                tirechange_tmp = True
            else:
                tirechange_tmp = False

            # fcy_types_tmp and fcy_start_end_progs_tmp ----------------------------------------------------------------
            fcy_types_tmp = None
            fcy_start_end_progs_tmp = [None, None]
            lap_frac_normal = 1.0

            for idx_fcy_phase, cur_fcy_phase in enumerate(fcy_phases):
                # check if a phase affects the current lap (phases are in the progress domain if this method is called)
                if cur_lap - 1.0 < cur_fcy_phase[1] and cur_fcy_phase[0] < cur_lap:
                    # end progress can only be foreseen until the end of the current lap
                    if cur_fcy_phase[2] == 'SC' or cur_fcy_phase[1] >= cur_lap:
                        end_prog_tmp = float(cur_lap)
                    else:
                        end_prog_tmp = cur_fcy_phase[1]

                    fcy_types_tmp = cur_fcy_phase[2]
                    fcy_start_end_progs_tmp = [cur_fcy_phase[0], end_prog_tmp]

                    # save lap fraction information for tire age calculation
                    lap_frac_normal = cur_lap - end_prog_tmp + max(cur_fcy_phase[0] - (cur_lap - 1.0), 0.0)

                    # break loop since we assume that not more than one phase can be active per lap
                    break

            # tire_age_tmp (done after FCY handling to be able to consider decreased aging during FCY phase) -----------
            if cur_lap > 1 and basic_strategy_info[-1][0] == cur_lap - 1:
                # reset tire age in case of a tire change
                tire_age_tmp = 0.0

            if fcy_types_tmp is None:
                tire_age_tmp += 1.0
            elif fcy_types_tmp == 'SC':
                tire_age_tmp += lap_frac_normal + mult_tiredeg_sc * (1.0 - lap_frac_normal)
            elif fcy_types_tmp == 'VSC':
                tire_age_tmp += lap_frac_normal + mult_tiredeg_fcy * (1.0 - lap_frac_normal)
            else:
                raise ValueError("Unknown FCY type!")

            # used2compounds -------------------------------------------------------------------------------------------
            used_2compounds_tmp = len({x[1] for x in basic_strategy_info}) > 1

            # ----------------------------------------------------------------------------------------------------------
            # GET TIRECHANGE DECISION ----------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            next_compound = self.decide_pitstop(driver_initials=[driver.initials],
                                                cur_compounds=[basic_strategy_info[-1][1]],
                                                no_past_tirechanges=[len(basic_strategy_info) - 1],
                                                tire_ages=[tire_age_tmp],
                                                positions_prevlap=np.ones(1),
                                                pit_prevlap=[tirechange_tmp],
                                                cur_lap=cur_lap,
                                                tot_no_laps=tot_no_laps,
                                                fcy_types=[fcy_types_tmp],
                                                fcy_start_end_progs=[fcy_start_end_progs_tmp],
                                                bool_driving=np.ones(1, dtype=np.bool),
                                                bool_driving_prevlap=np.ones(1, dtype=np.bool),
                                                racetimes_prevlap=np.zeros(1),
                                                location=location,
                                                used_2compounds=[used_2compounds_tmp])[0]

            if next_compound is not None:
                basic_strategy_info.append([cur_lap, next_compound, 0, 0.0])

        # reset VSE for subsequent race simulation
        self.reset()

        return basic_strategy_info


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
