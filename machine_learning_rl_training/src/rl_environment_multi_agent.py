"""
author:
André Thomaser

refactored by:
Alexander Heilmeier

date:
01.11.2020

.. description::
The functions in this file are related to the reinforcement learning (RL) approach taken to train a Virtual Strategy
Engineer (VSE) within the race simulation to make reasonable strategy decisions. The main script is located in
main_train_rl_agent_dqn.py.
"""

import racesim
import machine_learning
import math
import numpy as np
import tf_agents.environments
import tf_agents.specs
import tf_agents.trajectories
import tf_agents.utils


class RaceSimulation(tf_agents.environments.py_environment.PyEnvironment):
    """
    Description:
        This environment is based on the race simulation. The race simulation calculates the lap times for each driver.
        A pit stop can be executed after each lap of the race. The goal is to finish the race in the shortest possible
        time and with the best possible final position by choosing the right time for the tire change and right tire
        compound.

        Note: This environment is batched. It takes an action for each driver. After simulating one lap it gives back
        one time step for each driver with observation and reward.

    Observation (method: __calculate_observation()):
        cur_lap:                    [-] current lap
        position:                   [-] current position
        est_pos_loss:               [-] number of positions the driver would lose during a pit stop in this lap
        tire_age:                   [-] tire age in laps
        cur_compound:               [-] current compound
        used_2compounds:            [-] at least two different compounds used during the race
        fcy_stat_curlap:            [-] FCY phase in current lap
                                            0: no SC or VSC active (not more than 5% before the end of the lap)
                                            1: VSC started within this lap (at least 5% before the end of the lap)
                                            2: VSC active and didn't start within this lap
                                            3: SC started within this lap (at least 5% before the end of the lap)
                                            4: SC active and didn't start within this lap
        close_behind:               [-] interval is <= 1.5 s
        close_ahead:                [-] ahead is <= 1.5 s
        defendable_undercut:        [-] pit stop of pursuer in the previous lap, undercut possible and defendable
        driver_initials:            [-] driver initials

    Actions (method: _step()): (for example: available compounds are 'A2', 'A3', 'A5' (sorted from hard to soft))
        0   no pit stop in this lap
        1   pit stop in this lap and choose 'A2' as next compound
        2   pit stop in this lap and choose 'A3' as next compound
        3   pit stop in this lap and choose 'A5' as next compound

    Reward:
        Lap time (method: __calculate_rewards_laptime()): Reward for the lap time since the previous pit stop decision
        Position (method: __calculate_rewards_position()): Reward for positions won/lost since the previous pit stop
            decision
        Final position (method: __calculate_rewards_final_position()): Reward for the final position

    Starting state (method: _reset()):
        First pit stop decision after the first lap

    Episode termination (method: _step()):
        The race is finished when the number of laps reaches the total number of laps of the race.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = (
        # environment inputs
        "__use_prob_infl",        # boolean to set if probabilistic influences should be activated
        "__create_rand_events",   # boolean to activate the random creation of FCY (full course yellow) phases and
                                  # retirements in the race if the according entries in the parameter file
                                  # contain empty lists, otherwise the file entries are used
        "__race_pars_file",       # file with parameters for the race such as race, track, car parameters, ...
        "__mcs_pars_file",        # file with parameters for the probabilities of fcy-phases, accidents, failures, ...
        # further parameters race simulation ---------------------------------------------------------------------------
        "__pars_in",                     # simulation parameters for the race, created from pars_files
        "__all_driver_initials",         # list with all driver names participating in the race
        "__available_compounds",         # list with all available dry compounds in the race
        "__average_laptimes_presim",     # is calculated for the driver controlled by the agent in a pre-simulation
        "__idx_driver_behind_prev_lap",  # index of the driver behind the driver at the end of the previous lap
        # race object --------------------------------------------------------------------------------------------------
        "__race",                        # race object
        # preprocessor -------------------------------------------------------------------------------------------------
        "__cat_preprocessor",            # preprocessor for categorical features
        # tf-agents ----------------------------------------------------------------------------------------------------
        "_action_spec",                  # action spec of the environment
        "_observation_spec",             # observation spec of a time step
        "_episode_ended",                # true if race is finished
        "_current_time_step",            # current time step with the observation and reward values
        "_batch_size")                   # batch size of the environment (= number of drivers)

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 race_pars_file: str,
                 mcs_pars_file: str,
                 use_prob_infl: bool = True,
                 create_rand_events: bool = True) -> None:

        # save race simulation parameters ------------------------------------------------------------------------------
        self.race_pars_file = race_pars_file
        self.mcs_pars_file = mcs_pars_file
        self.use_prob_infl = use_prob_infl
        self.create_rand_events = create_rand_events

        # race simulation ----------------------------------------------------------------------------------------------
        # parameters for the race object, created from the pars_files in the create_race() method
        self.pars_in = None

        # create race object
        self.race = self.__create_race()

        # available tire compounds for this race without rain tire compounds
        self.available_compounds = self.race.vse.vse_pars["param_dry_compounds"]

        # get all drivers participating in the race for feature driver and sort initials
        self.all_driver_initials = [self.race.drivers_list[idx].initials for idx in range(self.race.no_drivers)]
        self.all_driver_initials.sort()

        # calculate the lowest possible average lap times for each driver
        self.average_laptimes_presim = self.race.execute_presim_average_laptimes()

        # cache for driver behind previous lap
        self.idx_driver_behind_prev_lap = [None] * self.race.no_drivers

        # categorical preprocessor -------------------------------------------------------------------------------------
        # there are 7 categorical features
        self.cat_preprocessor = machine_learning.src.preprocessor.Preprocessor(feature_types=['cat'] * 7, bins_buck=[])

        # fit string type categorical features
        self.cat_preprocessor.fit_cat_dict(X_cat_str=self.available_compounds, featurename='cur_compound')
        self.cat_preprocessor.fit_cat_dict(X_cat_str=self.all_driver_initials, featurename='driver_initials')

        # fitting of cat_preprocessor must be performed with all possible categories such that classes are
        # determined correctly
        no_rows = max(len(self.available_compounds), 2, 5, len(self.all_driver_initials))
        X_cat_fit = np.zeros((no_rows, 7))

        # set first and last column (cur_compound, driver_initials) to 1.0 since preprocessor starts at 1.0 when
        # converting string type features (does not change the result because categorical features are inserted
        # one-hot encoded into the NN)
        X_cat_fit[:, 0] = 1.0
        X_cat_fit[:, 6] = 1.0

        X_cat_fit[:len(self.available_compounds), 0] = \
            self.cat_preprocessor.transform_cat_dict(X_cat_str=self.available_compounds, featurename='cur_compound')
        X_cat_fit[0, 1] = True
        X_cat_fit[:5, 2] = list(range(0, 5))
        X_cat_fit[0, 3] = True
        X_cat_fit[0, 4] = True
        X_cat_fit[0, 5] = True
        X_cat_fit[:len(self.all_driver_initials), 6] = \
            self.cat_preprocessor.transform_cat_dict(X_cat_str=self.all_driver_initials, featurename='driver_initials')

        self.cat_preprocessor.fit(X=X_cat_fit)

        # tf-agent -----------------------------------------------------------------------------------------------------
        self._action_spec = tf_agents.specs.array_spec.BoundedArraySpec(shape=(),
                                                                        dtype=np.int32,
                                                                        minimum=0,
                                                                        maximum=len(self.available_compounds),
                                                                        name='action')

        self._observation_spec = tf_agents.specs.array_spec.\
            BoundedArraySpec(shape=(17 + len(self.available_compounds) + len(self.all_driver_initials),),
                             dtype=np.float32,
                             minimum=0.0,
                             maximum=1.0,
                             name='observation')

        self._episode_ended = False
        self.batch_size = self.race.no_drivers

        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_use_prob_infl(self): return self.__use_prob_infl
    def __set_use_prob_infl(self, x): self.__use_prob_infl = x
    use_prob_infl = property(__get_use_prob_infl, __set_use_prob_infl)

    def __get_create_rand_events(self): return self.__create_rand_events
    def __set_create_rand_events(self, x): self.__create_rand_events = x
    create_rand_events = property(__get_create_rand_events, __set_create_rand_events)

    def __get_race_pars_file(self): return self.__race_pars_file
    def __set_race_pars_file(self, x): self.__race_pars_file = x
    race_pars_file = property(__get_race_pars_file, __set_race_pars_file)

    def __get_mcs_pars_file(self): return self.__mcs_pars_file
    def __set_mcs_pars_file(self, x): self.__mcs_pars_file = x
    mcs_pars_file = property(__get_mcs_pars_file, __set_mcs_pars_file)

    def __get_pars_in(self): return self.__pars_in
    def __set_pars_in(self, x): self.__pars_in = x
    pars_in = property(__get_pars_in, __set_pars_in)

    def __get_available_compounds(self): return self.__available_compounds
    def __set_available_compounds(self, x): self.__available_compounds = x
    available_compounds = property(__get_available_compounds, __set_available_compounds)

    def __get_all_driver_initials(self): return self.__all_driver_initials
    def __set_all_driver_initials(self, x): self.__all_driver_initials = x
    all_driver_initials = property(__get_all_driver_initials, __set_all_driver_initials)

    def __get_average_laptimes_presim(self): return self.__average_laptimes_presim
    def __set_average_laptimes_presim(self, x): self.__average_laptimes_presim = x
    average_laptimes_presim = property(__get_average_laptimes_presim, __set_average_laptimes_presim)

    def __get_idx_driver_behind_prev_lap(self): return self.__idx_driver_behind_prev_lap
    def __set_idx_driver_behind_prev_lap(self, x): self.__idx_driver_behind_prev_lap = x
    idx_driver_behind_prev_lap = property(__get_idx_driver_behind_prev_lap, __set_idx_driver_behind_prev_lap)

    def __get_race(self) -> racesim.src.race_reinftrain.RaceReinftrain: return self.__race
    def __set_race(self, x: racesim.src.race_reinftrain.RaceReinftrain): self.__race = x
    race = property(__get_race, __set_race)

    def __get_cat_preprocessor(self) -> machine_learning.src.preprocessor.Preprocessor: return self.__cat_preprocessor
    def __set_cat_preprocessor(self, x: machine_learning.src.preprocessor.Preprocessor): self.__cat_preprocessor = x
    cat_preprocessor = property(__get_cat_preprocessor, __set_cat_preprocessor)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def batched(self):
        return True

    def batch_size(self):
        return self.batch_size

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def __create_race(self) -> racesim.src.race_reinftrain.RaceReinftrain:
        """
        This method returns the race object created with the inserted parameter_files and the simulation options.
        """

        # set simulation options
        sim_opts = {"use_prob_infl": self.use_prob_infl,
                    "create_rand_events": self.create_rand_events,
                    "use_vse": True,
                    "no_sim_runs": 1,
                    "no_workers": 1,
                    "use_print": False,
                    "use_print_result": False,
                    "use_plot": False}

        # load and change parameters -----------------------------------------------------------------------------------
        self.pars_in, vse_paths = racesim.src.import_pars.import_pars(use_print=False,
                                                                      use_vse=True,
                                                                      race_pars_file=self.race_pars_file,
                                                                      mcs_pars_file=self.mcs_pars_file)

        # set vse_type (-> determines which VSE type is used for the drivers)
        for driver in self.pars_in['vse_pars']['vse_type']:
            self.pars_in['vse_pars']['vse_type'][driver] = 'reinforcement_training'

        # clear FCY phases and retirements from the parameters such that they are created randomly
        self.pars_in['event_pars']['fcy_data']['phases'] = []
        self.pars_in['event_pars']['retire_data']['retirements'] = []

        # check parameters
        racesim.src.check_pars.check_pars(sim_opts=sim_opts, pars_in=self.pars_in)

        # create race object -------------------------------------------------------------------------------------------
        race = racesim.src.race_reinftrain.RaceReinftrain(race_pars=self.pars_in["race_pars"],
                                                          driver_pars=self.pars_in["driver_pars"],
                                                          car_pars=self.pars_in["car_pars"],
                                                          tireset_pars=self.pars_in["tireset_pars"],
                                                          track_pars=self.pars_in["track_pars"],
                                                          vse_pars=self.pars_in["vse_pars"],
                                                          vse_paths=vse_paths,
                                                          use_prob_infl=sim_opts["use_prob_infl"],
                                                          create_rand_events=sim_opts["create_rand_events"],
                                                          monte_carlo_pars=self.pars_in["monte_carlo_pars"],
                                                          event_pars=self.pars_in["event_pars"])

        # no retirements for all drivers (for the training)
        race.retire_data['retirements'] = [None] * race.no_drivers

        return race

    def __calculate_observation(self, idx_driver: int) -> np.ndarray:
        """
        This method reads, calculates and processes the features for the observation that is passed to the agent.
        """

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE CREATION ---------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # read features from object race -------------------------------------------------------------------------------
        cur_lap = self.race.cur_lap
        position = self.race.positions[cur_lap, idx_driver]
        tire_age = self.race.drivers_list[idx_driver].car.tireset.age_degr
        cur_compound = self.race.drivers_list[idx_driver].car.tireset.compound
        team = self.race.drivers_list[idx_driver].car.manufacturer
        strategy_info = self.race.drivers_list[idx_driver].strategy_info

        # calculate used_2compounds ------------------------------------------------------------------------------------
        used_2compounds = True if len({x[1] for x in strategy_info}) > 1 else False

        # calculate fcy_stat_curlap ------------------------------------------------------------------------------------
        idx_act_phase_rel = self.race.fcy_handling["idxs_act_phase"][idx_driver]
        fcy_type = self.race.fcy_data["phases"][idx_act_phase_rel][2] if idx_act_phase_rel is not None else None
        fcy_stat_curlap = 0

        # a status other than 0 is only set if a FCY phase is active for the current driver
        if fcy_type is not None:
            fcy_start_end_progs = self.race.fcy_handling["start_end_prog"][idx_driver]

            # FCY phase is only relevant for the decision if it was active at least 5% before the end of the lap and
            # if it is still active at the end of the lap
            if cur_lap - fcy_start_end_progs[0] >= 0.05 and math.isclose(fcy_start_end_progs[1], cur_lap):
                # CASE 1: VSC phase
                if fcy_type == 'VSC':
                    # check if phase started within this lap or not and set according status
                    if cur_lap - fcy_start_end_progs[0] <= 1.0:
                        fcy_stat_curlap = 1
                    else:
                        fcy_stat_curlap = 2

                # CASE 2: SC phase
                else:
                    # check if phase started within this lap or not and set according status
                    if cur_lap - fcy_start_end_progs[0] <= 1.0:
                        fcy_stat_curlap = 3
                    else:
                        fcy_stat_curlap = 4

        # calculate est_pos_loss ---------------------------------------------------------------------------------------
        # minimal pit stop standstill time to change tires
        t_lost = self.pars_in['track_pars']['t_pit_tirechange_min']

        # team-specific additional standstill time to change tires
        t_lost += self.pars_in['car_pars'][team]['t_pit_tirechange_add']

        # time loss driving through the pit (depending on FCY status)
        if fcy_stat_curlap == 0:
            t_lost += (self.pars_in['track_pars']['t_pitdrive_inlap'] + self.pars_in['track_pars']['t_pitdrive_outlap'])
        elif fcy_stat_curlap in [1, 2, 3]:
            t_lost += (self.pars_in['track_pars']['t_pitdrive_inlap_fcy']
                       + self.pars_in['track_pars']['t_pitdrive_outlap_fcy'])
        else:
            t_lost += (self.pars_in['track_pars']['t_pitdrive_inlap_sc']
                       + self.pars_in['track_pars']['t_pitdrive_outlap_sc'])

        racetimes_tmp = self.race.racetimes[self.race.cur_lap - 1] + self.race.laptimes[self.race.cur_lap]

        # number of drivers which are t_lost behind reinforcement driver
        est_pos_loss = len([x for x in racetimes_tmp
                            if racetimes_tmp[idx_driver] < x <= (racetimes_tmp[idx_driver] + t_lost)])

        # calculate close_ahead and close_behind -----------------------------------------------------------------------
        bool_driving = self.race.bool_driving[self.race.cur_lap]
        positions = self.race.positions[self.race.cur_lap]

        if positions[idx_driver] != 1:
            idx_driver_ahead = [idx for idx in range(len(positions))
                                if positions[idx] == (positions[idx_driver] - 1)][0]
            interval = racetimes_tmp[idx_driver] - racetimes_tmp[idx_driver_ahead]
        else:
            # no other driver ahead
            interval = np.inf

        if positions[idx_driver] == self.race.no_drivers:
            ahead = np.inf
        else:
            idx_driver_behind = [idx for idx in range(len(positions))
                                 if positions[idx] == (positions[idx_driver] + 1)][0]
            if not bool_driving[idx_driver_behind]:
                # pretend there is no driver behind if driver behind retired
                ahead = np.inf
            else:
                ahead = racetimes_tmp[idx_driver_behind] - racetimes_tmp[idx_driver]

        close_behind = interval <= 1.5
        close_ahead = ahead <= 1.5

        # calculate defendable_undercut --------------------------------------------------------------------------------
        defendable_undercut = False

        # undercut only possible if driver behind is in pit_driver_idxs
        if self.idx_driver_behind_prev_lap[idx_driver] in self.race.pit_driver_idxs:
            # gap between driver and driver behind minus the time a pit stop would cost
            gap_aft_stop = (racetimes_tmp[self.idx_driver_behind_prev_lap[idx_driver]] - racetimes_tmp[idx_driver]
                            - t_lost)

            # undercut is assumed to be possible and defendable if driver behind is within 10s after an imaginary
            # stop of the driver
            if 0.0 < gap_aft_stop <= 10.0:
                defendable_undercut = True

        # save driver behind previous lap for next step
        if positions[idx_driver] == self.race.no_drivers:
            # pursuer only available if driver is not on the last position
            self.idx_driver_behind_prev_lap[idx_driver] = None
        else:
            self.idx_driver_behind_prev_lap[idx_driver] = next((idx for idx, pos in enumerate(positions)
                                                                if pos == positions[idx_driver] + 1), None)

        # --------------------------------------------------------------------------------------------------------------
        # PREPARE CATEGORICAL FEATURES ---------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # prepare X_cat (categorical features only) -> bools are implicitly converted to floats
        X_cat = np.zeros(7)
        X_cat[0] = self.cat_preprocessor.transform_cat_dict(X_cat_str=[cur_compound], featurename='cur_compound')[0]
        X_cat[1] = used_2compounds
        X_cat[2] = fcy_stat_curlap
        X_cat[3] = close_behind
        X_cat[4] = close_ahead
        X_cat[5] = defendable_undercut
        X_cat[6] = self.cat_preprocessor.transform_cat_dict(X_cat_str=[self.race.drivers_list[idx_driver].initials],
                                                            featurename='driver_initials')[0]

        # --------------------------------------------------------------------------------------------------------------
        # CREATE OBSERVATION -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        observation = np.zeros(4 + self.cat_preprocessor.no_transf_cols, dtype=np.float32)

        # preprocessing (numerical features)
        observation[0] = cur_lap / self.race.race_pars["tot_no_laps"]
        observation[1] = (position - 1) / (self.race.no_drivers - 1)
        observation[2] = est_pos_loss / (self.race.no_drivers - 1)
        observation[3] = tire_age / self.race.race_pars["tot_no_laps"]

        # preprocessing (categorical features) -> one-hot encoding
        observation[4:] = self.cat_preprocessor.transform(X=X_cat, dtype_out=np.float32)

        return observation

    def __calculate_observations(self) -> np.ndarray:
        """
        This method returns the observations for all drivers.
        """

        observations = [self.__calculate_observation(idx_driver=idx) for idx in range(self.race.no_drivers)]
        return np.array(observations)

    def __calculate_rewards_laptime(self, t_pitdrive_inlap: np.ndarray) -> np.ndarray:
        """
        This method returns an reward defined as the difference between the driver's lap time and an average lap time
        (reward = average lap time - driver's lap time). The lap time for laps with SC/VSC phases is given for all
        drivers. The average lap time is calculated from the weighted sum of the lowest possible average lap time
        without SC/VSC from a pre-simulation and the lap time with SC/VSC. The weighting factor depends on the ratio of
        SC/VSC-phase and no SC/VSC-phase in the lap:
            average_laptime = delta * laptime_fcy + (1 - delta) * average_laptime_presim

        Pre-simulation: Simulates a race for each driver and returns the lowest possible average lap time with no other
                        drivers on the track and no SC/VSC-phases. The first lap is not considered because the agent has
                        no impact on the first lap.
        t_pitdrive_inlap: Lap time loss because of entering the pit, is added to the lap time after a pit stop decision.
                          It must also be added to the next lap time to take into account the time lost due to a pit
                          stop in the previous lap when calculating the reward.
        """

        cur_lap = self.race.cur_lap
        rewards = np.zeros(self.race.no_drivers)

        for idx_driver in range(self.race.no_drivers):
            # add t_pitdrive_inlap because of a pit stop in the previous lap to the current lap time
            cur_laptime = self.race.laptimes[cur_lap, idx_driver] + t_pitdrive_inlap[idx_driver]

            # distinguish between SC/VSC and no SC/VSC
            idx_act_phase_rel = self.race.fcy_handling["idxs_act_phase"][idx_driver]
            fcy_type = self.race.fcy_data["phases"][idx_act_phase_rel][2] if idx_act_phase_rel is not None else None

            if fcy_type is None:
                # no SC/VSC in this lap
                reward = self.average_laptimes_presim[idx_driver] - cur_laptime
            else:
                fcy_start_end_progs = self.race.fcy_handling["start_end_prog"][idx_driver]

                # determine lap fraction driven normally, delta = duration of FCY phase in this lap (0.0 < delta <= 1.0)
                if cur_lap - 1 >= fcy_start_end_progs[0] and math.isclose(fcy_start_end_progs[1], cur_lap):
                    # SC/VSC is active at the beginning and the end of the lap
                    delta = 1.0
                else:
                    # SC/VSC is not active at the beginning or the end of the lap
                    if cur_lap - 1 >= fcy_start_end_progs[0]:
                        # SC/VSC was active at the beginning of the lap
                        delta = 1 - (cur_lap - fcy_start_end_progs[1])
                    else:
                        delta = fcy_start_end_progs[1] - fcy_start_end_progs[0]

                # average laptime = delta * laptime_fcy + (1.0 - delta) * average_laptime_presim
                average_laptime = (delta * self.race.track.t_lap_fcy
                                   + (1.0 - delta) * self.average_laptimes_presim[idx_driver])
                reward = np.round(average_laptime - cur_laptime, 6)

                if fcy_type == 'SC':
                    racetime_tmp = (self.race.racetimes[cur_lap - 1, idx_driver]
                                    + self.race.laptimes[cur_lap, idx_driver])

                    sc_ghost_incl_min_dist = (self.race.fcy_handling["sc_ghost_racetimes"][idx_driver]
                                              + self.race.positions[cur_lap, idx_driver]
                                              * self.race.race_pars['min_t_dist_sc'])

                    if racetime_tmp <= sc_ghost_incl_min_dist:
                        # driver is behind SC and his laptime is now equal to ghost SC
                        reward = 0.0

            rewards[idx_driver] = reward

        return rewards

    def __calculate_rewards_position(self, delta_position: np.ndarray) -> np.ndarray:
        """
        This method returns an reward of +5.0/-5.0 for each position the driver won/lost until the next pit stop
        decision.
        delta_position: Positions lost or won between pit stop entrance and end of the previous lap.
        """

        rewards = (self.race.positions[self.race.cur_lap - 1]
                   - self.race.positions[self.race.cur_lap]
                   + delta_position)

        return 5.0 * rewards

    def __calculate_rewards_final_position(self) -> np.ndarray:
        """
        This method returns an extra reward at the end of the race depending on the final position of the driver and
        his average lap time from the pre-simulation compared to the other drivers who finished the race.
        """

        rewards = np.zeros(self.race.no_drivers)

        # average lap times from the pre-simulation for the drivers who finish the race
        average_laptimes = {self.race.drivers_list[idx].initials: tmp_laptime
                            for idx, tmp_laptime in enumerate(self.average_laptimes_presim)
                            if self.race.bool_driving[self.race.race_pars["tot_no_laps"]][idx]}

        # get driver initials in order of rising average lap times
        driver_initials_sorted = [k for k, v in sorted(average_laptimes.items(), key=lambda x: x[1])]

        for idx_driver in range(self.race.no_drivers):
            # get position of driver according to average lap times of the drivers finishing the race
            position_avg_laptime = driver_initials_sorted.index(self.race.drivers_list[idx_driver].initials) + 1

            # final race position
            position_race = self.race.positions[self.race.get_last_compl_lap(idx=idx_driver), idx_driver]

            # caculate reward
            rewards[idx_driver] = position_avg_laptime - position_race

        return rewards

    def _reset(self) -> tf_agents.trajectories.time_step.TimeStep:
        """
        This method resets the race, creates a new race object, and returns the observation at the first pit stop
        decision.
        """

        # new episode
        self._episode_ended = False

        # create new race
        self.race = self.__create_race()

        # cache for driver behind previous lap
        self.idx_driver_behind_prev_lap = [None] * self.race.no_drivers

        # simulate beginning of first lap until first pit stop decision
        self.race._RaceReinftrain__simulate_lap_start()

        # get the observations after the first lap and before first pit stop decision
        observations = self.__calculate_observations()
        time_steps = [tf_agents.trajectories.time_step.restart(observation=observations[idx])
                      for idx in range(self.race.no_drivers)]

        return tf_agents.utils.nest_utils.stack_nested_arrays(nested_arrays=time_steps)

    def _step(self, actions) -> tf_agents.trajectories.time_step.TimeStep:
        """
        This method returns the next time step with the observation and reward depending on the chosen action.
        Action == 0: no pit stop, action > 0: pit stop and choose available_compounds[action + 1] as next compound
        Example: available compounds are 'A2', 'A3', 'A5'
            0   no pit stop in this lap
            1   pit stop in this lap and choose 'A2' as next compound
            2   pit stop in this lap and choose 'A3' as next compound
            3   pit stop in this lap and choose 'A5' as next compound
        """

        if self._episode_ended:
            # the previous action ended the episode, therefore ignore the current action and start a new episode
            return self.reset()

        # evaluate actions ---------------------------------------------------------------------------------------------
        positions_before_simulate_lap = self.race.positions[self.race.cur_lap]
        laptimes_before_pitstop = self.race.laptimes[self.race.cur_lap]

        for idx_driver in range(self.race.no_drivers):
            if actions[idx_driver] in [1, 2, 3]:
                # pit stop in this lap, compound = available_compounds[action - 1]
                self.race.drivers_list[idx_driver].strategy_info.\
                    append([self.race.cur_lap, self.available_compounds[actions[idx_driver] - 1], 0, 0.0])

        # simulate next lap
        self.race._RaceReinftrain__simulate_lap()

        # positions lost or won during pit stop entrance and end of the lap
        position_after_simulate_lap = self.race.positions[self.race.cur_lap - 1]
        delta_position = positions_before_simulate_lap - position_after_simulate_lap

        # time lost because of the pit stop in the lap of the pit stop
        laptimes_after_pitstop = self.race.laptimes[self.race.cur_lap - 1]
        t_pitdrive_inlap = laptimes_after_pitstop - laptimes_before_pitstop

        # next observations and rewards --------------------------------------------------------------------------------
        observations = self.__calculate_observations()

        # multiply reward by 0.1 to decrease reward-values
        rewards = 0.1 * (self.__calculate_rewards_laptime(t_pitdrive_inlap)
                         + self.__calculate_rewards_position(delta_position))

        # check if race ended ------------------------------------------------------------------------------------------
        self._episode_ended = self.race.cur_lap >= self.race.race_pars["tot_no_laps"]

        if self._episode_ended:
            # simulate end of last lap and perform post race actions
            self.race._RaceReinftrain__simulate_lap_end()

            # rewards for the final positions
            rewards += self.__calculate_rewards_final_position()

            for idx_driver in range(self.race.no_drivers):
                # at least two different compounds used during the race
                used_2compounds = (True if len({x[1] for x in self.race.drivers_list[idx_driver].strategy_info}) > 1
                                   else False)

                if not used_2compounds:
                    # if all used tires during the race are the same compound add negative reward as punishment
                    rewards[idx_driver] -= 10.0

            # postprocessing
            if self.race.result_status != 0:
                print("WARNING: Simulation result was invalid!")

            time_steps = [tf_agents.trajectories.time_step.termination(observation=observations[idx],
                                                                       reward=rewards[idx])
                          for idx in range(self.race.no_drivers)]

            return tf_agents.utils.nest_utils.stack_nested_arrays(nested_arrays=time_steps)

        else:
            time_steps = [tf_agents.trajectories.time_step.transition(observation=observations[idx],
                                                                      reward=rewards[idx],
                                                                      discount=1.0)
                          for idx in range(self.race.no_drivers)]

            return tf_agents.utils.nest_utils.stack_nested_arrays(nested_arrays=time_steps)
