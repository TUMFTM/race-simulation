import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os


class RaceAnalysis(object):
    """
    author:
    Alexander Heilmeier

    date:
    11.06.2019

    .. description::
    Dummy class for outsourced methods related to race analysis.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SUPPORT FUNCTIONS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def check_valid_result(self) -> None:
        if not self.result_status == 0:
            print("WARNING: Race was either not fully simulated or marked as invalid, status %i!" % self.result_status)

    def __get_raceprogress(self, racetime: float, idx_driver: int) -> tuple:
        # find last driven lap
        last_driven_lap = self.get_last_compl_lap(idx=idx_driver)

        # find relevant lap for given racetime
        lap_rel = np.argmax(self.racetimes[:last_driven_lap + 1, idx_driver] > racetime)

        # CASE 1: normal case
        if lap_rel > 0:
            cur_lap = lap_rel
            est_fraction = (racetime - self.racetimes[lap_rel - 1, idx_driver]) / self.laptimes[lap_rel, idx_driver]
            last_laptime = self.laptimes[lap_rel - 1, idx_driver]
            last_racetime = self.racetimes[lap_rel - 1, idx_driver]

        # CASE 2: driver completed race before the given racetime
        elif lap_rel == 0 and last_driven_lap == self.race_pars["tot_no_laps"]:
            cur_lap = self.race_pars["tot_no_laps"]
            est_fraction = 1.0
            last_laptime = self.laptimes[self.race_pars["tot_no_laps"], idx_driver]
            last_racetime = self.racetimes[self.race_pars["tot_no_laps"], idx_driver]

        # CASE 3: driver retired before reaching the given racetime
        else:
            cur_lap = last_driven_lap + 1
            est_fraction = 0.0
            last_laptime = self.laptimes[last_driven_lap, idx_driver]
            last_racetime = self.racetimes[last_driven_lap, idx_driver]

        return cur_lap, est_fraction, last_laptime, last_racetime

    def get_race_results(self) -> dict:
        """
        Return a dict that contains all relevant states of the race for further analysis.
        """

        # check if race is finished and valid
        if not self.cur_lap == self.race_pars["tot_no_laps"]:
            print("WARNING: Race used for analysis is not fully simulated!")
        elif self.result_status != 0:
            print("WARNING: Result status %i indicates invalid race used for analysis!" % self.result_status)

        # create results dict
        results = {'driverinfo': {}}

        # add driver-specific information
        initials_list = [x.initials for x in self.drivers_list]

        for idx, initials in enumerate(initials_list):
            results['driverinfo'][initials] = {"carno": self.drivers_list[idx].carno,                   # int
                                               "strategy_info": self.drivers_list[idx].strategy_info,   # list of lists
                                               "team": self.drivers_list[idx].team,                     # str
                                               "racetimes": self.racetimes[:, idx],                     # np.ndarray
                                               "positions": self.positions[:, idx],                     # np.ndarray
                                               "bool_driving": self.bool_driving[:, idx],               # np.ndarray
                                               "progress": float(self.progress[idx]),                   # float
                                               "retirements": self.retire_data["retirements"][idx]}     # None/float

        # add general race information
        results['fcy_phases'] = self.fcy_data["phases"]                                                 # list of lists

        return results

    # ------------------------------------------------------------------------------------------------------------------
    # CONSOLE OUTPUT ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def print_details(self) -> None:
        driver_initials = [cur_driver.initials for cur_driver in self.drivers_list]

        laptimes_str = np.array2string(self.laptimes, precision=3)
        racetimes_str = np.array2string(self.racetimes, precision=3)
        positions_str = np.array2string(self.positions)

        print("RESULT: Drivers:\n" + str(driver_initials))
        print("RESULT: Laptimes:\n" + laptimes_str)
        print("RESULT: Racetimes:\n" + racetimes_str)
        print("RESULT: Positions:\n" + positions_str)

    def print_result(self) -> None:
        # print pre-simulation base strategy if using VSE (virtual strategy engineer)
        if self.presim_info["base_strategy_vse"] is not None:
            print("RESULT: Pre-simulation base strategy (by VSE) for %s: %s"
                  % (self.monte_carlo_pars['ref_driver'], str(self.presim_info["base_strategy_vse"])))

        # print pre-simulation race duration
        if self.presim_info["race_duration"] is not None:
            print("RESULT: Pre-simulation race duration for %s: %.3fs"
                  % (self.monte_carlo_pars['ref_driver'], self.presim_info["race_duration"]))

        # get driver initials and car numbers
        driver_initials = [cur_driver.initials for cur_driver in self.drivers_list]
        carnos = [cur_driver.carno for cur_driver in self.drivers_list]

        # get best laptimes (must be done driver by driver to avoid numpy warning for all nan slice)
        best_t_laps = np.full(self.no_drivers, np.nan)

        for idx in range(self.no_drivers):
            # all values nan often appears since we often have a SC + driver accident in the first lap
            if not np.all(np.isnan(self.laptimes[1:, idx])):
                best_t_laps[idx] = np.nanmin(self.laptimes[1:, idx])

        # get number of laps
        no_laps = np.zeros(self.no_drivers, dtype=np.int)

        for idx in range(self.no_drivers):
            no_laps[idx] = self.get_last_compl_lap(idx=idx)

        # get gaps to leader
        gaps = []
        idx_leader = np.argmax(self.positions[-1] == 1)

        for idx in range(self.no_drivers):
            if no_laps[idx] == no_laps[idx_leader]:
                gaps.append(self.racetimes[-1, idx] - self.racetimes[-1, idx_leader])

            else:
                gaps.append('%il' % (no_laps[idx_leader] - no_laps[idx]))

        # get interval to driver in front
        ints = []

        for idx in range(self.no_drivers):
            if self.positions[-1, idx] == 1:
                ints.append(0.0)
                continue

            idx_pos_front = np.argmax(self.positions[-1] == self.positions[-1, idx] - 1)

            if no_laps[idx] == no_laps[idx_pos_front]:
                ints.append(self.racetimes[no_laps[idx], idx] - self.racetimes[no_laps[idx_pos_front], idx_pos_front])

            else:
                ints.append('%il' % (no_laps[idx_pos_front] - no_laps[idx]))

        # get status of drivers at end of race (F for finished, DNF for did not finish)
        status = []

        for idx in range(self.no_drivers):
            if np.any(~self.bool_driving[:, idx]):
                status.append('DNF')
            else:
                status.append('F')

        # get strategy info (inlaps + compounds) as string -> [1:-1] to remove outer brackets in string
        strategy_info = [str([cur_stop[:2] for cur_stop in cur_driver.strategy_info])[1:-1]
                         for cur_driver in self.drivers_list]

        # convert to pandas
        tmp_pos = pd.DataFrame(self.positions[-1], columns=['pos'], index=driver_initials, dtype=np.int)
        tmp_carno = pd.DataFrame(carnos, columns=['carno'], index=driver_initials, dtype=np.int)
        tmp_t_race = pd.DataFrame(self.racetimes[no_laps, np.arange(self.no_drivers)], columns=['t_race'],
                                  index=driver_initials)
        tmp_gap = pd.DataFrame(gaps, columns=['gap'], index=driver_initials)
        tmp_int = pd.DataFrame(ints, columns=['int'], index=driver_initials)
        tmp_best_t_lap = pd.DataFrame(best_t_laps, columns=['best_t_lap'], index=driver_initials)
        tmp_no_laps = pd.DataFrame(no_laps, columns=['no_laps'], index=driver_initials, dtype=np.int)
        tmp_status = pd.DataFrame(status, columns=['status'], index=driver_initials, dtype=np.str)
        tmp_strategy_info = pd.DataFrame(strategy_info, columns=['strategy_info'], index=driver_initials, dtype=np.str)
        result = pd.concat((tmp_pos, tmp_carno, tmp_t_race, tmp_gap, tmp_int, tmp_best_t_lap, tmp_no_laps,
                            tmp_status, tmp_strategy_info), axis=1)

        # sort by position
        result.sort_values(by=["pos"], ascending=[True], inplace=True)

        # print result
        print("RESULT: Simulation result:")
        print(result.to_string(float_format='{:.3f}'.format))

        # print FCY phases
        if self.fcy_data["phases"]:
            str_tmp = "RESULT: FCY phases:"
            for cur_phase in self.fcy_data["phases"]:
                if cur_phase[2] == 'SC':
                    # for SC we have the end race time from the pre-calculation as well as from the simulation available
                    str_tmp += " [%s -> %.3fs - %.3fs/%.3fs (pre-calc./real SC end)]"\
                               % (cur_phase[2], cur_phase[0], cur_phase[1], cur_phase[5])
                else:
                    str_tmp += " [%s -> %.3fs - %.3fs]" % (cur_phase[2], cur_phase[0], cur_phase[1])
        else:
            str_tmp = "RESULT: No FCY phases!"
        print(str_tmp)

        # print retirements
        tmp_retirements = [[idx, x] for idx, x in enumerate(self.retire_data["retirements"]) if x is not None]
        tmp_retirements.sort(key=lambda x: x[1])

        for idx in range(len(tmp_retirements)):
            # replace driver index by initials
            tmp_retirements[idx][0] = self.drivers_list[tmp_retirements[idx][0]].initials

        if tmp_retirements:
            str_tmp = "RESULT: Retirements:"
            for cur_retirement in tmp_retirements:
                str_tmp += " [%s -> %.3fs]" % (cur_retirement[0], cur_retirement[1])
        else:
            str_tmp = "RESULT: No retirements!"
        print(str_tmp)

    def print_race_standings(self, racetime: float) -> None:
        # get driver initials for row names (index)
        driver_initials = [cur_driver.initials for cur_driver in self.drivers_list]

        # create empty leaderboard as pandas dataframe
        tmp_lap = pd.DataFrame(np.zeros(self.no_drivers, dtype=np.int), columns=["lap"], index=driver_initials)
        tmp_pos = pd.DataFrame(np.zeros(self.no_drivers, dtype=np.int), columns=["pos"], index=driver_initials)
        tmp_est_pos = pd.DataFrame(np.zeros(self.no_drivers, dtype=np.int), columns=["est_pos"], index=driver_initials)
        tmp_laptime = pd.DataFrame(np.zeros(self.no_drivers), columns=["last_laptime"], index=driver_initials)
        tmp_fraction = pd.DataFrame(np.zeros(self.no_drivers), columns=["est_fraction"], index=driver_initials)
        tmp_racetime = pd.DataFrame(np.zeros(self.no_drivers), columns=["last_racetime"], index=driver_initials)
        tmp_total = [tmp_lap, tmp_pos, tmp_est_pos, tmp_laptime, tmp_fraction, tmp_racetime]
        leaderboard = pd.concat(tmp_total, axis=1)

        # loop through all the drivers
        for idx in range(self.no_drivers):
            # get current lap, estimated lap fraction, last lap time and race time for every driver
            leaderboard.iloc[idx, [0, 4, 3, 5]] = self.__get_raceprogress(racetime=racetime,
                                                                          idx_driver=idx)

        # sort dataframe by lap and est_fraction and add estimated positions -> they show if a driver might overtake
        # during the current lap
        leaderboard.sort_values(by=["lap", "est_fraction"], ascending=[False, False], inplace=True)
        leaderboard.iloc[:, 2] = np.arange(self.no_drivers) + 1

        # sort dataframe by lap and last_racetime and add positions
        leaderboard.sort_values(by=["lap", "last_racetime"], ascending=[False, True], inplace=True)
        leaderboard.iloc[:, 1] = np.arange(self.no_drivers) + 1

        # print result
        print("RESULT: Leaderboard:")
        print(leaderboard)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def plot_raceprogress_over_racetime(self) -> None:
        # get maximum race time and ceil to next 10s step
        racetime_max = np.nanmax(self.racetimes)
        racetime_max = math.ceil(racetime_max / 10) * 10

        # get race progress of the drivers in 10s steps
        racetime_steps = np.arange(0.0, racetime_max + 10.0, 10.0)
        raceprogress = np.zeros((len(racetime_steps), self.no_drivers))  # array to save the values

        for idx_racetimestep, cur_racetime in enumerate(list(racetime_steps)):
            for idx_driver in range(self.no_drivers):
                cur_lap, est_fraction = self.__get_raceprogress(racetime=cur_racetime,
                                                                idx_driver=idx_driver)[:2]

                raceprogress[idx_racetimestep, idx_driver] = cur_lap - 1 + est_fraction

        # plot race progress over race time
        plot_data_x = []
        plot_data_y = []

        for idx_driver in range(self.no_drivers):
            plot_data_x.append(racetime_steps)
            plot_data_y.append(raceprogress[:, idx_driver])

        self.__plot_every_driver(plot_data_x=plot_data_x,
                                 plot_data_y=plot_data_y)

        # plot SC and VSC phases
        ax = plt.gca()
        self.__plot_fcy_phases_over_racetime(ax=ax)

        plt.xlabel("race time in s")
        plt.ylabel("race progress in laps")
        plt.title("race progress over race time")
        plt.grid()
        plt.show()

    def __plot_fcy_phases_over_laps(self, ax: plt.axes) -> None:
        # get y limits of axes
        y_lims = ax.get_ylim()

        # use flag state of race to plot a yellow rectangle for every FCY lap
        for lap in range(1, self.race_pars["tot_no_laps"] + 1):
            if self.flagstates[lap] in ['SC', 'VSC']:
                if self.flagstates[lap] == 'SC':
                    color = 'gold'
                else:
                    color = 'yellow'

                x_tmp = [lap - 0.5, lap - 0.5, lap + 0.5, lap + 0.5]
                y_tmp = [y_lims[0], y_lims[1], y_lims[1], y_lims[0]]
                ax.fill(x_tmp, y_tmp, color)

    def __plot_fcy_phases_over_racetime(self, ax: plt.axes) -> None:
        # get y limits of axes
        y_lims = ax.get_ylim()

        # use fcy_phases plot a yellow rectangle for every FCY phase
        for cur_phase in self.fcy_data["phases"]:
            if cur_phase[2] == 'SC':
                color = 'gold'
            else:
                color = 'yellow'

            x_tmp = [cur_phase[0], cur_phase[0], cur_phase[1], cur_phase[1]]
            y_tmp = [y_lims[0], y_lims[1], y_lims[1], y_lims[0]]
            ax.fill(x_tmp, y_tmp, color)

    def plot_laptimes(self) -> None:
        # plot laptimes vs laps
        self.__plot_every_driver_over_laps(plot_data=self.laptimes,
                                           jump_lap_zero=True)

        # plot SC and VSC phases
        ax = plt.gca()
        self.__plot_fcy_phases_over_laps(ax=ax)

        plt.xlabel("lap")
        plt.ylabel("laptime in s")
        plt.title("laptimes over laps")
        plt.grid()
        plt.show()

    def plot_positions(self) -> None:
        # plot positions vs laps
        self.__plot_every_driver_over_laps(plot_data=self.positions,
                                           jump_lap_zero=False)

        # plot SC and VSC phases
        ax = plt.gca()
        self.__plot_fcy_phases_over_laps(ax=ax)

        plt.gca().invert_yaxis()
        plt.xlabel("lap")
        plt.ylabel("position")
        plt.title("positions over laps")
        plt.grid()
        plt.show()

    def plot_racetime_diffto_refdriver(self, ref_position: int) -> None:
        # calculate racetime differences
        racetime_diff = np.zeros(self.racetimes.shape)

        ref_driver_b = self.positions[-1] == ref_position  # get driver in reference position in last lap
        for cur_lap in range(self.race_pars["tot_no_laps"] + 1):
            ref_driver_racetime = self.racetimes[cur_lap, ref_driver_b]
            racetime_diff[cur_lap] = self.racetimes[cur_lap] - ref_driver_racetime

        # plot racetime differences vs. laps
        self.__plot_every_driver_over_laps(plot_data=racetime_diff,
                                           jump_lap_zero=False)

        # plot SC and VSC phases
        ax = plt.gca()
        self.__plot_fcy_phases_over_laps(ax=ax)

        plt.xlabel("lap")
        plt.ylabel("racetime difference to reference driver in s")
        plt.title("racetime differences to reference driver over laps")
        plt.grid()
        plt.show()

    def plot_racetime_diffto_reflaptime(self, ref_laptime: float) -> None:
        # calculate racetime differences
        racetime_diff = np.zeros(self.racetimes.shape)

        for cur_lap in range(self.race_pars["tot_no_laps"] + 1):
            ref_racetime = ref_laptime * cur_lap
            racetime_diff[cur_lap] = self.racetimes[cur_lap] - ref_racetime

        # plot racetime differences vs. laps
        self.__plot_every_driver_over_laps(plot_data=racetime_diff,
                                           jump_lap_zero=False)

        # plot SC and VSC phases
        ax = plt.gca()
        self.__plot_fcy_phases_over_laps(ax=ax)

        plt.xlabel("lap")
        plt.ylabel("racetime difference to reference laptime in s")
        plt.title("racetime differences to reference laptime over laps")
        plt.grid()
        plt.show()

    def __plot_every_driver_over_laps(self,
                                      plot_data: np.ndarray,
                                      jump_lap_zero: bool = False) -> None:
        """This method can be used by other plot functions to plot data over laps for every driver. If jump_lap_zero is
        set it will start plotting with lap 1."""

        plot_data_x = []
        plot_data_y = []

        for idx_driver, cur_driver in enumerate(self.drivers_list):
            # get last driven lap of current driver
            last_driven_lap = self.get_last_compl_lap(idx=idx_driver)

            # if we start plotting in lap 1 we need to assure that driver has set a lap time
            if last_driven_lap < 1 and jump_lap_zero:
                plot_data_x.append(None)
                plot_data_y.append(None)
                continue

            # get plot data
            if jump_lap_zero:
                lap_start = 1
            else:
                lap_start = 0

            plot_data_x.append(np.arange(lap_start, last_driven_lap + 1))
            plot_data_y.append(plot_data[lap_start:last_driven_lap + 1, idx_driver])

        self.__plot_every_driver(plot_data_x=plot_data_x,
                                 plot_data_y=plot_data_y)

    def __plot_every_driver(self,
                            plot_data_x: list,
                            plot_data_y: list) -> None:
        """This method can be used by other plot functions to plot the data in plot_data_y over the data in plot_data_x
        (e.g. laps) for every driver."""

        initials = []
        line_handles = []
        teams_tmp = {}  # used to count the number of drivers per team with an equal team color
        markers = ['.-', '+-', 'h-', '1-', '>-', '*-']  # different markers are used for drivers in same team

        for idx_driver, cur_driver in enumerate(self.drivers_list):
            # continue if no data available for current driver
            if plot_data_x[idx_driver] is None or plot_data_y[idx_driver] is None:
                continue

            # update number of drivers in team
            team_tmp = cur_driver.team

            if team_tmp in teams_tmp:
                teams_tmp[team_tmp] += 1
            else:
                teams_tmp[team_tmp] = 1

            # determine marker style
            if teams_tmp[team_tmp] <= len(markers):
                fmt = markers[teams_tmp[team_tmp] - 1]
            else:
                fmt = markers[0]

            # plot data
            line_handles.append(plt.plot(plot_data_x[idx_driver],
                                         plot_data_y[idx_driver],
                                         fmt,
                                         color=cur_driver.car.color,
                                         markersize=7)[0])

            # plot final values using a big marker
            plt.plot(plot_data_x[idx_driver][-1],
                     plot_data_y[idx_driver][-1],
                     fmt,
                     color=cur_driver.car.color,
                     markersize=15)

            # save current driver initials for legend
            initials.append(cur_driver.initials)

        # plot legend (done here since it only depends on the driver initials and line style)
        plt.legend(line_handles, initials)

    def export_results_as_csv(self, results_path: str) -> None:
        location = self.track.name
        season = self.race_pars["season"]

        racetimes_file_path = os.path.join(results_path, "%s_%i_racetimes.csv" % (location, season))
        laptimes_file_path = os.path.join(results_path, "%s_%i_laptimes.csv" % (location, season))
        positions_file_path = os.path.join(results_path, "%s_%i_positions.csv" % (location, season))

        initials = [cur_driver.initials for cur_driver in self.drivers_list]
        header = ",".join(['lap'] + initials)

        np.savetxt(racetimes_file_path,
                   np.column_stack((np.arange(1, self.race_pars["tot_no_laps"] + 1), self.racetimes[1:])),
                   fmt='%i' + ',%.3f' * self.no_drivers, header=header, comments='')
        np.savetxt(laptimes_file_path,
                   np.column_stack((np.arange(1, self.race_pars["tot_no_laps"] + 1), self.laptimes[1:])),
                   fmt='%i' + ',%.3f' * self.no_drivers, header=header, comments='')
        np.savetxt(positions_file_path,
                   np.column_stack((np.arange(0, self.race_pars["tot_no_laps"] + 1), self.positions)),
                   fmt='%i' + ',%i' * self.no_drivers, header=header, comments='')

        # save information about laps affected by SC, VSC, pitstop inlap, pitstop outlap
        lap_influences = []

        for cur_lap in range(1, self.race_pars["tot_no_laps"] + 1):
            lap_influences.append({})

            for idx_driver in range(self.no_drivers):
                cur_initials = initials[idx_driver]

                if cur_lap in self.drivers_list[idx_driver].lap_influences:
                    infl_tmp = self.drivers_list[idx_driver].lap_influences[cur_lap]
                    lap_influences[cur_lap - 1][cur_initials] = '+'.join(infl_tmp)
                else:
                    lap_influences[cur_lap - 1][cur_initials] = 'none'

        lapinfluences_file_path = os.path.join(results_path, "%s_%i_lapinfluences.csv" % (location, season))

        with open(lapinfluences_file_path, 'w') as fh:
            fh.write(header + "\n")

            for cur_lap in range(1, self.race_pars["tot_no_laps"] + 1):
                tmp_str = '%i' % cur_lap

                for cur_initials in lap_influences[cur_lap - 1]:
                    tmp_str = tmp_str + ",%s" % lap_influences[cur_lap - 1][cur_initials]

                fh.write(tmp_str + "\n")
