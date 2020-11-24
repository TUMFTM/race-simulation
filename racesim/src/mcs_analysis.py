import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def mcs_analysis(race_results: list,
                 use_print_result: bool,
                 use_plot: bool):
    """
    author:
    Alexander Heilmeier

    date:
    01.03.2019

    .. description::
    This function either creates bar plots showing the distribution of final driver positions after the simulated races
    or box plots for these distributions if number of bunches is > 1.

    .. inputs:
    :param race_results:        list containing the result dicts of the simulated races (from race.race_results())
    :type race_results:         list
    :param use_print_result:    determines if result prints to console should be created or not
    :type use_print_result:     bool
    :param use_plot:            determines if plots should be created or not
    :type use_plot:             bool
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPROCESSING ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not type(race_results) is list and type(race_results[0]) is dict:
        raise RuntimeError("List of dicts required as result_objs (list of results from race.race_results())!")

    no_sim_runs = len(race_results)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE DATAFRAME CONTAINING THE PROCESSED RESULT POSITIONS -------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create pandas dataframe in the form [driver_initials, no_pos1, no_pos2, ...] for cumulated results
    driver_initials = list(race_results[0]["driverinfo"].keys())
    no_drivers = len(driver_initials)
    col_names = ["no_pos" + str(i) for i in range(1, no_drivers + 1)]

    race_results_df = pd.DataFrame(np.zeros((no_drivers, no_drivers), dtype=np.int32),
                                   columns=col_names,
                                   index=driver_initials)

    # count number of positions for current driver
    for initials in driver_initials:
        tmp_pos_results = [0] * no_drivers

        for idx_race in range(no_sim_runs):
            cur_result_pos = int(race_results[idx_race]["driverinfo"][initials]["positions"][-1])
            tmp_pos_results[cur_result_pos - 1] += 1

        # add tmp_results to driver's row in pandas dataframe
        race_results_df.loc[initials] = tmp_pos_results

    # ------------------------------------------------------------------------------------------------------------------
    # PRINT MEAN POSITIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_print_result:
        print("RESULT: Mean positions after %i simulation runs..." % no_sim_runs)
        mean_posis = []

        for idx_driver in range(no_drivers):
            no_pos = race_results_df.iloc[idx_driver].values
            mean_pos = np.sum(no_pos * np.arange(1, no_drivers + 1)) / no_sim_runs

            mean_posis.append([driver_initials[idx_driver], mean_pos])

        # sort list by mean position
        mean_posis.sort(key=lambda x: x[1])

        # print list
        for entry in mean_posis:
            print("RESULT: %s: %.1f" % (entry[0], entry[1]))

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_plot:
        subplots_per_row = 3
        no_rows = int(np.ceil(no_drivers / subplots_per_row))

        fig, axes = plt.subplots(no_rows, subplots_per_row, sharex="all", sharey="all")

        # create one subplot per driver
        cur_col = 0
        cur_row = 0

        for idx_driver in range(no_drivers):
            # use bar plots to show position distributions
            axes[cur_row][cur_col].\
                bar(range(1, no_drivers + 1),
                    list(race_results_df.iloc[idx_driver] / no_sim_runs * 100.0),
                    tick_label=range(1, no_drivers + 1))

            # add driver initials above plot
            axes[cur_row][cur_col].set_title(driver_initials[idx_driver])

            # set ylabel for first plot per row
            if cur_col == 0:
                axes[cur_row][cur_col].set_ylabel("percentage")

            # set xlabel for last plot per column
            if cur_row == no_rows - 1:
                axes[cur_row][cur_col].set_xlabel("rank position")

            # count up column and row indices
            cur_col += 1
            if cur_col >= subplots_per_row:
                cur_col = 0
                cur_row += 1

        fig.suptitle("Distribution of final positions (%i simulated races)" % no_sim_runs)
        plt.show()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
