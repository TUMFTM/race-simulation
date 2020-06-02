import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics


def mcs_analysis(result_objs: list,
                 no_bunches: int,
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
    :param result_objs:         list containing the result dicts of the simulated races (from race.race_results())
    :type result_objs:          list
    :param no_bunches:          number of simulation bunches for statistical evaluation
    :type no_bunches:           int
    :param use_print_result:    determines if result prints to console should be created or not
    :type use_print_result:     bool
    :param use_plot:            determines if plots should be created or not
    :type use_plot:             bool
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPROCESSING ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check input
    if not type(result_objs) is list and type(result_objs[0]) is dict:
        raise ValueError("List of dicts required as result_objs (list of results from race.race_results())!")

    # get race results out of the result object list and divide them into bunches
    no_races_per_bunch = int(len(result_objs) / no_bunches)
    race_results = [result_objs[i * no_races_per_bunch:(i + 1) * no_races_per_bunch] for i in range(no_bunches)]

    # ------------------------------------------------------------------------------------------------------------------
    # DETERMINE MAGNITUDES FOR SUBSEQUENT STATISTICAL ANALYSIS ---------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if no_bunches > 1:
        # get order of magnitude (direct calculation using math.log shows numerical problems, therefore done "by hand")
        min_no_results = min([len(x) for x in race_results])

        if min_no_results < int(1e3):
            raise ValueError("For the statistical analysis there should be at least 1,000 valid races per bunch!")
        elif min_no_results < int(1e4):
            order_of_mag = 3
        elif min_no_results < int(1e5):
            order_of_mag = 4
        elif min_no_results < int(1e6):
            order_of_mag = 5
        elif min_no_results < int(1e7):
            order_of_mag = 6
        else:
            raise ValueError("Too many simulations, case is not implemented!")

        # create list of magnitudes for subsequent analysis
        no_magnitudes = 3  # user input -> determine how many magnitudes shell be analyzed
        magnitudes = []

        for i in range(order_of_mag - no_magnitudes + 1, order_of_mag + 1):
            magnitudes.append(int(math.pow(10.0, i)))

        if use_print_result:
            print("INFO: The following magnitudes will be analyzed with box plots: " + str(magnitudes))

    else:
        # no statistical analysis if there is only a single bunch of races
        magnitudes = [len(race_results[0])]

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE DICTIONARY CONTAINING THE PROCESSED RESULTING POSITIONS ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create template of pandas dataframe in the form [driver_initials no_pos1 no_pos2 ...] for cumulated results
    driver_initials = list(race_results[0][0]["driverinfo"].keys())
    no_drivers = len(driver_initials)
    col_names = ["no_pos" + str(i) for i in range(1, no_drivers + 1)]

    race_results_df = pd.DataFrame(np.zeros((no_drivers, no_drivers), dtype=np.int),
                                   columns=col_names,
                                   index=driver_initials)

    # create dictionary with an entry per bunch (which is again a dict with a pandas dataframe per magnitude)
    race_results_processed = dict.fromkeys(list(range(no_bunches)))

    for idx_bunch in race_results_processed:
        race_results_processed[idx_bunch] = dict.fromkeys(magnitudes)

        for idx_mag in race_results_processed[idx_bunch]:
            race_results_processed[idx_bunch][idx_mag] = race_results_df.copy(deep=True)

    # gather data
    for idx_bunch in range(no_bunches):
        for initials in driver_initials:
            # count number of positions for current driver in current bunch of races
            tmp_pos_results = [0] * no_drivers

            for idx_race in range(len(race_results[idx_bunch])):
                # count positions
                cur_result_pos = int(race_results[idx_bunch][idx_race]["driverinfo"][initials]["positions"][-1])
                tmp_pos_results[cur_result_pos - 1] += 1

                # if number of races reaches any of the magnitudes save current positions information
                if idx_race + 1 in magnitudes:
                    # determine dict key (current magnitude)
                    cur_mag = idx_race + 1

                    # add tmp_results to drivers row in pandas dataframe
                    race_results_processed[idx_bunch][cur_mag].loc[initials] = tmp_pos_results

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_plot:
        subplots_per_row = 3
        no_rows = int(np.ceil(no_drivers / subplots_per_row))

        # create one figure for every magnitude
        for cur_mag in magnitudes:
            fig, axes = plt.subplots(no_rows, subplots_per_row, sharex="all", sharey="all")

            # create one subplot per driver
            cur_col = 0
            cur_row = 0

            for idx_driver in range(no_drivers):
                # CASE 1: one bunch -> use bar plots to show position distributions
                if no_bunches == 1:
                    axes[cur_row][cur_col].\
                        bar(range(1, no_drivers + 1),
                            list(race_results_processed[0][cur_mag].iloc[idx_driver, :] / cur_mag * 100.0),
                            tick_label=range(1, no_drivers + 1))

                # CASE 2: several bunches -> use box plots to show additional information on the position distributions
                else:
                    # the box plot of every driver requires a list containing one sublist per rank position, which
                    # contains the according fraction for every bunch -> [[65%, 56%, 63%, ...] -> pos1, ...]
                    boxplot_data = []

                    for cur_pos in range(no_drivers):
                        boxplot_data.append([])

                        for idx_bunch in range(no_bunches):
                            boxplot_data[cur_pos].append(race_results_processed[idx_bunch][cur_mag].
                                                         iloc[idx_driver, cur_pos] / cur_mag * 100.0)

                    # create box plots (all data is valid -> do not assume outliers)
                    axes[cur_row][cur_col].boxplot(boxplot_data, whis="range")

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

            # set figure title
            if no_bunches == 1:
                fig.suptitle("Distribution of final positions (%i simulated races)" % cur_mag)
            else:
                fig.suptitle("Distribution of final positions (%i bunches with %i simulated races each)"
                             % (no_bunches, cur_mag))

            # show figure
            plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PRINT MEAN POSITION AND DEVIATION PER MAGNITUDE AND DRIVER -------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # it is desirable to determine the mean position and the deviation per driver and magnitude for statistical analysis
    if no_bunches > 1:
        for cur_mag in magnitudes:
            if use_print_result:
                print("RESULT: Mean positions and 95%% confidence deviations for %i races..." % cur_mag)

            for idx_driver in range(no_drivers):
                # determine mean position per bunch for current magnitude
                mean_pos_perbunch = []

                for idx_bunch in range(no_bunches):
                    no_pos_curbunch = race_results_processed[idx_bunch][cur_mag].iloc[idx_driver].values
                    mean_pos_curbunch = np.sum(no_pos_curbunch * np.arange(1, no_drivers + 1)) / cur_mag
                    mean_pos_perbunch.append(mean_pos_curbunch)

                # determine overall mean position and deviation for current magnitude
                mean_pos_permag = statistics.mean(mean_pos_perbunch)
                stdev_pos_permag = statistics.pstdev(mean_pos_perbunch, mu=mean_pos_permag)

                # determine overall deviation with 95% confidence for current magnitude -> 95% of the distribution is
                # within 1.96 standard deviations of the mean if it's a gauss distribution
                stdev_pos_permag_95 = 1.96 * stdev_pos_permag

                # print mean value and deviation (with 95% confidence)
                if use_print_result:
                    print("RESULT: %s: %.3f +- %.3f"
                          % (driver_initials[idx_driver], mean_pos_permag, stdev_pos_permag_95))


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
