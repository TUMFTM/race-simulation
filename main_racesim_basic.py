import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import racesim_basic
import os
import pkg_resources
import helper_funcs.src.calc_tire_degradation

"""
author:
Alexander Heilmeier

date:
18.09.2019

.. description::
This script calculates the best race strategy (wihtout regarding traffic on the race track) for a given maximum number
of pit stops on the basis of the fitted parameters. The main influence is the tire degradation model.

Attention:
- Refueling (fuel or energy) is not optimized at the moment since this is not relevant anymore for many racing series.
- The QP optimization only works for a linear tire model and without FCY phases. This is due to the fact that it
  basically optimizes the stint lengths to minimize solely the tire degradation time losses.
"""

# ----------------------------------------------------------------------------------------------------------------------
# CHECK PYTHON DEPENDENCIES --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# get repo path
repo_path_ = os.path.dirname(os.path.abspath(__file__))

# read dependencies from requirements.txt
requirements_path = os.path.join(repo_path_, 'requirements.txt')
dependencies = []

with open(requirements_path, 'r') as fh_:
    line = fh_.readline()

    while line:
        dependencies.append(line.rstrip())
        line = fh_.readline()

# check dependencies
pkg_resources.require(dependencies)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def main(sim_opts: dict, pars_in: dict) -> tuple:

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE ALL POSSIBLE TIRE COMPOUND COMBINATIONS -------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    strategy_combinations = helper_funcs.src.get_strat_combinations.\
        get_strat_combinations(available_compounds=pars_in['available_compounds'],
                               min_no_pitstops=sim_opts["min_no_pitstops"],
                               max_no_pitstops=sim_opts["max_no_pitstops"],
                               enforce_diff_compounds=sim_opts["enforce_diff_compounds"],
                               start_compound=sim_opts["start_compound"],
                               all_orders=False)

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    exit_qp = False             # used to exit QP loops in the case that no solution was found in the MIQP
    t_race_fastest = {}         # t_race_fastest = {cur_no_pitstops: [(strategy), racetime]}
    t_race_full_factorial = {}

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE RACE TIMES (QP) ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if sim_opts["use_qp"]:
        # iterate over all desired numbers of pitstops
        for cur_no_pitstops in range(sim_opts["min_no_pitstops"], sim_opts["max_no_pitstops"] + 1):
            # iterate over all possible strategy combinations with cur_no_pitstops
            t_race_fastest[cur_no_pitstops] = []

            for cur_comp_strat in strategy_combinations[cur_no_pitstops]:
                # calculate optimal stint lengths using the QP
                tires = [[comp, 0] for comp in cur_comp_strat]
                tires[0][1] = sim_opts["start_age"]

                opt_stint_lengths = racesim_basic.src.opt_strategy_basic.\
                    opt_strategy_basic(tot_no_laps=pars_in['race_pars']['tot_no_laps'],
                                       tire_pars=pars_in['driver_pars']["tire_pars"],
                                       tires=tires)

                # if no solution was found exit QP and use full factorial instead
                if opt_stint_lengths is None:
                    print("INFO: Could not find a solution using the QP, moving to full factorial instead!")
                    exit_qp = True
                    t_race_fastest = {}  # reset value
                    break

                # set up strategy and calculate final race time
                laps_tmp = 0
                strategy = []           # [[inlap, compound, age, refueling], ...]
                strategy_stints = []    # [stint_length, compound, stint_length, compound, ...]

                for i in range(cur_no_pitstops + 1):
                    strategy.append([laps_tmp,          # inlap
                                     tires[i][0],       # set next compound
                                     tires[i][1],       # [-] tire age
                                     0.0])              # [kg or kWh] refueling during pit stop
                    strategy_stints.extend([opt_stint_lengths[i], tires[i][0]])
                    laps_tmp += opt_stint_lengths[i]

                t_race_tmp = racesim_basic.src.calc_racetimes_basic.\
                    calc_racetimes_basic(t_base=pars_in['driver_pars']["t_base"],
                                         tot_no_laps=pars_in['race_pars']["tot_no_laps"],
                                         t_lap_sens_mass=pars_in['track_pars']["t_lap_sens_mass"],
                                         t_pitdrive_inlap=pars_in['track_pars']["t_pitdrive_inlap"],
                                         t_pitdrive_outlap=pars_in['track_pars']["t_pitdrive_outlap"],
                                         t_pitdrive_inlap_fcy=pars_in['track_pars']["t_pitdrive_inlap_fcy"],
                                         t_pitdrive_outlap_fcy=pars_in['track_pars']["t_pitdrive_outlap_fcy"],
                                         t_pitdrive_inlap_sc=pars_in['track_pars']["t_pitdrive_inlap_sc"],
                                         t_pitdrive_outlap_sc=pars_in['track_pars']["t_pitdrive_outlap_sc"],
                                         t_pit_tirechange=pars_in['driver_pars']["t_pit_tirechange"],
                                         pits_aft_finishline=pars_in['track_pars']["pits_aft_finishline"],
                                         tire_pars=pars_in['driver_pars']["tire_pars"],
                                         p_grid=pars_in['driver_pars']["p_grid"],
                                         t_loss_pergridpos=pars_in['track_pars']["t_loss_pergridpos"],
                                         t_loss_firstlap=pars_in['track_pars']["t_loss_firstlap"],
                                         strategy=strategy,
                                         drivetype=pars_in['driver_pars']["drivetype"],
                                         m_fuel_init=pars_in['driver_pars']["m_fuel_init"],
                                         b_fuel_perlap=pars_in['driver_pars']["b_fuel_perlap"],
                                         t_pit_refuel_perkg=pars_in['driver_pars']["t_pit_refuel_perkg"],
                                         t_pit_charge_perkwh=pars_in['driver_pars']["t_pit_charge_perkwh"],
                                         fcy_phases=None,
                                         t_lap_sc=pars_in['track_pars']["t_lap_sc"],
                                         t_lap_fcy=pars_in['track_pars']["t_lap_fcy"])[0][-1]

                t_race_fastest[cur_no_pitstops].append([tuple(strategy_stints), t_race_tmp])

            # if no solution was found exit QP and use full factorial instead
            if exit_qp:
                break

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING (QP) ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

        if not exit_qp:
            # sort t_race_fastest by race times
            for cur_no_pitstops in t_race_fastest:
                t_race_fastest[cur_no_pitstops] = sorted(t_race_fastest[cur_no_pitstops], key=lambda x: x[1])

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE RACE TIMES (FULL FACTORIAL) ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not sim_opts["use_qp"] or exit_qp:
        # iterate over all desired numbers of pitstops
        for cur_no_pitstops in range(sim_opts["min_no_pitstops"], sim_opts["max_no_pitstops"] + 1):
            # create n-D array fitting for cur_no_pitstops: first dimension = first stop, second dimension = second stop
            # etc
            t_race_template = np.zeros((pars_in['race_pars']["tot_no_laps"] - 1,) * cur_no_pitstops)

            # copy template for every possible strategy with cur_no_pitstops
            t_race_full_factorial[cur_no_pitstops] = {cur_comp_strat: np.copy(t_race_template)
                                                      for cur_comp_strat in strategy_combinations[cur_no_pitstops]}

            # iterate over all possible strategy combinations with cur_no_pitstops
            for cur_comp_strat in t_race_full_factorial[cur_no_pitstops]:

                # iterate over all inlap combinations with cur_no_pitstops to calculate race time when doing the stop in
                # the according laps (tot_no_laps - 1 is not included as race must not be finished in pit)
                for idxs_cur_inlaps in itertools.product(range(pars_in['race_pars']["tot_no_laps"] - 1),
                                                         repeat=cur_no_pitstops):
                    # check if inlaps appear in a rising order
                    if not all([x < y for x, y in zip(idxs_cur_inlaps, idxs_cur_inlaps[1:])]):
                        t_race_full_factorial[cur_no_pitstops][cur_comp_strat][idxs_cur_inlaps] = np.nan
                        continue

                    # set up strategy and calculate final race time [[inlap, compound, age, refueling], ...]
                    strategy = [[0, cur_comp_strat[0], sim_opts["start_age"], 0.0]]

                    for i in range(cur_no_pitstops):
                        strategy.append([idxs_cur_inlaps[i] + 1,    # inlap = idx + 1
                                         cur_comp_strat[i + 1],     # set next compound
                                         0,                         # [-] tire age
                                         0.0])                      # [kg or kWh] refueling during pit stop

                    t_race_full_factorial[cur_no_pitstops][cur_comp_strat][idxs_cur_inlaps] = racesim_basic.src. \
                        calc_racetimes_basic.calc_racetimes_basic(t_base=pars_in['driver_pars']["t_base"],
                                                                  tot_no_laps=pars_in['race_pars']["tot_no_laps"],
                                                                  t_lap_sens_mass=pars_in['track_pars'][
                                                                      "t_lap_sens_mass"],
                                                                  t_pitdrive_inlap=pars_in['track_pars'][
                                                                      "t_pitdrive_inlap"],
                                                                  t_pitdrive_outlap=pars_in['track_pars'][
                                                                      "t_pitdrive_outlap"],
                                                                  t_pitdrive_inlap_fcy=pars_in['track_pars'][
                                                                      "t_pitdrive_inlap_fcy"],
                                                                  t_pitdrive_outlap_fcy=pars_in['track_pars'][
                                                                      "t_pitdrive_outlap_fcy"],
                                                                  t_pitdrive_inlap_sc=pars_in['track_pars'][
                                                                      "t_pitdrive_inlap_sc"],
                                                                  t_pitdrive_outlap_sc=pars_in['track_pars'][
                                                                      "t_pitdrive_outlap_sc"],
                                                                  pits_aft_finishline=pars_in['track_pars'][
                                                                      "pits_aft_finishline"],
                                                                  t_pit_tirechange=pars_in['driver_pars'][
                                                                      "t_pit_tirechange"],
                                                                  tire_pars=pars_in['driver_pars']["tire_pars"],
                                                                  p_grid=pars_in['driver_pars']["p_grid"],
                                                                  t_loss_pergridpos=pars_in['track_pars'][
                                                                      "t_loss_pergridpos"],
                                                                  t_loss_firstlap=pars_in['track_pars'][
                                                                      "t_loss_firstlap"],
                                                                  strategy=strategy,
                                                                  drivetype=pars_in['driver_pars']["drivetype"],
                                                                  m_fuel_init=pars_in['driver_pars']["m_fuel_init"],
                                                                  b_fuel_perlap=pars_in['driver_pars']["b_fuel_perlap"],
                                                                  t_pit_refuel_perkg=pars_in['driver_pars'][
                                                                      "t_pit_refuel_perkg"],
                                                                  t_pit_charge_perkwh=pars_in['driver_pars'][
                                                                      "t_pit_charge_perkwh"],
                                                                  fcy_phases=sim_opts["fcy_phases"],
                                                                  t_lap_sc=pars_in['track_pars']["t_lap_sc"],
                                                                  t_lap_fcy=pars_in['track_pars']["t_lap_fcy"])[0][-1]

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING (FULL FACTORIAL) ----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

        # find fastest stint lengths for every compound combination
        for cur_no_pitstops in t_race_full_factorial:
            t_race_fastest[cur_no_pitstops] = []

            for cur_comp_strat in t_race_full_factorial[cur_no_pitstops]:
                # get index of fastest race time
                idx_tmp = np.nanargmin(t_race_full_factorial[cur_no_pitstops][cur_comp_strat])

                # get inlap indices
                opt_inlap_idxs = np.unravel_index(idx_tmp, t_race_full_factorial[cur_no_pitstops][cur_comp_strat].shape)

                # calculate stint lengths from inlap indices
                laps_tmp = 0
                opt_stint_lengths = []

                for i in range(cur_no_pitstops):
                    opt_stint_lengths.append(opt_inlap_idxs[i] + 1 - laps_tmp)  # inlap = idx + 1
                    laps_tmp += opt_stint_lengths[-1]

                opt_stint_lengths.append(pars_in['race_pars']['tot_no_laps'] - laps_tmp)

                # set together strategy stints [stint_length, compound, stint_length, compound, ...]
                strategy_stints = []

                for tmp in zip(opt_stint_lengths, cur_comp_strat):
                    strategy_stints.extend(list(tmp))

                # get race time
                t_race_tmp = t_race_full_factorial[cur_no_pitstops][cur_comp_strat][opt_inlap_idxs]

                # save data
                t_race_fastest[cur_no_pitstops].append([tuple(strategy_stints), t_race_tmp])

        # sort t_race_fastest by race times
        for cur_no_pitstops in t_race_fastest:
            t_race_fastest[cur_no_pitstops] = sorted(t_race_fastest[cur_no_pitstops], key=lambda x: x[1])

    return t_race_fastest, t_race_full_factorial


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set race parameter file name
    race_pars_file_ = 'pars_YasMarina_2017.ini'

    # min_no_pitstops:          set minimum number of pitstops (mostly 1)
    # max_no_pitstops:          set maximum number of pitstops
    # start_compound:           enforce that the given start compound is included (set None if it is free)
    # start_age:                age of start tire set
    # enforce_diff_compounds:   enforce that at least two different compounds must be used in the race
    # use_qp:                   activate quadratic optim. to determine the optimal inlaps -> requires linear model, is
    #                           fast, reduced plotting
    # fcy_phases:               either None or [[start race progress, stop race progress, phase type], [...], ...]
    #                           -> only considered in full factorial calculations, not in QP!
    #                           -> start and stop race progress must be in range [0.0, tot_no_laps] (e.g. if SC comes
    #                           at 30% of the first lap and leaves at the end of lap 2 it would be [[0.3, 2.0, 'SC']])
    #                           -> valid FCY phase types are 'SC' and 'VSC'

    sim_opts_ = {"min_no_pitstops": 1,
                 "max_no_pitstops": 2,
                 "start_compound": None,
                 "start_age": 0,
                 "enforce_diff_compounds": True,
                 "use_qp": False,
                 "fcy_phases": None}

    # use_plot:                 set if plotting should be used or not (will be shown up to max. 2 stops)
    # use_print:                set if prints to console should be used or not (does not suppress hints/warnings)
    # use_print_result:         set if result should be printed to console or not

    use_plot = False
    use_print = True
    use_print_result = True

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load parameters
    pars_in_ = racesim_basic.src.import_pars.import_pars(use_print=use_print, race_pars_file=race_pars_file_)

    # check parameters
    racesim_basic.src.check_pars.check_pars(sim_opts=sim_opts_, pars_in=pars_in_, use_plot=use_plot)

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    t_start = time.perf_counter()

    t_race_fastest_, t_race_full_factorial_ = main(sim_opts=sim_opts_, pars_in=pars_in_)

    if use_print:
        print('INFO: Calculation time: %.3fs' % (time.perf_counter() - t_start))

    # ------------------------------------------------------------------------------------------------------------------
    # PRINT RESULTS ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # print resulting order and stint lengths (pit stop laps do not make sense as other orders are equally fast)
    if use_print_result:
        print('RESULT: Printing stint lengths instead of inlaps in the following because stint order is not relevant!')

        for cur_no_pitstops_, strategies_cur_no_pitstops in t_race_fastest_.items():
            print('RESULT: Race times for %i stop strategies:' % cur_no_pitstops_)

            for strategy_ in strategies_cur_no_pitstops:
                # set together print string
                print_string = ''

                for entry in strategy_[0]:
                    print_string += str(entry) + ' '

                print(print_string + ': %.3fs' % strategy_[1])

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_plot:
        # basic tire degradation plot ----------------------------------------------------------------------------------
        stint_length = 25

        t_c1_degr = helper_funcs.src.calc_tire_degradation. \
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=pars_in_['available_compounds'][0],
                                  tire_pars=pars_in_['driver_pars']["tire_pars"])

        t_c2_degr = helper_funcs.src.calc_tire_degradation. \
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=pars_in_['available_compounds'][1],
                                  tire_pars=pars_in_['driver_pars']["tire_pars"])

        t_c3_degr = helper_funcs.src.calc_tire_degradation. \
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=pars_in_['available_compounds'][2],
                                  tire_pars=pars_in_['driver_pars']["tire_pars"])

        # plot
        fig = plt.figure()
        ax = fig.gca()

        laps_tmp_ = np.arange(1, stint_length + 1)
        ax.plot(laps_tmp_, t_c1_degr)
        ax.plot(laps_tmp_, t_c2_degr, 'x-')
        ax.plot(laps_tmp_, t_c3_degr, 'o-')

        x_min = 0
        x_max = laps_tmp_[-1] - 1
        ax.set_xlim(left=x_min, right=x_max)
        plt.hlines((t_c1_degr[0], t_c2_degr[0], t_c3_degr[0]), x_min, x_max, color='grey', linestyle='--')

        # set title and axis labels
        plt.legend(pars_in_['available_compounds'])
        plt.title('Tire degradation plot')
        plt.ylabel('(Relative) Time loss in s/lap')
        plt.xlabel('Tire age in laps')

        plt.grid()
        plt.show()

        # plot 1 stop strategies ---------------------------------------------------------------------------------------
        if not sim_opts_["use_qp"]:
            for cur_comp_strat_ in t_race_full_factorial_[1]:
                fig = plt.figure()
                ax = fig.gca()

                laps_tmp_ = np.arange(1, pars_in_['race_pars']["tot_no_laps"] + 1)
                # -1 as race must not be finished in pit
                ax.plot(laps_tmp_[:-1], t_race_full_factorial_[1][cur_comp_strat_])

                t_race_min = np.amin(t_race_full_factorial_[1][cur_comp_strat_])
                plt.title('Current strategy: ' + str(cur_comp_strat_) + '\nMinimum race time: %.3fs' % t_race_min)
                plt.xlabel('Lap of pitstop')
                plt.ylabel('Race time in s')

                plt.grid()
                plt.show()

            # plot 2 stop strategies -----------------------------------------------------------------------------------
            for cur_comp_strat_ in t_race_full_factorial_[2]:
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                laps_tmp_ = np.arange(1, pars_in_['race_pars']["tot_no_laps"] + 1)
                x_, y_ = np.meshgrid(laps_tmp_[:-1], laps_tmp_[:-1])  # -1 as race must not be finished in pit
                ax.plot_wireframe(x_, y_, t_race_full_factorial_[2][cur_comp_strat_])

                t_race_min = np.nanmin(t_race_full_factorial_[2][cur_comp_strat_])
                plt.title('Current strategy: ' + str(cur_comp_strat_) + '\nMinimum race time: %.3fs' % t_race_min)
                plt.ylabel('Lap of first pitstop')
                plt.xlabel('Lap of second pitstop')
                ax.set_zlabel('Race time in s')

                plt.show()

            if sim_opts_["max_no_pitstops"] > 2:
                print('INFO: Plotting of strategies with more than 2 stops is not possible!')
