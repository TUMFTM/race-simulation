import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import racesim_basic
import configparser
import os
import json
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
# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def main(sim_opts: dict) -> None:

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK PYTHON DEPENDENCIES ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get repo path
    repo_path = os.path.dirname(os.path.abspath(__file__))

    # read dependencies from requirements.txt
    requirements_path = os.path.join(repo_path, 'requirements.txt')
    dependencies = []

    with open(requirements_path, 'r') as fh:
        line = fh.readline()

        while line:
            dependencies.append(line.rstrip())
            line = fh.readline()

    # check dependencies
    pkg_resources.require(dependencies)

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set paths
    repo_path = os.path.dirname(os.path.abspath(__file__))
    par_file_path = os.path.join(repo_path, "racesim_basic", "input", "parameters", sim_opts["pars_file"])

    # load parameters
    parser = configparser.ConfigParser()

    if not parser.read(par_file_path):
        raise ValueError('Specified config file does not exist or is empty!')

    driver_pars = json.loads(parser.get('DRIVER_PARS', 'driver_pars'))
    track_pars = json.loads(parser.get('TRACK_PARS', 'track_pars'))
    race_pars = json.loads(parser.get('RACE_PARS', 'race_pars'))

    # determine some additionally required variables
    available_compounds = list(driver_pars["tire_pars"].keys())
    available_compounds.remove('tire_deg_model')
    if driver_pars["drivetype"] == "combustion" and driver_pars["b_fuel_perlap"] is None:
        # calculate approximate fuel consumption per lap
        driver_pars["b_fuel_perlap"] = driver_pars["m_fuel_init"] / race_pars["tot_no_laps"]
        if sim_opts["use_print"]:
            print("INFO: Fuel consumption was automatically determined to %.2fkg/lap!" % driver_pars["b_fuel_perlap"])

    # check user input
    if driver_pars['tire_pars']['tire_deg_model'] != 'lin' and sim_opts["use_qp"]:
        raise ValueError('QP is only available for a linear tire degradation model!')

    if sim_opts["use_plot"] and sim_opts["use_qp"]:
        print('INFO: Plotting will be reduced since the derived data from the QP is much less than for full factorial!')

    if not 0 <= sim_opts["min_no_pitstops"] < sim_opts["max_no_pitstops"]:
        raise ValueError('Minimum number of pit stops must be less than maximum number of pit stops and greater than'
                         ' 0!')

    if sim_opts["min_no_pitstops"] == 0 and sim_opts["enforce_diff_compounds"]:
        print('WARNING: Different compounds cannot be enforced if number of pitstops is zero!')

    if sim_opts["use_qp"] and sim_opts["fcy_phases"]:
        print("WARNING: FCY phases cannot be considered when using the quadratic optimization, they will therefore be"
              " neglected!")

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE ALL POSSIBLE TIRE COMPOUND COMBINATIONS -------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    t_start = time.perf_counter()
    strategy_combinations = {}

    for cur_no_pitstops in range(sim_opts["min_no_pitstops"], sim_opts["max_no_pitstops"] + 1):
        # combinations used since the chosen order of tire compounds does not matter for the final race time
        strategy_combinations[cur_no_pitstops] = list(itertools.combinations_with_replacement(available_compounds,
                                                                                              r=cur_no_pitstops + 1))

        # remove strategy combinations using only a single tire compound if enforced
        if sim_opts["enforce_diff_compounds"]:
            strategy_combinations[cur_no_pitstops] = [strat_tmp for strat_tmp in strategy_combinations[cur_no_pitstops]
                                                      if not len(set(strat_tmp)) == 1]

        # remove strategy combinations that do not include the starting tire compound if enforced
        if driver_pars["start_compound"] is not None:
            strategy_combinations[cur_no_pitstops] = [strat_tmp for strat_tmp in strategy_combinations[cur_no_pitstops]
                                                      if driver_pars["start_compound"] in strat_tmp]

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE RACE TIMES (QP) ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if sim_opts["use_qp"]:
        # iterate over all desired numbers of pitstops
        t_race_fastest = {}  # t_race_fastest = {cur_no_pitstops: [(strategy), racetime]}

        for cur_no_pitstops in range(sim_opts["min_no_pitstops"], sim_opts["max_no_pitstops"] + 1):
            # iterate over all possible strategy combinations with cur_no_pitstops
            t_race_fastest[cur_no_pitstops] = []

            for cur_comp_strat in strategy_combinations[cur_no_pitstops]:
                # calculate optimal stint lengths using the QP
                tires_start_age = 0
                tires = [list(a) for a in zip(cur_comp_strat,
                                              [tires_start_age] * len(cur_comp_strat))]  # [[compound, age], ...]

                opt_stint_lengths = racesim_basic.src.opt_strategy_basic.\
                    opt_strategy_basic(tot_no_laps=race_pars['tot_no_laps'],
                                       tire_pars=driver_pars["tire_pars"],
                                       tires=tires)

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
                    calc_racetimes_basic(t_base=driver_pars["t_base"],
                                         tot_no_laps=race_pars["tot_no_laps"],
                                         t_lap_sens_mass=track_pars["t_lap_sens_mass"],
                                         t_pitdrive_inlap=track_pars["t_pitdrive_inlap"],
                                         t_pitdrive_outlap=track_pars["t_pitdrive_outlap"],
                                         t_pitdrive_inlap_fcy=track_pars["t_pitdrive_inlap_fcy"],
                                         t_pitdrive_outlap_fcy=track_pars["t_pitdrive_outlap_fcy"],
                                         t_pitdrive_inlap_sc=track_pars["t_pitdrive_inlap_sc"],
                                         t_pitdrive_outlap_sc=track_pars["t_pitdrive_outlap_sc"],
                                         t_pit_tirechange=driver_pars["t_pit_tirechange"],
                                         pits_aft_finishline=track_pars["pits_aft_finishline"],
                                         tire_pars=driver_pars["tire_pars"],
                                         p_grid=driver_pars["p_grid"],
                                         t_loss_pergridpos=track_pars["t_loss_pergridpos"],
                                         t_loss_firstlap=track_pars["t_loss_firstlap"],
                                         strategy=strategy,
                                         drivetype=driver_pars["drivetype"],
                                         m_fuel_init=driver_pars["m_fuel_init"],
                                         b_fuel_perlap=driver_pars["b_fuel_perlap"],
                                         t_pit_refuel_perkg=driver_pars["t_pit_refuel_perkg"],
                                         t_pit_charge_perkwh=driver_pars["t_pit_charge_perkwh"],
                                         fcy_phases=None,
                                         t_lap_sc=track_pars["t_lap_sc"],
                                         t_lap_fcy=track_pars["t_lap_fcy"])[0][-1]

                t_race_fastest[cur_no_pitstops].append([tuple(strategy_stints), t_race_tmp])

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING (QP) ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

        # sort t_race_fastest by race times
        for cur_no_pitstops in t_race_fastest:
            t_race_fastest[cur_no_pitstops] = sorted(t_race_fastest[cur_no_pitstops], key=lambda x: x[1])

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE RACE TIMES (FULL FACTORIAL) ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    else:
        # iterate over all desired numbers of pitstops
        t_race_full_factorial = {}

        for cur_no_pitstops in range(sim_opts["min_no_pitstops"], sim_opts["max_no_pitstops"] + 1):
            # create n-D array fitting for cur_no_pitstops: first dimension = first stop, second dimension = second stop
            # etc
            t_race_template = np.zeros((race_pars["tot_no_laps"] - 1,) * cur_no_pitstops)

            # copy template for every possible strategy with cur_no_pitstops
            t_race_full_factorial[cur_no_pitstops] = {cur_comp_strat: np.copy(t_race_template)
                                                      for cur_comp_strat in strategy_combinations[cur_no_pitstops]}

            # iterate over all possible strategy combinations with cur_no_pitstops
            for cur_comp_strat in t_race_full_factorial[cur_no_pitstops]:

                # iterate over all inlap combinations with cur_no_pitstops to calculate race time when doing the stop in
                # the according laps (tot_no_laps - 1 is not included as race must not be finished in pit)
                for idxs_cur_inlaps in itertools.product(range(race_pars["tot_no_laps"] - 1), repeat=cur_no_pitstops):
                    # check if inlaps appear in a rising order
                    if not all([x < y for x, y in zip(idxs_cur_inlaps, idxs_cur_inlaps[1:])]):
                        t_race_full_factorial[cur_no_pitstops][cur_comp_strat][idxs_cur_inlaps] = np.nan
                        continue

                    # set up strategy and calculate final race time
                    strategy = [[0, cur_comp_strat[0], 0, 0.0]]  # [[inlap, compound, age, refueling], ...]

                    for i in range(cur_no_pitstops):
                        strategy.append([idxs_cur_inlaps[i] + 1,    # inlap = idx + 1
                                         cur_comp_strat[i + 1],     # set next compound
                                         0,                         # [-] tire age
                                         0.0])                      # [kg or kWh] refueling during pit stop

                    t_race_full_factorial[cur_no_pitstops][cur_comp_strat][idxs_cur_inlaps] = racesim_basic.src. \
                        calc_racetimes_basic.calc_racetimes_basic(t_base=driver_pars["t_base"],
                                                                  tot_no_laps=race_pars["tot_no_laps"],
                                                                  t_lap_sens_mass=track_pars["t_lap_sens_mass"],
                                                                  t_pitdrive_inlap=track_pars["t_pitdrive_inlap"],
                                                                  t_pitdrive_outlap=track_pars["t_pitdrive_outlap"],
                                                                  t_pitdrive_inlap_fcy=track_pars[
                                                                      "t_pitdrive_inlap_fcy"],
                                                                  t_pitdrive_outlap_fcy=track_pars[
                                                                      "t_pitdrive_outlap_fcy"],
                                                                  t_pitdrive_inlap_sc=track_pars["t_pitdrive_inlap_sc"],
                                                                  t_pitdrive_outlap_sc=track_pars[
                                                                      "t_pitdrive_outlap_sc"],
                                                                  pits_aft_finishline=track_pars["pits_aft_finishline"],
                                                                  t_pit_tirechange=driver_pars["t_pit_tirechange"],
                                                                  tire_pars=driver_pars["tire_pars"],
                                                                  p_grid=driver_pars["p_grid"],
                                                                  t_loss_pergridpos=track_pars["t_loss_pergridpos"],
                                                                  t_loss_firstlap=track_pars["t_loss_firstlap"],
                                                                  strategy=strategy,
                                                                  drivetype=driver_pars["drivetype"],
                                                                  m_fuel_init=driver_pars["m_fuel_init"],
                                                                  b_fuel_perlap=driver_pars["b_fuel_perlap"],
                                                                  t_pit_refuel_perkg=driver_pars["t_pit_refuel_perkg"],
                                                                  t_pit_charge_perkwh=driver_pars[
                                                                      "t_pit_charge_perkwh"],
                                                                  fcy_phases=sim_opts["fcy_phases"],
                                                                  t_lap_sc=track_pars["t_lap_sc"],
                                                                  t_lap_fcy=track_pars["t_lap_fcy"])[0][-1]

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING (FULL FACTORIAL) ----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

        # find fastest stint lengths for every compound combination
        t_race_fastest = {}

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

                opt_stint_lengths.append(race_pars['tot_no_laps'] - laps_tmp)

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

    # ------------------------------------------------------------------------------------------------------------------
    # PRINT RESULTS ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # print calculation time
    if sim_opts["use_print"]:
        print('INFO: Calculation time: %.3fs' % (time.perf_counter() - t_start))

    # print resulting order and stint lengths (pit stop laps do not make sense as other orders are equally fast)
    if sim_opts["use_print_result"]:
        print('RESULT: Printing stint lengths instead of inlaps in the following because stint order is not relevant!')

        for cur_no_pitstops, strategies_cur_no_pitstops in t_race_fastest.items():
            print('RESULT: Race times for %i stop strategies:' % cur_no_pitstops)

            for strategy in strategies_cur_no_pitstops:
                # set together print string
                print_string = ''

                for entry in strategy[0]:
                    print_string += str(entry) + ' '

                print(print_string + ': %.3fs' % strategy[1])

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if sim_opts["use_plot"]:
        # basic tire degradation plot ----------------------------------------------------------------------------------
        stint_length = 25

        t_c1_degr = helper_funcs.src.calc_tire_degradation.\
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=available_compounds[0],
                                  tire_pars=driver_pars["tire_pars"])

        t_c2_degr = helper_funcs.src.calc_tire_degradation.\
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=available_compounds[1],
                                  tire_pars=driver_pars["tire_pars"])

        t_c3_degr = helper_funcs.src.calc_tire_degradation.\
            calc_tire_degradation(tire_age_start=0,
                                  stint_length=stint_length,
                                  compound=available_compounds[2],
                                  tire_pars=driver_pars["tire_pars"])

        # plot
        fig = plt.figure()
        ax = fig.gca()

        laps_tmp = np.arange(1, stint_length + 1)
        ax.plot(laps_tmp, t_c1_degr)
        ax.plot(laps_tmp, t_c2_degr, 'x-')
        ax.plot(laps_tmp, t_c3_degr, 'o-')

        x_min = 0
        x_max = laps_tmp[-1] - 1
        ax.set_xlim(left=x_min, right=x_max)
        plt.hlines((t_c1_degr[0], t_c2_degr[0], t_c3_degr[0]), x_min, x_max, color='grey', linestyle='--')

        # set title and axis labels
        plt.legend(available_compounds)
        plt.title('Tire degradation plot')
        plt.ylabel('(Relative) Time loss in s/lap')
        plt.xlabel('Tire age in laps')

        plt.grid()
        plt.show()

        # plot 1 stop strategies ---------------------------------------------------------------------------------------
        if not sim_opts["use_qp"]:
            for cur_comp_strat in t_race_full_factorial[1]:
                fig = plt.figure()
                ax = fig.gca()

                laps_tmp = np.arange(1, race_pars["tot_no_laps"] + 1)
                # -1 as race must not be finished in pit
                ax.plot(laps_tmp[:-1], t_race_full_factorial[1][cur_comp_strat])

                t_race_min = np.amin(t_race_full_factorial[1][cur_comp_strat])
                plt.title('Current strategy: ' + str(cur_comp_strat) + '\nMinimum race time: %.3fs' % t_race_min)
                plt.xlabel('Lap of pitstop')
                plt.ylabel('Race time in s')

                plt.grid()
                plt.show()

        # plot 2 stop strategies ---------------------------------------------------------------------------------------
            for cur_comp_strat in t_race_full_factorial[2]:
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                laps_tmp = np.arange(1, race_pars["tot_no_laps"] + 1)
                x, y = np.meshgrid(laps_tmp[:-1], laps_tmp[:-1])  # -1 as race must not be finished in pit
                ax.plot_wireframe(x, y, t_race_full_factorial[2][cur_comp_strat])

                t_race_min = np.nanmin(t_race_full_factorial[2][cur_comp_strat])
                plt.title('Current strategy: ' + str(cur_comp_strat) + '\nMinimum race time: %.3fs' % t_race_min)
                plt.ylabel('Lap of first pitstop')
                plt.xlabel('Lap of second pitstop')
                ax.set_zlabel('Race time in s')

                plt.show()

            if sim_opts["max_no_pitstops"] > 2:
                print('INFO: Plotting of strategies with more than 2 stops is not possible!')


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # pars_file:                set the parameter file
    # min_no_pitstops:          set minimum number of pitstops (mostly 1)
    # max_no_pitstops:          set maximum number of pitstops
    # enforce_diff_compounds:   enforce that at least two different compounds must be used in the race
    # use_qp:                   activate quadratic optim. to determine the optimal inlaps -> requires linear model, is
    #                           fast, reduced plotting
    # fcy_phases:               either None or [[start race progress, stop race progress, phase type], [...], ...]
    #                           -> only considered in full factorial calculations, not in QP!
    #                           -> start and stop race progress must be in range [0.0, tot_no_laps] (e.g. if SC comes
    #                           at 30% of the first lap and leaves at the end of lap 2 it would be [[0.3, 2.0, 'SC']])
    #                           -> valid FCY phase types are 'SC' and 'VSC'
    # use_plot:                 set if plotting should be used or not (will be shown up to max. 2 stops)
    # use_print:                set if prints to console should be used or not (does not suppress hints/warnings)
    # use_print_result:         set if result should be printed to console or not

    sim_opts_ = {"pars_file": "pars_YasMarina_2017.ini",
                 "min_no_pitstops": 1,
                 "max_no_pitstops": 2,
                 "enforce_diff_compounds": True,
                 "use_qp": False,
                 "fcy_phases": None,
                 "use_plot": False,
                 "use_print": True,
                 "use_print_result": True}

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(sim_opts=sim_opts_)
