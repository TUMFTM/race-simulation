import racesim
import helper_funcs
from racesim.src.race_handle import race_handle
from concurrent import futures  # required for parallel computing
import numpy as np
import time
import os
import pkg_resources
import pickle

"""
author:
Alexander Heilmeier

date:
12.07.2018

.. description::
This file includes the main function as well as required plot functions. The script part required to run
the simulation is located at the bottom. Have a look there to insert the required user parameters.
"""


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def main(sim_opts: dict, race_pars_file: str, mcs_pars_file: str) -> list:

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

    # create output folders (if not existing)
    output_path = os.path.join(repo_path, "racesim", "output")

    results_path = os.path.join(output_path, "results")
    os.makedirs(results_path, exist_ok=True)

    invalid_dumps_path = os.path.join(output_path, "invalid_dumps")
    os.makedirs(invalid_dumps_path, exist_ok=True)

    testobjects_path = os.path.join(output_path, "testobjects")
    os.makedirs(testobjects_path, exist_ok=True)

    # load parameters
    pars_in = racesim.src.import_pars.import_pars(use_print=sim_opts["use_print"],
                                                  race_pars_file=race_pars_file,
                                                  mcs_pars_file=mcs_pars_file)

    # check parameters
    racesim.src.check_pars.check_pars(sim_opts=sim_opts, pars_in=pars_in)

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create list containing the simulated race objects with valid results
    result_objects = []

    # calculate total number of (valid) races to simulate
    tot_no_races = sim_opts["no_bunches"] * sim_opts["no_races_per_bunch"]

    # save start time for runtime calculation
    if sim_opts["use_print"]:
        print("INFO: Starting simulations...")
    t_start = time.perf_counter()

    # iteration variables
    no_races_left = tot_no_races    # counter for the number of races left for simulation
    ctr_invalid = 0                 # counter for the number of simulated races marked as invalid

    # SINGLE PROCESS ---------------------------------------------------------------------------------------------------
    if sim_opts["no_workers"] == 1:

        while no_races_left > 0:
            # simulate race
            tmp_race_handle = race_handle(pars_in=pars_in,
                                          use_random=sim_opts['use_random'])
            no_races_left -= 1

            # CASE 1: result is valid
            if tmp_race_handle.result_status == 0:
                # save race object for later evaluation (single race) or simple race results (MCS)
                if tot_no_races > 1:
                    result_objects.append(tmp_race_handle.get_race_results())
                else:
                    result_objects.append(tmp_race_handle)

            # CASE 2: result is invalid
            else:
                # increase no_races_left
                ctr_invalid += 1
                no_races_left += 1

                # pickle race object for further analysis
                if tmp_race_handle.result_status >= 10 or tmp_race_handle.result_status == -1:
                    cur_time_str = time.strftime("%Y%m%d_%H%M%S")
                    tmp_file_path = os.path.join(invalid_dumps_path, cur_time_str + "_invalid_race_%i_%i.pkl"
                                                 % (ctr_invalid, tmp_race_handle.result_status))

                    with open(tmp_file_path, 'wb') as fh:
                        pickle.dump(tmp_race_handle, fh)

            # print progressbar
            if sim_opts["use_print"]:
                helper_funcs.src.progressbar.progressbar(i=tot_no_races - no_races_left,
                                                         i_total=tot_no_races,
                                                         prefix="INFO: Simulation progress:")

    # MULTIPLE PROCESSES -----------------------------------------------------------------------------------------------
    else:
        # set maximum number of jobs in the waiting queue at the same time -> limits RAM usage
        max_no_concurrent_jobs = 200

        # create executor instance (pool of processes available for parallel calculations)
        with futures.ProcessPoolExecutor(max_workers=sim_opts["no_workers"]) as executor:

            while no_races_left > 0:
                # reset job queue (list containing current simulation jobs)
                job_queue = []

                # submit simulations to the waiting queue of the executor instance as long as we have races left for
                # simulation and the job queue is not full
                while len(job_queue) <= max_no_concurrent_jobs and no_races_left > 0:
                    job_queue.append(executor.submit(race_handle,
                                                     pars_in,
                                                     sim_opts['use_random']))
                    no_races_left -= 1

                # collect results as soon as they are available
                for job_handle in futures.as_completed(job_queue):
                    tmp_race_handle = job_handle.result()

                    # CASE 1: result is valid
                    if tmp_race_handle.result_status == 0:
                        # save race object for later evaluation (single race) or simple race results (MCS)
                        if tot_no_races > 1:
                            result_objects.append(tmp_race_handle.get_race_results())
                        else:
                            result_objects.append(tmp_race_handle)

                    # CASE 2: result is invalid
                    else:
                        # increase no_races_left
                        ctr_invalid += 1
                        no_races_left += 1

                        # pickle race object for further analysis
                        if tmp_race_handle.result_status >= 10 or tmp_race_handle.result_status == -1:
                            cur_time_str = time.strftime("%Y%m%d_%H%M%S")
                            tmp_file_path = os.path.join(invalid_dumps_path, cur_time_str + "_invalid_race_%i_%i.pkl"
                                                         % (ctr_invalid, tmp_race_handle.result_status))

                            with open(tmp_file_path, 'wb') as fh:
                                pickle.dump(tmp_race_handle, fh)

                # print progressbar
                if sim_opts["use_print"]:
                    helper_funcs.src.progressbar.progressbar(i=tot_no_races - no_races_left,
                                                             i_total=tot_no_races,
                                                             prefix="INFO: Simulation progress:")

    # print number of invalid races
    if sim_opts["use_print"]:
        print("INFO: There were %i invalid races!" % ctr_invalid)

    # print runtime into console window
    if sim_opts["use_print"]:
        runtime = time.perf_counter() - t_start
        print("INFO: Simulation runtime: {:.3f}s ({:.3f}ms per race)".format(runtime, runtime / tot_no_races * 1000))

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if sim_opts["use_print"]:
        print("INFO: Postprocessing in progress...")

    # SINGLE RACE ------------------------------------------------------------------------------------------------------
    if tot_no_races == 1:
        result_objects[0].check_valid_result()

        if sim_opts["use_print_result"]:
            result_objects[0].print_result()
            # result_objects[0].print_details()

        if sim_opts["use_plot"]:
            # result_objects[0].plot_laptimes()
            # result_objects[0].plot_positions()
            # result_objects[0].plot_racetime_diffto_refdriver(1)
            # result_objects[0].plot_raceprogress_over_racetime()

            laps_simulated = result_objects[0].cur_lap
            t_race_winner = np.sort(result_objects[0].racetimes[laps_simulated, :])[0]
            result_objects[0].plot_racetime_diffto_reflaptime(ref_laptime=t_race_winner / laps_simulated)

        # evaluation
        # result_objects[0].print_race_standings(racetime=2520.2)

        # save lap times, race times and positions to csv files
        result_objects[0].export_results_as_csv(results_path=results_path)

        # pickle race object for possible CI testing
        result_objects_file_path = os.path.join(testobjects_path, "testobj_racesim_%s_%i.pkl"
                                                % (pars_in["track_pars"]["name"], pars_in["race_pars"]["season"]))
        with open(result_objects_file_path, 'wb') as fh:
            pickle.dump(result_objects[0], fh)

    # MULTIPLE RACES ---------------------------------------------------------------------------------------------------
    else:
        # plot histograms
        racesim.src.mcs_analysis.mcs_analysis(result_objs=result_objects,
                                              no_bunches=sim_opts["no_bunches"],
                                              use_print_result=sim_opts["use_print_result"],
                                              use_plot=sim_opts["use_plot"])

    if sim_opts["use_print"]:
        print("INFO: Simulation finished successfully!")

    return result_objects  # return required in case of CI testing


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set race parameter file names
    race_pars_file_ = 'pars_YasMarina_2017.ini'
    mcs_pars_file_ = 'pars_mcs.ini'

    # set simulation options
    # use_random:           use randomness within the race simulation -> random events (FCY phases and retirements) will
    #                       be created automatically if the according entries in the parameter file contain empty lists
    # no_bunches:           number of bunches of race simulations (to be able to determine error plots, should be set 1
    #                       in most cases)
    # no_races_per_bunch:   number of (valid) races to simulate per bunch
    # no_workers:           defines number of workers for multiprocess calculations, 1 for single process, >1 for
    #                       multi-process (you can use print(multiprocessing.cpu_count()) to determine the max. number)
    # use_print:            set if prints to console should be used or not (does not suppress hints/warnings)
    # use_print_result:     set if result should be printed to console or not
    # use_plot:             set if plotting should be used or not
    sim_opts_ = {"use_random": False,
                 "no_bunches": 1,
                 "no_races_per_bunch": 1,
                 "no_workers": 1,
                 "use_print": True,
                 "use_print_result": True,
                 "use_plot": False}

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(sim_opts=sim_opts_,
         race_pars_file=race_pars_file_,
         mcs_pars_file=mcs_pars_file_)
