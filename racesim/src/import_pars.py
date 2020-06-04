import os
import configparser
import json


def import_pars(use_print: bool, race_pars_file: str, mcs_pars_file: str) -> dict:
    # get repo path
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # load race parameter file -----------------------------------------------------------------------------------------
    if use_print:
        print("INFO: Loading race parameters...")
    par_file_path = os.path.join(repo_path, "racesim", "input", "parameters", race_pars_file)

    parser = configparser.ConfigParser()
    pars_in = {}

    if not parser.read(par_file_path):
        raise ValueError('Specified race parameter config file does not exist or is empty!')

    pars_in["race_pars"] = json.loads(parser.get('RACE_PARS', 'race_pars'))
    pars_in["monte_carlo_pars"] = json.loads(parser.get('MONTE_CARLO_PARS', 'monte_carlo_pars'))
    pars_in["track_pars"] = json.loads(parser.get('TRACK_PARS', 'track_pars'))
    pars_in["car_pars"] = json.loads(parser.get('CAR_PARS', 'car_pars'))
    pars_in["tireset_pars"] = json.loads(parser.get('TIRESET_PARS', 'tireset_pars'))
    pars_in["driver_pars"] = json.loads(parser.get('DRIVER_PARS', 'driver_pars'))
    pars_in["event_pars"] = json.loads(parser.get('EVENT_PARS', 'event_pars'))

    # perform sorting of FCY phases and retirements --------------------------------------------------------------------

    # check that we got a list of lists for FCY phases (further checks are performed later on)
    if any(True if type(x) is not list else False for x in pars_in["event_pars"]["fcy_data"]["phases"]):
        raise TypeError("FCY phases must be a list of lists!")

    # sort FCY phases by start
    pars_in["event_pars"]["fcy_data"]["phases"].sort(key=lambda x: x[0])

    # check that we got a list of lists for retirements (further checks are performed later on)
    if any(True if type(x) is not list else False for x in pars_in["event_pars"]["retire_data"]["retirements"]):
        raise TypeError("Retirement data must be a list of lists!")

    # sort retirements by start
    pars_in["event_pars"]["retire_data"]["retirements"].sort(key=lambda x: x[1])

    # load MCS parameter file ------------------------------------------------------------------------------------------
    if use_print:
        print("INFO: Loading MCS parameters...")
    par_file_path = os.path.join(repo_path, "racesim", "input", "parameters", mcs_pars_file)

    if not parser.read(par_file_path):
        raise ValueError('Specified MCS parameter config file does not exist or is empty!')

    season_tmp = pars_in["race_pars"]["season"]  # get currently simulated season to load the correct parameters

    p_accident = json.loads(parser.get('SEASON_%i' % season_tmp, 'p_accident'))
    p_failure = json.loads(parser.get('SEASON_%i' % season_tmp, 'p_failure'))
    p_fcy_phases = json.loads(parser.get('ALL_SEASONS', 'p_fcy_phases'))
    t_pit_var_fisk_pars = json.loads(parser.get('ALL_SEASONS', 't_pit_var_fisk_pars'))
    t_lap_var_sigma = json.loads(parser.get('ALL_SEASONS', 't_lap_var_sigma'))
    t_startperf = json.loads(parser.get('ALL_SEASONS', 't_startperf'))

    # add MCS parameters to the correct driver and car parameters ------------------------------------------------------
    for initials in pars_in["driver_pars"]:
        name_tmp = pars_in["driver_pars"][initials]["name"]
        pars_in["driver_pars"][initials]["p_accident"] = p_accident[name_tmp]

        # use "unknown" driver if desired driver is not included in the start performance data
        if name_tmp in t_lap_var_sigma:
            pars_in["driver_pars"][initials]["t_lap_var_sigma"] = t_lap_var_sigma[name_tmp]
        else:
            pars_in["driver_pars"][initials]["t_lap_var_sigma"] = t_lap_var_sigma["unknown"]

        # use "unknown" driver if desired driver is not included in the start performance data
        if name_tmp in t_startperf:
            pars_in["driver_pars"][initials]["t_startperf"] = t_startperf[name_tmp]
        else:
            pars_in["driver_pars"][initials]["t_startperf"] = t_startperf["unknown"]

    for team in pars_in["car_pars"]:
        pars_in["car_pars"][team]["p_failure"] = p_failure[team]
        pars_in["car_pars"][team]["t_pit_var_fisk_pars"] = t_pit_var_fisk_pars[team]

    for param in p_fcy_phases:
        pars_in["monte_carlo_pars"][param] = p_fcy_phases[param]

    return pars_in
