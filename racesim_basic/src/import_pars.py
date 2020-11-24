import os
import configparser
import json


def import_pars(use_print: bool, race_pars_file: str) -> dict:
    # get repo path
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # load race parameter file -----------------------------------------------------------------------------------------
    if use_print:
        print("INFO: Loading race parameters...")
    par_file_path = os.path.join(repo_path, "racesim_basic", "input", "parameters", race_pars_file)

    parser = configparser.ConfigParser()
    pars_in = {}

    if not parser.read(par_file_path):
        raise RuntimeError('Specified race parameter config file does not exist or is empty!')

    pars_in['driver_pars'] = json.loads(parser.get('DRIVER_PARS', 'driver_pars'))
    pars_in['track_pars'] = json.loads(parser.get('TRACK_PARS', 'track_pars'))
    pars_in['race_pars'] = json.loads(parser.get('RACE_PARS', 'race_pars'))

    # determine some additionally required variables
    pars_in['available_compounds'] = [key for key in pars_in['driver_pars']["tire_pars"].keys()
                                      if key in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'I', 'W']]

    if pars_in['driver_pars']["drivetype"] == "combustion" and pars_in['driver_pars']["b_fuel_perlap"] is None:
        # calculate approximate fuel consumption per lap
        pars_in['driver_pars']["b_fuel_perlap"] = (pars_in['driver_pars']["m_fuel_init"]
                                                   / pars_in['race_pars']["tot_no_laps"])

        if use_print:
            print("INFO: Fuel consumption was automatically determined to %.2fkg/lap!"
                  % pars_in['driver_pars']["b_fuel_perlap"])

    return pars_in
