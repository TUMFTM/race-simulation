import racesim.src.import_pars


def import_ext_params(use_print: bool, race_pars_file: str, driver_initials: str) -> dict:
    # load parameters --------------------------------------------------------------------------------------------------
    pars_in = racesim.src.import_pars.import_pars(use_print=use_print,
                                                  use_vse=False,
                                                  race_pars_file=race_pars_file,
                                                  mcs_pars_file='pars_mcs.ini')[0]

    # convert parameter format -----------------------------------------------------------------------------------------
    pars_basic = {}

    pars_basic['track_pars'] = {"t_pitdrive_inlap": pars_in['track_pars']['t_pitdrive_inlap'],
                                "t_pitdrive_outlap": pars_in['track_pars']['t_pitdrive_outlap'],
                                "t_pitdrive_inlap_fcy": pars_in['track_pars']['t_pitdrive_inlap_fcy'],
                                "t_pitdrive_outlap_fcy": pars_in['track_pars']['t_pitdrive_outlap_fcy'],
                                "t_pitdrive_inlap_sc": pars_in['track_pars']['t_pitdrive_inlap_sc'],
                                "t_pitdrive_outlap_sc": pars_in['track_pars']['t_pitdrive_outlap_sc'],
                                "pits_aft_finishline": pars_in['track_pars']['pits_aft_finishline'],
                                "t_lap_fcy": ((pars_in['track_pars']['t_q'] + pars_in['track_pars']['t_gap_racepace'])
                                              * pars_in['track_pars']['mult_t_lap_fcy']),
                                "t_lap_sc": ((pars_in['track_pars']['t_q'] + pars_in['track_pars']['t_gap_racepace'])
                                             * pars_in['track_pars']['mult_t_lap_sc']),
                                "t_lap_sens_mass": pars_in['track_pars']['t_lap_sens_mass'],
                                "t_loss_pergridpos": pars_in['track_pars']['t_loss_pergridpos'],
                                "t_loss_firstlap": pars_in['track_pars']['t_loss_firstlap']}

    pars_basic['race_pars'] = {'tot_no_laps': pars_in['race_pars']['tot_no_laps']}

    # raise error if there are not at least two different dry compounds parameterized (mostly due to wet races)
    if len(pars_in['vse_pars']['param_dry_compounds']) >= 2:
        pars_basic['available_compounds'] = pars_in['vse_pars']['param_dry_compounds']
    else:
        raise RuntimeError("There should be at least two different dry compounds available in the parameter file!")

    # determine driver specific parameters
    team_tmp = pars_in['driver_pars'][driver_initials]['team']

    pars_basic['driver_pars'] = {"t_base": (pars_in['track_pars']['t_q'] + pars_in['track_pars']['t_gap_racepace']
                                            + pars_in['driver_pars'][driver_initials]['t_driver']
                                            + pars_in['car_pars'][team_tmp]['t_car']),
                                 "p_grid": pars_in['driver_pars'][driver_initials]['p_grid'],
                                 "tire_pars": pars_in['tireset_pars'][driver_initials],
                                 "drivetype": pars_in['car_pars'][team_tmp]['drivetype'],
                                 "t_pit_tirechange": (pars_in['track_pars']['t_pit_tirechange_min']
                                                      + pars_in['car_pars'][team_tmp]['t_pit_tirechange_add']),
                                 "m_fuel_init": pars_in['car_pars'][team_tmp]['m_fuel'],
                                 "b_fuel_perlap": pars_in['car_pars'][team_tmp]['b_fuel_perlap'],
                                 "t_pit_refuel_perkg": pars_in['car_pars'][team_tmp]['t_pit_refuel_perkg'],
                                 "t_pit_charge_perkwh": pars_in['car_pars'][team_tmp]['t_pit_charge_perkwh']}

    return pars_basic
