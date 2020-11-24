import multiprocessing


def check_pars(sim_opts: dict, pars_in: dict) -> None:
    if sim_opts["use_print"]:
        print("INFO: Checking user input...")

    if sim_opts["no_workers"] > multiprocessing.cpu_count():
        print("HINT: Number of workers is higher than number of available CPU cores, this may affect performance!")

    if sim_opts["no_sim_runs"] > 1 and not sim_opts["use_prob_infl"] and not sim_opts["create_rand_events"]:
        print("HINT: Simulating more than one race without randomness makes no sense!")

    if sim_opts["no_sim_runs"] > 1000 and sim_opts["no_workers"] == 1:
        print("HINT: Think about increasing the number of workers when simulating a big amount of races!")

    p_grids = [pars_in["driver_pars"][initials]["p_grid"] for initials in pars_in["driver_pars"]]
    if not len(set(p_grids)) == len(p_grids):
        raise RuntimeError("Grid positions are not unique!")

    if sim_opts["use_print"]:
        print("INFO: FCY phases for the race simulation were set as follows:",
              pars_in["event_pars"]["fcy_data"]["phases"])

    if pars_in["event_pars"]["fcy_data"]["phases"]:
        # check domain type
        if pars_in["event_pars"]["fcy_data"]["domain"] not in ['progress', 'time']:
            raise RuntimeError("Unknown FCY domain type!")

        # check FCY phases for valid values and type
        for cur_phase in pars_in["event_pars"]["fcy_data"]["phases"]:
            # check if phase information is complete
            if not len(cur_phase) == 5:
                raise RuntimeError("A FCY phase must contain 5 entries: [start, end, type, SC delay, SC duration]. The"
                                   " latter 2 must be set null except it is an SC phase given in the time domain!")

            # check start and end race progress of FCY phases
            if not cur_phase[0] < cur_phase[1] \
                    or cur_phase[0] < 0.0 \
                    or (pars_in["event_pars"]["fcy_data"]["domain"] == 'progress'
                        and cur_phase[1] > pars_in["race_pars"]["tot_no_laps"]):
                raise RuntimeError("Start and end of a FCY phase is unreasonable!")

            # check type of FCY phases
            if cur_phase[2] not in ["SC", "VSC"]:
                raise RuntimeError("Unknown FCY phase type!")

            # check SC delay
            if pars_in["event_pars"]["fcy_data"]["domain"] == 'progress' and cur_phase[3] is not None:
                raise RuntimeError("SC delay information must only be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and cur_phase[3] is None:
                raise RuntimeError("SC delay information must be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and not 0.0 <= cur_phase[3] <= 100.0:
                raise RuntimeError("SC delay seems not to have a valid length!")

            # check SC duration
            if pars_in["event_pars"]["fcy_data"]["domain"] == 'progress' and cur_phase[4] is not None:
                raise RuntimeError("SC duration information must only be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and cur_phase[4] is None:
                raise RuntimeError("SC duration information must be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and not 2 <= cur_phase[4] <= 10:
                raise RuntimeError("SC duration seems not to have a valid length!")

    if sim_opts["use_print"]:
        print("INFO: Retirements for the race simulation were set as follows:",
              pars_in["event_pars"]["retire_data"]["retirements"])

    if pars_in["event_pars"]["retire_data"]["retirements"]:
        # check domain type
        if pars_in["event_pars"]["retire_data"]["domain"] not in ['progress', 'time']:
            raise RuntimeError("Unknown retirements domain type!")

        # check retirement data for valid initials and start points
        for cur_retirement in pars_in["event_pars"]["retire_data"]["retirements"]:
            if cur_retirement[0] not in pars_in["race_pars"]["participants"]\
                    or cur_retirement[1] < 0.0 \
                    or (pars_in["event_pars"]["retire_data"]["domain"] == 'progress'
                        and cur_retirement[1] > pars_in["race_pars"]["tot_no_laps"]):
                raise RuntimeError("A retiring driver does not participate in the race or the start of his retirement"
                                   " is unreasonable!")

    if sim_opts["use_print"] and sim_opts["use_vse"]:
        print("INFO: Using VSE (virtual strategy engineer) to take tire change decisions!")

    if sim_opts["use_vse"]:
        # assure that available and parameterized compounds are in rising order
        pars_in["vse_pars"]["available_compounds"].sort()
        pars_in["vse_pars"]["param_dry_compounds"].sort()

        # assure that there are 2 or 3 dry compounds available in the race
        no_dry_compounds = sum(1 if x in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"] else 0
                               for x in pars_in["vse_pars"]["available_compounds"])

        if not 2 <= no_dry_compounds <= 3:
            raise RuntimeError("VSE is trained for 2 to 3 different dry compounds but %i were given!"
                               % no_dry_compounds)

        # assure that every driver has at least 2 dry compounds available (i.e. parameterized)
        for initials in pars_in["driver_pars"]:
            if not len([x for x in pars_in["tireset_pars"][initials]
                        if x in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]]) >= 2:
                raise RuntimeError("There must be at least two different tire compounds available for every driver!"
                                   " This is not fulfilled for %s!" % initials)

        # assure that chosen VSE types are supported
        for key in pars_in["vse_pars"]["vse_type"]:
            if pars_in["vse_pars"]["vse_type"][key] \
                    not in ["supervised", "reinforcement", "basestrategy", "realstrategy", "reinforcement_training"]:
                raise RuntimeError("Unknown VSE type %s!" % pars_in["vse_pars"]["vse_type"][key])
