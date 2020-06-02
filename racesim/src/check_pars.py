import multiprocessing


def check_pars(sim_opts: dict, pars_in: dict) -> None:
    if sim_opts["use_print"]:
        print("INFO: Checking user input...")

    if sim_opts["no_workers"] > multiprocessing.cpu_count():
        print("HINT: Number of workers is higher than number of available CPU cores, this may affect performance!")

    if sim_opts["no_bunches"] * sim_opts["no_races_per_bunch"] > 1 and not sim_opts["use_random"]:
        print("HINT: Simulating more than one race without randomness makes no sense!")

    if sim_opts["no_bunches"] * sim_opts["no_races_per_bunch"] > 1000 and sim_opts["no_workers"] == 1:
        print("HINT: Think about increasing the number of workers when simulating a big amount of races!")

    if sim_opts["no_bunches"] > 1 and sim_opts["no_races_per_bunch"] < 1000:
        raise ValueError("For the statistical analysis there should be at least 1,000 valid races per bunch!")

    if sim_opts["use_print"]:
        print("INFO: FCY phases for the race simulation were set as follows:",
              pars_in["event_pars"]["fcy_data"]["phases"])

    if pars_in["event_pars"]["fcy_data"]["phases"]:
        # check domain type
        if pars_in["event_pars"]["fcy_data"]["domain"] not in ['progress', 'time']:
            raise ValueError("Unknown FCY domain type!")

        # check FCY phases for valid values and type
        for cur_phase in pars_in["event_pars"]["fcy_data"]["phases"]:
            # check if phase information is complete
            if not len(cur_phase) == 5:
                raise ValueError("A FCY phase must contain 5 entries: [start, end, type, SC delay, SC duration]. The"
                                 " latter 2 must be set null except it is an SC phase given in the time domain!")

            # check start and end race progress of FCY phases
            if not cur_phase[0] < cur_phase[1] \
                    or cur_phase[0] < 0.0 \
                    or (pars_in["event_pars"]["fcy_data"]["domain"] == 'progress'
                        and cur_phase[1] > pars_in["race_pars"]["tot_no_laps"]):
                raise ValueError("Start and end of a FCY phase is unreasonable!")

            # check type of FCY phases
            if cur_phase[2] not in ["SC", "VSC"]:
                raise ValueError("Unknown FCY phase type!")

            # check SC delay
            if pars_in["event_pars"]["fcy_data"]["domain"] == 'progress' and cur_phase[3] is not None:
                raise ValueError("SC delay information must only be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and cur_phase[3] is None:
                raise ValueError("SC delay information must be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and not 0.0 <= cur_phase[3] <= 100.0:
                raise ValueError("SC delay seems not to have a valid length!")

            # check SC duration
            if pars_in["event_pars"]["fcy_data"]["domain"] == 'progress' and cur_phase[4] is not None:
                raise ValueError("SC duration information must only be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and cur_phase[4] is None:
                raise ValueError("SC duration information must be given if domain is time!")
            elif pars_in["event_pars"]["fcy_data"]["domain"] == 'time' and not 2 <= cur_phase[4] <= 10:
                raise ValueError("SC duration seems not to have a valid length!")

    if sim_opts["use_print"]:
        print("INFO: Retirements for the race simulation were set as follows:",
              pars_in["event_pars"]["retire_data"]["retirements"])

    if pars_in["event_pars"]["retire_data"]["retirements"]:
        # check domain type
        if pars_in["event_pars"]["retire_data"]["domain"] not in ['progress', 'time']:
            raise ValueError("Unknown retirements domain type!")

        # check retirement data for valid initials and start points
        for cur_retirement in pars_in["event_pars"]["retire_data"]["retirements"]:
            if cur_retirement[0] not in pars_in["race_pars"]["participants"]\
                    or cur_retirement[1] < 0.0 \
                    or (pars_in["event_pars"]["retire_data"]["domain"] == 'progress'
                        and cur_retirement[1] > pars_in["race_pars"]["tot_no_laps"]):
                raise ValueError("A retiring driver does not participate in the race or the start of his retirement"
                                 " is unreasonable!")
