def check_pars(sim_opts: dict, pars_in: dict, use_plot: bool) -> None:
    # check user input
    if pars_in['driver_pars']['tire_pars']['tire_deg_model'] != 'lin' and sim_opts["use_qp"]:
        raise RuntimeError('QP is only available for a linear tire degradation model!')

    if use_plot and sim_opts["use_qp"]:
        print('INFO: Plotting will be reduced since the derived data from the QP is much less than for full factorial!')

    if not 0 <= sim_opts["min_no_pitstops"] < sim_opts["max_no_pitstops"]:
        raise RuntimeError('Minimum number of pit stops must be less than maximum number of pit stops and greater than'
                           ' 0!')

    if sim_opts["min_no_pitstops"] == 0 and sim_opts["enforce_diff_compounds"]:
        print('WARNING: Different compounds cannot be enforced if number of pitstops is zero!')

    if sim_opts["use_qp"] and sim_opts["fcy_phases"]:
        print("WARNING: FCY phases cannot be considered when using the quadratic optimization, they will therefore be"
              " neglected!")
