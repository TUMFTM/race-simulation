from racesim.src.race import Race

"""
author:
Alexander Heilmeier

date:
23.10.2018

.. description::
The handle is required for multiprocess calculations such that every executor gets his own instances of the classes.
"""


def race_handle(pars_in: dict, use_prob_infl: bool, create_rand_events: bool, vse_paths: dict) -> Race:
    # create race object
    race = Race(race_pars=pars_in["race_pars"],
                driver_pars=pars_in["driver_pars"],
                car_pars=pars_in["car_pars"],
                tireset_pars=pars_in["tireset_pars"],
                track_pars=pars_in["track_pars"],
                vse_pars=pars_in["vse_pars"],
                vse_paths=vse_paths,
                use_prob_infl=use_prob_infl,
                create_rand_events=create_rand_events,
                monte_carlo_pars=pars_in["monte_carlo_pars"],
                event_pars=pars_in["event_pars"])

    # simulate race
    race.simulate_race()

    # remove VSE before returning because TF models cannot be serialized leading to an error in the case of
    # multiprocessing
    race.vse = None

    return race
