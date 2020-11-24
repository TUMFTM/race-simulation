import numpy as np
import helper_funcs.src.calc_tire_degradation
import math


def calc_racetimes_basic(t_base: float,
                         tot_no_laps: int,
                         t_lap_sens_mass: float,
                         t_pitdrive_inlap: float,
                         t_pitdrive_outlap: float,
                         t_pit_tirechange: float,
                         pits_aft_finishline: bool,
                         tire_pars: dict,
                         p_grid: int,
                         t_loss_pergridpos: float,
                         t_loss_firstlap: float,
                         strategy: list,
                         drivetype: str,
                         m_fuel_init: float,
                         b_fuel_perlap: float,
                         t_pit_refuel_perkg: float,
                         t_pit_charge_perkwh: float,
                         t_pitdrive_inlap_fcy: float = None,
                         t_pitdrive_outlap_fcy: float = None,
                         t_pitdrive_inlap_sc: float = None,
                         t_pitdrive_outlap_sc: float = None,
                         fcy_phases: list = None,
                         t_lap_sc: float = None,
                         t_lap_fcy: float = None) -> tuple:

    """
    author:
    Alexander Heilmeier

    date:
    01.10.2019

    .. description::
    This basic simulation calculates the expected race times after every lap of a race under the assumption of a free
    track on the basis of the given parameters. Furthermore, it returns the estimated start and end race times for
    every FCY phase on the basis of their inserted start and stop race progress. The function is capable of considering
    FCY phases (VSC and SC). Therefore, it is assumed that the driver is directly affected by a FCY phase as soon as it
    appears. The slower FCY lap time is instantly applied, the safety car lap time is considered starting with the next
    lap. Driving through the pit under an active FCY phase leads to smaller time losses compared to normal race speed.
    Therefore, the modified time losses must be supplied if FCY phases are not None.

    .. inputs::
    :param t_base:                  [s] base lap time (= t_q + t_gap,racepace + t_car + t_driver)
    :type t_base:                   float
    :param tot_no_laps:             number of laps in current race
    :type tot_no_laps:              int
    :param t_lap_sens_mass:         [s/kg] lap time sensitivity against fuel mass
    :type t_lap_sens_mass:          float
    :param t_pitdrive_inlap:        [s] lap time loss in current lap when entering the pit
    :type t_pitdrive_inlap:         float
    :param t_pitdrive_outlap:       [s] lap time loss driving through the pit during the outlap
    :type t_pitdrive_outlap:        float
    :param t_pit_tirechange:        [s] standstill time to change tires during pit stop
    :type t_pit_tirechange:         float
    :param tire_pars:               tire model parameters for every compound -> see param file
    :type tire_pars:                dict
    :param pits_aft_finishline:     boolean indicating if pits are located before or after the finish line
    :type pits_aft_finishline:      bool
    :param p_grid:                  [-] grid position at race start
    :type p_grid:                   int
    :param t_loss_pergridpos:       [s/pos] lap time loss between two grid positions
    :type t_loss_pergridpos:        float
    :param t_loss_firstlap:         [s] lap time loss due to start from standstill
    :type t_loss_firstlap:          float
    :param strategy:                race strategy: [[inlap, compound, age, refueling (kg or kWh)], [...], ...], strategy
                                    entry 0 must be [0, start compound, start tire age, 0.0]
    :type strategy:                 list
    :param drivetype:               either combustion or electric
    :type drivetype:                str
    :param m_fuel_init:             [kg] initial fuel mass -> None for electric
    :type m_fuel_init:              float
    :param b_fuel_perlap:           [kg/lap] fuel consumption per lap -> None for electric
    :type b_fuel_perlap:            float
    :param t_pit_refuel_perkg:      [s/kg] time per fuel added in pit
    :type t_pit_refuel_perkg:       float
    :param t_pit_charge_perkwh:     [s/kWh] time per energy added in pit
    :type t_pit_charge_perkwh:      float
    :param t_pitdrive_inlap_fcy:    [s] lap time loss in current lap when entering the pit (under FCY condition)
                                    (optional)
    :type t_pitdrive_inlap_fcy:     float
    :param t_pitdrive_outlap_fcy:   [s] lap time loss driving through the pit during the outlap (under FCY condition)
                                    (optional)
    :type t_pitdrive_outlap_fcy:    float
    :param t_pitdrive_inlap_sc:     [s] lap time loss in current lap when entering the pit (under SC condition)
                                    (optional)
    :type t_pitdrive_inlap_sc:      float
    :param t_pitdrive_outlap_sc:    [s] lap time loss driving through the pit during the outlap (under SC condition)
                                    (optional)
    :type t_pitdrive_outlap_sc:     float
    :param fcy_phases:              [[start race progress, stop race progress, type, (None, None)], ...] list of FCY
                                    phases (optional). Start and stop race progress must be in the range
                                    [0.0, tot_no_laps] (e.g. if SC comes at 30% of the first lap and leaves at the end
                                    of lap 2 it would be [[0.3, 2.0, 'SC']]). Valid FCY phase types are 'SC' and 'VSC'.
                                    The both None values can be inserted but are not considered within this function.
    :type fcy_phases:               list
    :param t_lap_sc:                [s] lap time behind SC (safety car) (required if fcy_phases is not None)
    :type t_lap_sc:                 float
    :param t_lap_fcy:               [s] lap time during FCY (full course yellow) phase (required if fcy_phases is not
                                    None)
    :type t_lap_fcy:                float

    .. outputs::
    :return t_race_laps:            [s] cumulated lap times (i.e. race times) after every lap
    :rtype t_race_laps:             np.ndarray
    :return fcy_phases_conv:        [[start race time, stop race time, type, SC delay, SC duration], ...] list of
                                    converted FCY phases (optional) containing race times instead of race progress. SC
                                    delay and duration are only set in case of an SC phase, otherwise None.
                                    fcy_phases_conv is None if input fcy_phases was None.
    :rtype fcy_phases_conv:         list
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check strategy input
    if len(strategy) == 0:
        raise RuntimeError('Start compound information must be provided!')
    elif len(strategy) == 1:
        print('WARNING: There is no pitstop given in the strategy data!')

    if not all([len(x) == 4 for x in strategy]):
        raise RuntimeError('Inserted strategy data does not contain [inlap, compound, age, refueling] for all pit'
                           ' stops!')

    # check if inlaps within strategy input appear in a rising order
    if not all([x[0] < y[0] for x, y in zip(strategy, strategy[1:])]):
        raise RuntimeError('The given inlaps are not sorted in a rising order!')

    # check drivetype and consumption
    if drivetype == 'combustion':
        if m_fuel_init is None or b_fuel_perlap is None:
            raise RuntimeError('Parameters m_fuel_init and b_fuel_perlap are required for a combustion car!')
    elif drivetype == 'electric':
        # electric consumption not required since the car does not lose any mass
        pass
    else:
        raise RuntimeError('Unknown drivetype!')

    # check possible refueling/recharging during pitstops
    if any(x[3] != 0.0 for x in strategy):
        if drivetype == 'combustion' and t_pit_refuel_perkg is None:
            raise RuntimeError('Refueling is set but t_pit_refuel_perkg is not set!')
        elif drivetype == 'electric' and t_pit_charge_perkwh is not None:
            raise RuntimeError('Recharging is set but t_pit_charge_perkwh is not set!')

    # check FCY phases
    if fcy_phases is not None and (t_lap_fcy is None or t_lap_sc is None):
        print("WARNING: t_lap_fcy and t_lap_sc are required if fcy_phases is not None! Using 140% and 160% of the"
              " base lap time instead!")
        t_lap_fcy = t_base * 1.4
        t_lap_sc = t_base * 1.6

    if fcy_phases is not None and any(False if x[2] in ['SC', 'VSC'] else True for x in fcy_phases):
        raise RuntimeError("Unknown FCY phase type!")

    # check if FCY phase list is sorted by start race progress (requirement for proper calculation of start and stop
    # race times later)
    if fcy_phases is not None and not all([x[0] < y[0] for x, y in zip(fcy_phases, fcy_phases[1:])]):
        raise RuntimeError('The given FCY phases are not sorted in a rising order!')

    # check if pit time losses are supplied in case of FCY phases
    if fcy_phases is not None \
            and (t_pitdrive_inlap_fcy is None or t_pitdrive_outlap_fcy is None
                 or t_pitdrive_inlap_sc is None or t_pitdrive_outlap_sc is None):
        raise RuntimeError("t_pitdrive_inlap_fcy/sc and t_pitdrive_outlap_fcy/sc must all be supplied if there are FCY"
                           " phases to consider!")

    # assure FCY phases end within the race
    if fcy_phases is not None:
        for idx_phase in range(len(fcy_phases)):
            if fcy_phases[idx_phase][1] > float(tot_no_laps):
                print("WARNING: Inserted FCY phase ends after the last lap of the race, reducing it to end with the"
                      " final lap!")
                fcy_phases[idx_phase][1] = float(tot_no_laps)

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER BASE LAP TIME, FUEL MASS LOSS AND RACE START ------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set base lap time
    t_laps = np.ones(tot_no_laps) * t_base

    # add fuel mass time loss for every lap (considered with fuel mass at start of respective lap)
    if drivetype == 'combustion':
        t_laps += (m_fuel_init - b_fuel_perlap * np.arange(0, tot_no_laps)) * t_lap_sens_mass

    # add race start losses
    t_laps[0] += t_loss_firstlap + (p_grid - 1) * t_loss_pergridpos

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER TIRE DEGRADATION ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # loop through all the pit stops
    for idx in range(len(strategy)):
        cur_inlap = strategy[idx][0]

        # get current stint length
        if idx + 1 < len(strategy):
            len_cur_stint = strategy[idx + 1][0] - cur_inlap
        else:
            len_cur_stint = tot_no_laps - cur_inlap

        # get compound until current pitstop
        comp_cur_stint = strategy[idx][1]

        # get tire age at the beginning of this stint
        age_cur_stint = strategy[idx][2]

        # add tire losses (degradation considered on basis of the tire age at the start of a lap)
        t_laps[cur_inlap:cur_inlap + len_cur_stint] += helper_funcs.src.calc_tire_degradation.\
            calc_tire_degradation(tire_age_start=age_cur_stint,
                                  stint_length=len_cur_stint,
                                  compound=comp_cur_stint,
                                  tire_pars=tire_pars)

        # consider cold tires in the first lap of a stint (if inlap is not the last lap of the race)
        if cur_inlap < tot_no_laps:
            t_laps[cur_inlap] += tire_pars["t_add_coldtires"]

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER PIT STOPS -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    THIS LIST IS USED FOR THE CALCULATION OF THE CORRECT LAP TIMES!
    When considering pit stops we have to check if they are performed during an active FCY phase. If this is
    the case we have to take modified pit stop driving time losses. Since these time losses are fixed, i.e. independent
    of when a FCY phase starts during a lap (as long as it is active for the entire pit stop driving) they should not
    be affected any further afterwards. Therefore, an extra array t_laps_pit is used to store them. In contrast, the
    normal lap times are modified by FCY phases according to the affected lap fractions, e.g. if a FCY starts at 30% of
    a lap.
    """

    # create array which is not affected by FCY phases afterwards
    t_laps_pit = np.zeros(tot_no_laps)

    """
    THIS LIST IS USED FOR THE CALCULATION OF THE CORRECT START AND END RACE TIMES OF FCY PHASES!
    To calculate the FCY start and end race times as exactly as possible we have to consider that pit stop time
    losses might have appeared before the FCY phase starts or ends. Therefore, the list t_pit_before_fcy_start_end
    stores these losses so that they can be considered later. We consider losses for this list only if the according
    event (e.g. inlap or outlap driving time losses) is completed before the start of the FCY phase respectively starts
    after the end of the FCY phase.
    For the start of a FCY phase we consider the outlap time losses if the FCY phase starts within the lap after the
    pit exit is passed (i.e. if the FCY phase starts at > lap_fraction_pit_outlap). Inlap time losses are not stored
    since the inlap driving is completed on the start finish line, i.e. it does not count for a FCY phase starting
    within this lap.
    For the end of a FCY phase we consider the outlap time losses if the FCY phase ends within the lap after the pit
    exit is passed (i.e. if the FCY phase ends at > lap_fraction_pit_outlap of the lap). Furthermore, we consider the
    inlap time losses if the FCY phase ends exactly at the end of the lap.
    """

    # create array to store the pit time losses that already appeared when a FCY starts or ends
    if fcy_phases is not None:
        t_pit_before_fcy_start_end = [[0.0, 0.0]] * len(fcy_phases)
    else:
        t_pit_before_fcy_start_end = None

    # estimate lap fraction of the pit entry and exit assuming the total pit length is about 6% of a normal lap length
    if pits_aft_finishline:
        lap_fraction_pit_inlap = 0.01
        lap_fraction_pit_outlap = 0.05
    else:
        lap_fraction_pit_inlap = 0.05
        lap_fraction_pit_outlap = 0.01

    # loop through all the pit stops -----------------------------------------------------------------------------------
    for idx in range(len(strategy)):
        cur_inlap = strategy[idx][0]

        # continue if this is the start stint of the strategy list (i.e. no real pit stop)
        if cur_inlap == 0:
            continue

        # pit losses (inlap) -------------------------------------------------------------------------------------------
        t_pit_inlap = 0.0

        # consider standstill time
        if not pits_aft_finishline:
            t_pit_inlap += __perform_pitstop_standstill(t_pit_tirechange=t_pit_tirechange,
                                                        drivetype=drivetype,
                                                        cur_stop=strategy[idx],
                                                        t_pit_refuel_perkg=t_pit_refuel_perkg,
                                                        t_pit_charge_perkwh=t_pit_charge_perkwh)

        # pit driving (inlap)
        if fcy_phases is not None:
            cur_phase = next((x for x in fcy_phases
                              if x[0] <= cur_inlap - lap_fraction_pit_inlap and cur_inlap <= x[1]), None)
        else:
            cur_phase = None

        if cur_phase is None:
            t_pit_inlap += t_pitdrive_inlap
        elif cur_phase[2] == 'SC':
            if cur_phase[0] < cur_inlap - 1.0:
                # SC started already before the inlap -> driver ran up to the SC already
                t_pit_inlap += t_pitdrive_inlap_sc
            else:
                # if SC phase started during the inlap no driver ran up to the SC already -> time loss is equal to
                # entering the pit during a FCY phase
                t_pit_inlap += t_pitdrive_inlap_fcy
        elif cur_phase[2] == 'VSC':
            t_pit_inlap += t_pitdrive_inlap_fcy
        else:
            raise RuntimeError("Unknown FCY phase type!")

        # add pit stop loss to t_laps_pit
        t_laps_pit[cur_inlap - 1] += t_pit_inlap

        # pit losses (outlap) ------------------------------------------------------------------------------------------
        # continue if inlap was the last lap of the race and therefore outlap is not driven anymore
        if cur_inlap >= tot_no_laps:
            continue

        t_pit_outlap = 0.0

        # consider standstill time
        if pits_aft_finishline:
            t_pit_outlap += __perform_pitstop_standstill(t_pit_tirechange=t_pit_tirechange,
                                                         drivetype=drivetype,
                                                         cur_stop=strategy[idx],
                                                         t_pit_refuel_perkg=t_pit_refuel_perkg,
                                                         t_pit_charge_perkwh=t_pit_charge_perkwh)

        # pit driving (outlap)
        if fcy_phases is not None:
            cur_phase = next((x for x in fcy_phases
                              if x[0] <= cur_inlap and cur_inlap + lap_fraction_pit_outlap <= x[1]), None)
        else:
            cur_phase = None

        if cur_phase is None:
            t_pit_outlap += t_pitdrive_outlap
        elif cur_phase[2] == 'SC':
            if cur_phase[0] < cur_inlap - 1.0:
                # SC started already before the inlap -> driver ran up to the SC already
                t_pit_outlap += t_pitdrive_outlap_sc
            else:
                # if SC phase started during the inlap no driver ran up to the SC already -> time loss is equal to
                # exiting the pit during a FCY phase
                t_pit_outlap += t_pitdrive_outlap_fcy
        elif cur_phase[2] == 'VSC':
            t_pit_outlap += t_pitdrive_outlap_fcy
        else:
            raise RuntimeError("Unknown FCY phase type!")

        # add pit stop loss to t_laps_pit
        t_laps_pit[cur_inlap] += t_pit_outlap

        # fill t_pit_before_fcy_start_end ------------------------------------------------------------------------------
        # if a FCY phase starts or ends after the driver finished a pit stop outlap or inlap we have to store the time
        # losses separately (see more detailed description in the block comment above)
        if fcy_phases is not None:
            for idx_fcy_phase, cur_phase in enumerate(fcy_phases):
                # check if current FCY phase starts within the lap after completed current pit stop (outlap)
                if cur_inlap + lap_fraction_pit_outlap < cur_phase[0] < cur_inlap + 1.0:
                    t_pit_before_fcy_start_end[idx_fcy_phase][0] += t_pit_outlap

                # check if current FCY phase ends exactly with completed current pit stop (inlap)
                if math.isclose(cur_phase[1], cur_inlap):
                    t_pit_before_fcy_start_end[idx_fcy_phase][1] += t_pit_inlap

                # check if current FCY phase ends within the lap after completed current pit stop (outlap)
                elif cur_inlap + lap_fraction_pit_outlap < cur_phase[1] <= cur_inlap + 1.0:
                    t_pit_before_fcy_start_end[idx_fcy_phase][1] += t_pit_outlap

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER FCY PHASES ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    It is important that all lap times are finally calculated until this point, i.e. considering the FCY phases must
    be the last step within this function since we use the summed lap times to calculate the FCY start and stop race
    times. The SC ghost concept is intentionally not used for this basic simulation. Instead, it is assumed that the SC
    waits for the driver at the beginning of the next lap to run up.
    """

    if fcy_phases is not None:
        # create list to save converted FCY phases with the following content
        # case VSC: [start race time, end race time, type, None, None]
        # case SC: [start race time, end race time, type, SC delay in seconds, SC duration in laps]
        fcy_phases_conv = [[None, None, x[2], None, None] for x in fcy_phases]

        # loop through the phases from front to back (it was assured that the phases are sorted in an ascending order)
        for idx_phase, cur_phase in enumerate(fcy_phases):
            # loop through the affected laps
            start_idx = math.floor(cur_phase[0])
            stop_idx = math.ceil(cur_phase[1])

            for idx_lap in range(start_idx, stop_idx):
                # determine affected lap fraction, slow lap time depending on FCY phase type and FCY phase start and
                # stop race times
                cur_progress = float(idx_lap)  # current race progress at start of lap

                if cur_progress <= cur_phase[0]:
                    # CASE 1: FCY phase starts within current lap (and might also end here) ----------------------------
                    # get lap fraction driven with normal speed before FCY phase
                    lap_frac_normal_bef = cur_phase[0] - cur_progress

                    # get lap fraction driven with normal speed after FCY phase (if FCY phase is also ended within this
                    # lap)
                    if cur_progress + 1.0 >= cur_phase[1]:
                        lap_frac_normal_aft = cur_progress + 1.0 - cur_phase[1]
                    else:
                        lap_frac_normal_aft = 0.0

                    # calculate lap fractions driven normally and slowly
                    lap_frac_normal = lap_frac_normal_bef + lap_frac_normal_aft
                    lap_frac_slow = 1.0 - lap_frac_normal

                    # determine slow lap time -> in the case of an SC phase this is nevertheless the FCY lap time in the
                    # first lap of the phase, since the driver did not run up to the SC so far (SC waits for the driver
                    # in the new lap)
                    t_lap_slow = t_lap_fcy

                    # fill start entry of converted FCY phase and consider possibly finished pit stops before start
                    # (t_laps_pit is not considered for the current lap therefore)
                    fcy_phases_conv[idx_phase][0] = (np.sum(t_laps[:idx_lap] + t_laps_pit[:idx_lap])
                                                     + lap_frac_normal_bef * t_laps[idx_lap]
                                                     + t_pit_before_fcy_start_end[idx_phase][0])

                    # if the phase affects only one lap also fill the end entry of converted FCY phase
                    if cur_progress + 1.0 >= cur_phase[1]:
                        if math.isclose(cur_phase[1], tot_no_laps):
                            # CASE 1: phase lasts until the end of the race -> set end race time inf
                            fcy_phases_conv[idx_phase][1] = math.inf
                        else:
                            # CASE 2: normal case
                            fcy_phases_conv[idx_phase][1] = fcy_phases_conv[idx_phase][0] + lap_frac_slow * t_lap_slow

                elif cur_progress + 1.0 >= cur_phase[1]:
                    # CASE 2: FCY phase was already started before and ends within current lap -------------------------
                    # determine lap fractions and slow lap time
                    if cur_phase[2] == 'SC':
                        # for the SC it is assumed that it ends exactly on the finish line
                        lap_frac_normal = 0.0
                        lap_frac_slow = 1.0
                        t_lap_slow = t_lap_sc
                    else:
                        lap_frac_normal = cur_progress + 1.0 - cur_phase[1]
                        lap_frac_slow = 1.0 - lap_frac_normal
                        t_lap_slow = t_lap_fcy

                    # fill stop entry of converted FCY phase (t_laps_pit is not considered for the current lap
                    # therefore)
                    if math.isclose(cur_phase[1], tot_no_laps):
                        # CASE 1: phase lasts until the end of the race -> set end race time inf
                        fcy_phases_conv[idx_phase][1] = math.inf
                    else:
                        # CASE 2: normal case
                        fcy_phases_conv[idx_phase][1] = (np.sum(t_laps[:idx_lap] + t_laps_pit[:idx_lap])
                                                         + lap_frac_slow * t_lap_slow
                                                         + t_pit_before_fcy_start_end[idx_phase][1])

                else:
                    # CASE 3: whole lap affected by FCY phase (neither starting nor ending here) -----------------------
                    lap_frac_normal = 0.0
                    lap_frac_slow = 1.0

                    # determine slow lap time
                    if cur_phase[2] == 'SC':
                        t_lap_slow = t_lap_sc
                    else:
                        t_lap_slow = t_lap_fcy

                # set lap time (t_laps_pit not affected)
                t_lap_tmp = lap_frac_normal * t_laps[idx_lap] + lap_frac_slow * t_lap_slow

                if t_laps[idx_lap] < t_lap_tmp:
                    t_laps[idx_lap] = t_lap_tmp
                else:
                    print("WARNING: The calculated lap time affected by the FCY phase is faster than the normal lap"
                          " time. This should be checked!")

            # calculate SC delay and duration of SC phase
            if cur_phase[2] == 'SC':
                # get the race time at the end of the lap in which the SC starts
                t_race_sc_start = np.sum(t_laps[:start_idx + 1] + t_laps_pit[:start_idx + 1])

                # calculate the difference to SC phase start and save this information as SC delay into the phase
                fcy_phases_conv[idx_phase][3] = t_race_sc_start - fcy_phases_conv[idx_phase][0]

                # assure that SC delay in first lap is at least 33% of the SC lap time since the SC is driving slower in
                # the beginning to ease it for the drivers to catch up
                if fcy_phases_conv[idx_phase][3] < 0.33 * t_lap_sc:
                    # get required difference to original SC delay
                    t_sc_delay_diff = 0.33 * t_lap_sc - fcy_phases_conv[idx_phase][3]

                    # increase SC delay, SC end time as well as first lap time behind the SC by required difference
                    fcy_phases_conv[idx_phase][3] += t_sc_delay_diff
                    fcy_phases_conv[idx_phase][1] += t_sc_delay_diff  # works also in case of end = math.inf
                    if start_idx + 1 < tot_no_laps:
                        # increasing the lap time is only valid if lap after start of the SC is part of the race
                        t_laps[start_idx + 1] += t_sc_delay_diff

                # calculate SC duration (full laps only) -> set inf if SC phase lasts until the end of the race
                if math.isclose(cur_phase[1], tot_no_laps):
                    fcy_phases_conv[idx_phase][4] = math.inf
                else:
                    fcy_phases_conv[idx_phase][4] = stop_idx - start_idx - 1

    else:
        fcy_phases_conv = None

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE LAPWISE RACE TIMES -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    t_race_lapwise = np.cumsum(t_laps + t_laps_pit)

    return t_race_lapwise, fcy_phases_conv


def __perform_pitstop_standstill(t_pit_tirechange: float, drivetype: str, cur_stop: list, t_pit_refuel_perkg: float,
                                 t_pit_charge_perkwh: float) -> float:
    """This method is used to calculate the correct standstill time while giving the possibility to refuel/recharge."""

    # standstill time for tire change
    timeloss_standstill = t_pit_tirechange

    # refueling / recharging -> check if it lasts longer than tire change standstill time
    if drivetype == 'combustion' \
            and cur_stop[3] != 0.0 \
            and cur_stop[3] * t_pit_refuel_perkg > timeloss_standstill:
        timeloss_standstill = cur_stop[3] * t_pit_refuel_perkg

    elif drivetype == 'electric' \
            and cur_stop[3] != 0.0 \
            and cur_stop[3] * t_pit_charge_perkwh > timeloss_standstill:
        timeloss_standstill = cur_stop[3] * t_pit_charge_perkwh

    return timeloss_standstill
