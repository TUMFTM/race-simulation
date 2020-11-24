class Track(object):
    """
    author:
    Alexander Heilmeier

    date:
    01.11.2017

    .. description::
    XXXX
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__name",
                 "__t_q",
                 "__t_gap_racepace",
                 "__t_lap_fcy",
                 "__t_lap_sc",
                 "__t_lap_sens_mass",
                 "__t_pit_tirechange_min",
                 "__t_pitdrive_inlap",
                 "__t_pitdrive_outlap",
                 "__t_pitdrive_inlap_fcy",
                 "__t_pitdrive_outlap_fcy",
                 "__t_pitdrive_inlap_sc",
                 "__t_pitdrive_outlap_sc",
                 "__pits_aft_finishline",
                 "__t_loss_pergridpos",
                 "__t_loss_firstlap",
                 "__t_gap_overtake",
                 "__t_gap_overtake_vel",
                 "__t_drseffect")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, track_pars: dict) -> None:
        # general
        self.name = track_pars["name"]

        # base time and laptime sensitivity
        self.t_q = track_pars["t_q"]                                        # [s] best qualifying laptime
        self.t_gap_racepace = track_pars["t_gap_racepace"]                  # [s] time gap between qualifying and race
        self.t_lap_sens_mass = track_pars["t_lap_sens_mass"]                # [s/kg] laptime mass sensitivity

        # set estimated FCY and SC lap times
        t_base = self.t_q + self.t_gap_racepace                             # [s] base lap time of the race
        self.t_lap_fcy = t_base * track_pars["mult_t_lap_fcy"]              # [s] estimated FCY lap time
        self.t_lap_sc = t_base * track_pars["mult_t_lap_sc"]                # [s] estimated SC lap time

        # pit
        self.t_pit_tirechange_min = track_pars["t_pit_tirechange_min"]      # [s] minimal standstill time for tirechange
        self.t_pitdrive_inlap = track_pars["t_pitdrive_inlap"]              # [s] time loss inlap
        self.t_pitdrive_outlap = track_pars["t_pitdrive_outlap"]            # [s] time loss outlap (wo standstill)
        self.t_pitdrive_inlap_fcy = track_pars["t_pitdrive_inlap_fcy"]      # [s] time loss inlap (FCY)
        self.t_pitdrive_outlap_fcy = track_pars["t_pitdrive_outlap_fcy"]    # [s] time loss outlap (wo standstill) (FCY)
        self.t_pitdrive_inlap_sc = track_pars["t_pitdrive_inlap_sc"]        # [s] time loss inlap (SC)
        self.t_pitdrive_outlap_sc = track_pars["t_pitdrive_outlap_sc"]      # [s] time loss outlap (wo standstill) (SC)
        # [-] indicates if pits are located before or after the finish line
        self.pits_aft_finishline = track_pars["pits_aft_finishline"]

        # starting grid
        self.t_loss_pergridpos = track_pars["t_loss_pergridpos"]    # [s/pos] timeloss in first lap due to grid position
        self.t_loss_firstlap = track_pars["t_loss_firstlap"]        # [s] timeloss in first lap due to standstill start

        # overtake
        self.t_gap_overtake = track_pars["t_gap_overtake"]                  # [s] track-dependent factor
        self.t_gap_overtake_vel = track_pars["t_gap_overtake_vel"]          # [s/km/h]  velocity-dependent factor
        self.t_drseffect = track_pars["t_drseffect"]                        # [s] time gain due to DRS (negative!)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_name(self) -> str: return self.__name
    def __set_name(self, x: str) -> None: self.__name = x
    name = property(__get_name, __set_name)

    def __get_t_q(self) -> float: return self.__t_q

    def __set_t_q(self, x: float) -> None:
        if not 30.0 < x < 200.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_q = x
    t_q = property(__get_t_q, __set_t_q)

    def __get_t_gap_racepace(self) -> float: return self.__t_gap_racepace

    def __set_t_gap_racepace(self, x: float) -> None:
        if not -20.0 <= x < 10.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_gap_racepace = x
    t_gap_racepace = property(__get_t_gap_racepace, __set_t_gap_racepace)

    def __get_t_lap_fcy(self) -> float: return self.__t_lap_fcy

    def __set_t_lap_fcy(self, x: float) -> None:
        if not 30.0 <= x < 300.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_lap_fcy = x
    t_lap_fcy = property(__get_t_lap_fcy, __set_t_lap_fcy)

    def __get_t_lap_sc(self) -> float: return self.__t_lap_sc

    def __set_t_lap_sc(self, x: float) -> None:
        if not 30.0 <= x < 300.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_lap_sc = x
    t_lap_sc = property(__get_t_lap_sc, __set_t_lap_sc)

    def __get_t_lap_sens_mass(self) -> float: return self.__t_lap_sens_mass

    def __set_t_lap_sens_mass(self, x: float) -> None:
        if not 0.0 < x < 1.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_lap_sens_mass = x
    t_lap_sens_mass = property(__get_t_lap_sens_mass, __set_t_lap_sens_mass)

    def __get_t_pit_tirechange_min(self) -> float: return self.__t_pit_tirechange_min

    def __set_t_pit_tirechange_min(self, x: float) -> None:
        if not 1.5 < x < 10.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pit_tirechange_min = x
    t_pit_tirechange_min = property(__get_t_pit_tirechange_min, __set_t_pit_tirechange_min)

    def __get_t_pitdrive_inlap(self) -> float: return self.__t_pitdrive_inlap

    def __set_t_pitdrive_inlap(self, x: float) -> None:
        if not -5.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_inlap = x
    t_pitdrive_inlap = property(__get_t_pitdrive_inlap, __set_t_pitdrive_inlap)

    def __get_t_pitdrive_outlap(self) -> float: return self.__t_pitdrive_outlap

    def __set_t_pitdrive_outlap(self, x: float) -> None:
        if not -5.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_outlap = x
    t_pitdrive_outlap = property(__get_t_pitdrive_outlap, __set_t_pitdrive_outlap)

    def __get_t_pitdrive_inlap_fcy(self) -> float: return self.__t_pitdrive_inlap_fcy

    def __set_t_pitdrive_inlap_fcy(self, x: float) -> None:
        if not -15.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_inlap_fcy = x
    t_pitdrive_inlap_fcy = property(__get_t_pitdrive_inlap_fcy, __set_t_pitdrive_inlap_fcy)

    def __get_t_pitdrive_outlap_fcy(self) -> float: return self.__t_pitdrive_outlap_fcy

    def __set_t_pitdrive_outlap_fcy(self, x: float) -> None:
        if not -5.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_outlap_fcy = x
    t_pitdrive_outlap_fcy = property(__get_t_pitdrive_outlap_fcy, __set_t_pitdrive_outlap_fcy)

    def __get_t_pitdrive_inlap_sc(self) -> float: return self.__t_pitdrive_inlap_sc

    def __set_t_pitdrive_inlap_sc(self, x: float) -> None:
        if not -15.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_inlap_sc = x
    t_pitdrive_inlap_sc = property(__get_t_pitdrive_inlap_sc, __set_t_pitdrive_inlap_sc)

    def __get_t_pitdrive_outlap_sc(self) -> float: return self.__t_pitdrive_outlap_sc

    def __set_t_pitdrive_outlap_sc(self, x: float) -> None:
        if not -5.0 < x < 30.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_pitdrive_outlap_sc = x
    t_pitdrive_outlap_sc = property(__get_t_pitdrive_outlap_sc, __set_t_pitdrive_outlap_sc)

    def __get_pits_aft_finishline(self) -> bool: return self.__pits_aft_finishline
    def __set_pits_aft_finishline(self, x: bool) -> None: self.__pits_aft_finishline = x
    pits_aft_finishline = property(__get_pits_aft_finishline, __set_pits_aft_finishline)

    def __get_t_loss_pergridpos(self) -> float: return self.__t_loss_pergridpos

    def __set_t_loss_pergridpos(self, x: float) -> None:
        if not 0.0 < x < 10.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_loss_pergridpos = x
    t_loss_pergridpos = property(__get_t_loss_pergridpos, __set_t_loss_pergridpos)

    def __get_t_loss_firstlap(self) -> float: return self.__t_loss_firstlap

    def __set_t_loss_firstlap(self, x: float) -> None:
        if not 0.0 < x < 15.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_loss_firstlap = x
    t_loss_firstlap = property(__get_t_loss_firstlap, __set_t_loss_firstlap)

    def __get_t_gap_overtake(self) -> float: return self.__t_gap_overtake

    def __set_t_gap_overtake(self, x: float) -> None:
        if not 0.0 < x < 5.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_gap_overtake = x
    t_gap_overtake = property(__get_t_gap_overtake, __set_t_gap_overtake)

    def __get_t_gap_overtake_vel(self) -> float: return self.__t_gap_overtake_vel

    def __set_t_gap_overtake_vel(self, x: float) -> None:
        if not -0.2 < x <= 0.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_gap_overtake_vel = x
    t_gap_overtake_vel = property(__get_t_gap_overtake_vel, __set_t_gap_overtake_vel)

    def __get_t_drseffect(self) -> float: return self.__t_drseffect

    def __set_t_drseffect(self, x: float) -> None:
        if not -10.0 < x <= 0.0:
            raise RuntimeError("Unreasonable value!", x)
        self.__t_drseffect = x
    t_drseffect = property(__get_t_drseffect, __set_t_drseffect)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # no methods defined


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
