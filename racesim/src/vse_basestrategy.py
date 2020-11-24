import numpy as np


class VSE_BASESTRATEGY(object):
    """
    author:
    Alexander Heilmeier

    date:
    29.06.2020

    .. description::
    XXXX
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = "__base_strategies"

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, base_strategies: dict) -> None:
        self.base_strategies = base_strategies

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_base_strategies(self) -> dict: return self.__base_strategies
    def __set_base_strategies(self, x: dict) -> None: self.__base_strategies = x
    base_strategies = property(__get_base_strategies, __set_base_strategies)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        pass

    def make_decision(self,
                      cur_lap: int,
                      driver_initials: list,
                      bool_driving: list or np.ndarray,
                      param_dry_compounds: list) -> list:

        # --------------------------------------------------------------------------------------------------------------
        # TAKE DECISIONS -----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        next_compounds = [None] * len(driver_initials)

        for idx_driver, initials in enumerate(driver_initials):
            # continue if driver retired
            if not bool_driving[idx_driver]:
                continue

            # get next compound if current lap is an inlap
            next_compound = next((stint_data[1] for stint_data in self.base_strategies[initials]
                                  if stint_data[0] == cur_lap), None)

            # continue of driver does not change tires this lap
            if next_compound is None:
                continue

            # set new compound
            if next_compound not in param_dry_compounds:
                raise RuntimeError("Chosen compound is not parameterized!")
            else:
                next_compounds[idx_driver] = next_compound

        return next_compounds


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
