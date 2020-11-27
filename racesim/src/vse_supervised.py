import pickle
import numpy as np
import tensorflow as tf


class VSE_SUPERVISED(object):
    """
    author:
    Alexander Heilmeier

    date:
    02.04.2020

    .. description::
    This class provides handles two neural networks (and the according preprocessors) to take tire change decisions,
    i.e. the timing of the pit stop (tc = tirechange) as well as the compound decision (cc = compound choice), during
    races on the basis of the current situation. It provides four main methods: preprocessing of the data for tc and cc
    and taking the decision for tc and cc. Keep in mind that some of the inputs have to be in the state at the end of
    the previous lap!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__preprocessor_cc",
                 "__preprocessor_tc",
                 "__nnmodel_cc",
                 "__nnmodel_tc",
                 "__X_conv_cc",
                 "__X_conv_tc",
                 "__no_timesteps_tc")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 preprocessor_cc_path: str,
                 preprocessor_tc_path: str,
                 nnmodel_cc_path: str,
                 nnmodel_tc_path: str) -> None:

        with open(preprocessor_cc_path, 'rb') as fh:
            self.preprocessor_cc = pickle.load(fh)

        with open(preprocessor_tc_path, 'rb') as fh:
            self.preprocessor_tc = pickle.load(fh)

        self.nnmodel_cc = {"interpreter": tf.lite.Interpreter(model_path=nnmodel_cc_path)}
        self.nnmodel_tc = {"interpreter": tf.lite.Interpreter(model_path=nnmodel_tc_path)}

        self.X_conv_cc = None
        self.X_conv_tc = None

        # initialize tf lite interpreters
        self.nnmodel_cc["interpreter"].allocate_tensors()
        self.nnmodel_cc["input_index"] = self.nnmodel_cc["interpreter"].get_input_details()[0]['index']
        self.nnmodel_cc["output_index"] = self.nnmodel_cc["interpreter"].get_output_details()[0]['index']

        self.nnmodel_tc["interpreter"].allocate_tensors()
        self.nnmodel_tc["input_index"] = self.nnmodel_tc["interpreter"].get_input_details()[0]['index']
        self.nnmodel_tc["output_index"] = self.nnmodel_tc["interpreter"].get_output_details()[0]['index']

        self.no_timesteps_tc = self.nnmodel_tc["interpreter"].get_input_details()[0]['shape'][1]

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_preprocessor_cc(self): return self.__preprocessor_cc
    def __set_preprocessor_cc(self, x) -> None: self.__preprocessor_cc = x
    preprocessor_cc = property(__get_preprocessor_cc, __set_preprocessor_cc)

    def __get_preprocessor_tc(self): return self.__preprocessor_tc
    def __set_preprocessor_tc(self, x) -> None: self.__preprocessor_tc = x
    preprocessor_tc = property(__get_preprocessor_tc, __set_preprocessor_tc)

    def __get_nnmodel_cc(self) -> dict: return self.__nnmodel_cc
    def __set_nnmodel_cc(self, x: dict) -> None: self.__nnmodel_cc = x
    nnmodel_cc = property(__get_nnmodel_cc, __set_nnmodel_cc)

    def __get_nnmodel_tc(self) -> dict: return self.__nnmodel_tc
    def __set_nnmodel_tc(self, x: dict) -> None: self.__nnmodel_tc = x
    nnmodel_tc = property(__get_nnmodel_tc, __set_nnmodel_tc)

    def __get_X_conv_cc(self) -> np.ndarray: return self.__X_conv_cc
    def __set_X_conv_cc(self, x: np.ndarray) -> None: self.__X_conv_cc = x
    X_conv_cc = property(__get_X_conv_cc, __set_X_conv_cc)

    def __get_X_conv_tc(self) -> np.ndarray: return self.__X_conv_tc
    def __set_X_conv_tc(self, x: np.ndarray) -> None: self.__X_conv_tc = x
    X_conv_tc = property(__get_X_conv_tc, __set_X_conv_tc)

    def __get_no_timesteps_tc(self) -> int: return self.__no_timesteps_tc
    def __set_no_timesteps_tc(self, x: int) -> None: self.__no_timesteps_tc = x
    no_timesteps_tc = property(__get_no_timesteps_tc, __set_no_timesteps_tc)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        # reset VSE such that it can be used to simulate (same race) again
        self.X_conv_cc = None
        self.X_conv_tc = None

    def preprocess_features(self,
                            # TC -----------
                            tireageprogress_corr_zeroinchange: list,
                            raceprogress: float,
                            position: list,
                            rel_compound_num_nl: list,
                            fcy_stat_nl: list,
                            remainingtirechanges_nl: list,
                            tirechange_pursuer: list,
                            location_cat: int,
                            close_ahead_prevlap: list,
                            # CC -----------
                            location: str,
                            used_2compounds_nl: list,
                            no_avail_dry_compounds: int) -> None:

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPARATION (TIRE CHANGE DECISION) -------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # get number of drivers
        no_drivers_tmp = len(position)

        # set everything together
        X = np.zeros((no_drivers_tmp, 9))  # currently we have 9 input features
        X[:, 0] = tireageprogress_corr_zeroinchange
        X[:, 1] = raceprogress
        X[:, 2] = position
        X[:, 3] = rel_compound_num_nl
        X[:, 4] = fcy_stat_nl
        X[:, 5] = remainingtirechanges_nl
        X[:, 6] = tirechange_pursuer
        X[:, 7] = location_cat
        X[:, 8] = close_ahead_prevlap

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPROCESSING (TIRE CHANGE DECISION) -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.X_conv_tc is None:
            # initialize X_conv_tc if called for the first time
            self.X_conv_tc = np.zeros((no_drivers_tmp, self.no_timesteps_tc, self.preprocessor_tc.no_transf_cols),
                                      dtype=np.float32)

            # set correct initial values for every driver
            for idx_driver in range(no_drivers_tmp):
                X_conv_tc_tmp = np.tile(X[idx_driver], (self.no_timesteps_tc, 1))

                # set FCY status 0 for every lap except lap 0
                X_conv_tc_tmp[:-1, 4] = 0

                # process features
                self.X_conv_tc[idx_driver] = self.preprocessor_tc.transform(X_conv_tc_tmp, dtype_out=np.float32)

        else:
            # process new features
            X_conv_tc_tmp = self.preprocessor_tc.transform(X, dtype_out=np.float32)

            # replace last entry in X_conv_tc for every driver by new data
            self.X_conv_tc = np.roll(self.X_conv_tc, -1, axis=1)
            self.X_conv_tc[:, -1] = X_conv_tc_tmp

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPARATION (COMPOUND CHOICE DECISION) ---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        X = np.zeros((no_drivers_tmp, 6))  # currently we have 6 input features
        X[:, 0] = raceprogress
        X[:, 1] = self.preprocessor_cc.transform_cat_dict(X_cat_str=[location], featurename='location')
        X[:, 2] = rel_compound_num_nl
        X[:, 3] = remainingtirechanges_nl
        X[:, 4] = used_2compounds_nl
        X[:, 5] = no_avail_dry_compounds

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPROCESSING (COMPOUND CHOICE DECISION) -------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # process new features
        self.X_conv_cc = self.preprocessor_cc.transform(X, dtype_out=np.float32)

    def make_decision(self,
                      bool_driving: list or np.ndarray,
                      avail_dry_compounds: list,
                      param_dry_compounds: list,
                      remainingtirechanges_curlap: list,
                      used_2compounds: list,
                      cur_compounds: list,
                      raceprogress_prevlap: float) -> list:

        # get number of drivers and create output list
        no_drivers_tmp = self.X_conv_tc.shape[0]
        next_compounds = [None] * no_drivers_tmp

        # --------------------------------------------------------------------------------------------------------------
        # TIRE CHANGE DECISION -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create array for prediction probabilities
        pitstop_probs = np.zeros(no_drivers_tmp, dtype=np.float32)

        for idx_driver in range(no_drivers_tmp):
            # continue if driver does not participate anymore
            if not bool_driving[idx_driver]:
                continue

            # set NN input
            self.nnmodel_tc["interpreter"].set_tensor(self.nnmodel_tc["input_index"],
                                                      np.expand_dims(self.X_conv_tc[idx_driver], axis=0))

            # invoke NN
            self.nnmodel_tc["interpreter"].invoke()

            # fetch NN output
            pitstop_probs[idx_driver] = self.nnmodel_tc["interpreter"].get_tensor(self.nnmodel_tc["output_index"])

        # get indices of the drivers that have a predicted pitstop probability above 50%
        idxs_driver_pitstop = list(np.flatnonzero(np.round(pitstop_probs)))

        # assure that every driver used two different compounds in a race (as soon as a raceprogress of 90% is exceeded)
        if raceprogress_prevlap > 0.9 and not all(used_2compounds):
            for idx_driver in range(no_drivers_tmp):
                if not used_2compounds[idx_driver] \
                        and bool_driving[idx_driver] \
                        and idx_driver not in idxs_driver_pitstop:

                    idxs_driver_pitstop.append(idx_driver)
                    print("WARNING: Had to enforce a pit stop for supervised VSE above 90%% race progress (driver at"
                          " index %i of supervised drivers)!" % idx_driver)

            idxs_driver_pitstop.sort()

        # --------------------------------------------------------------------------------------------------------------
        # COMPOUND CHOICE DECISION -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # compound choice is only performed if any driver was chosen for a pit stop
        if idxs_driver_pitstop:

            # create array for prediction probabilities (NN was trained with 3 different compounds to choose from)
            rel_compound_probs = np.zeros((len(idxs_driver_pitstop), 3), dtype=np.float32)

            for idx_rel, idx_abs in enumerate(idxs_driver_pitstop):
                # set NN input
                self.nnmodel_cc["interpreter"].set_tensor(self.nnmodel_cc["input_index"],
                                                          np.expand_dims(self.X_conv_cc[idx_abs], axis=0))

                # invoke NN
                self.nnmodel_cc["interpreter"].invoke()

                # fetch NN output
                rel_compound_probs[idx_rel] = \
                    self.nnmodel_cc["interpreter"].get_tensor(self.nnmodel_cc["output_index"])

            # get array with indices of relative compounds sorted by highest -> lowest probability
            idxs_rel_compound_sorted = list(np.argsort(-rel_compound_probs, axis=1))

            # make sure that VSE chooses only from available compounds (only 2 different compounds were available before
            # 2016, some of the compounds might not be parameterized in the current race for some drivers) -> use the
            # compound with the next lower probability if chosen compound is not available
            for idx_rel, idx_abs in enumerate(idxs_driver_pitstop):

                # loop through relative compound indices from highest to lowest probability
                for idx_rel_compound in idxs_rel_compound_sorted[idx_rel]:

                    # case two available compounds: in 2014 and 2015 we have only medium and soft -> subtract 1 from
                    # decision to fit available compounds with indices 0 and 1
                    if len(avail_dry_compounds) == 2:
                        idx_rel_compound_corr = idx_rel_compound - 1

                        # continue to compound with next lower probability if current compound is not available in the
                        # race (index - 1) or not parameterized
                        if idx_rel_compound_corr < 0 \
                                or avail_dry_compounds[idx_rel_compound_corr] not in param_dry_compounds:
                            continue

                    # case three available compounds
                    else:
                        idx_rel_compound_corr = idx_rel_compound

                        # continue to compound with next lower probability if current compound is not parameterized
                        if avail_dry_compounds[idx_rel_compound_corr] not in param_dry_compounds:
                            continue

                    # continue to compound with next lower probability if this is the last planned pit stop (or if it is
                    # probably the last pit stop because 90% race progress are already exceeded) and if the driver would
                    # not drive two different compounds in the race if the current compound was chosen
                    if (remainingtirechanges_curlap[idx_abs] == 1 or raceprogress_prevlap > 0.9) \
                            and not used_2compounds[idx_abs] \
                            and avail_dry_compounds[idx_rel_compound_corr] == cur_compounds[idx_abs]:
                        continue

                    # set new compound
                    next_compounds[idx_abs] = avail_dry_compounds[idx_rel_compound_corr]
                    break

        return next_compounds


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
