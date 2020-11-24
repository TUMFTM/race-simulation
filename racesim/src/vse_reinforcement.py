import numpy as np
import tensorflow as tf
import pickle
import machine_learning


class VSE_REINFORCEMENT(object):
    """
    author:
    Alexander Heilmeier

    date:
    28.05.2020

    .. description::
    The NN model is a Q net in this case, X_conv contains the converted observation.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__cat_preprocessor",
                 "__nn_model",
                 "__X_conv",
                 "__no_actions")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 preprocessor_path: str,
                 nnmodel_path: str) -> None:

        # use preprocessor for one-hot encoding of categorical features, numerical features must not be processed for
        # reinforcement agent
        with open(preprocessor_path, 'rb') as fh:
            self.cat_preprocessor = pickle.load(fh)

        self.nn_model = {"interpreter": tf.lite.Interpreter(model_path=nnmodel_path)}
        self.X_conv = None

        # initialize tf lite interpreter
        self.nn_model["interpreter"].allocate_tensors()
        self.nn_model["input_index"] = self.nn_model["interpreter"].get_input_details()[0]['index']
        self.nn_model["output_index"] = self.nn_model["interpreter"].get_output_details()[0]['index']

        # get number of available actions for current race (3 actions if there are 2 available dry compounds (2014 and
        # 2015), 4 actions if there are 3 available dry compounds (>= 2016)
        self.no_actions = int(self.nn_model["interpreter"].get_output_details()[0]['shape'][1])

        if self.no_actions not in [3, 4]:
            raise RuntimeError("RL VSE does not match current race, the number of available actions should be 3 (2014 +"
                               " 2015) or 4 (>= 2016)!")

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_cat_preprocessor(self) -> machine_learning.src.preprocessor.Preprocessor: return self.__cat_preprocessor

    def __set_cat_preprocessor(self, x: machine_learning.src.preprocessor.Preprocessor) -> None:
        self.__cat_preprocessor = x
    cat_preprocessor = property(__get_cat_preprocessor, __set_cat_preprocessor)

    def __get_nn_model(self) -> dict: return self.__nn_model
    def __set_nn_model(self, x: dict) -> None: self.__nn_model = x
    nn_model = property(__get_nn_model, __set_nn_model)

    def __get_X_conv(self) -> np.ndarray: return self.__X_conv
    def __set_X_conv(self, x: np.ndarray) -> None: self.__X_conv = x
    X_conv = property(__get_X_conv, __set_X_conv)

    def __get_no_actions(self) -> int: return self.__no_actions
    def __set_no_actions(self, x: int) -> None: self.__no_actions = x
    no_actions = property(__get_no_actions, __set_no_actions)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        # reset VSE such that it can be used to simulate (same race) again
        self.X_conv = None

    def preprocess_features(self,
                            raceprogress_curlap: float,
                            rel_position: list,
                            rel_est_pos_loss: list,
                            tireageprogress_corr: list,
                            cur_compound: list,
                            used_2compounds: list,
                            fcy_stat_curlap: list,
                            close_behind: list,
                            close_ahead: list,
                            defendable_undercut: list,
                            driver_initials: list) -> None:

        if self.cat_preprocessor.no_transf_cols == 0:
            raise RuntimeError("Preprocessor was not trained!")

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPARATION ------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # get number of drivers
        no_drivers_tmp = len(rel_position)

        # prepare X_cat (categorical features only) -> bools are implicitly converted to floats
        X_cat = np.zeros((no_drivers_tmp, 7))
        X_cat[:, 0] = self.cat_preprocessor.transform_cat_dict(X_cat_str=cur_compound, featurename='cur_compound')
        X_cat[:, 1] = used_2compounds
        X_cat[:, 2] = fcy_stat_curlap
        X_cat[:, 3] = close_behind
        X_cat[:, 4] = close_ahead
        X_cat[:, 5] = defendable_undercut
        X_cat[:, 6] = self.cat_preprocessor.transform_cat_dict(X_cat_str=driver_initials, featurename='driver_initials')

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPROCESSING ----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialize X_conv if called for the first time
        if self.X_conv is None:
            self.X_conv = np.zeros((no_drivers_tmp, 4 + self.cat_preprocessor.no_transf_cols), dtype=np.float32)

        # preprocessing (numerical features)
        self.X_conv[:, 0] = raceprogress_curlap
        self.X_conv[:, 1] = rel_position
        self.X_conv[:, 2] = rel_est_pos_loss
        self.X_conv[:, 3] = tireageprogress_corr

        # preprocessing (categorical features) -> one-hot encoding
        self.X_conv[:, 4:] = self.cat_preprocessor.transform(X=X_cat, dtype_out=np.float32)

    def make_decision(self,
                      bool_driving: list or np.ndarray,
                      param_dry_compounds: list,
                      used_2compounds: list,
                      cur_compounds: list,
                      raceprogress_prevlap: float) -> list:

        # get number of drivers and create output list and array for Q values of the possible actions (required for the
        # check if each driver used 2 compounds)
        no_drivers_tmp = self.X_conv.shape[0]
        next_compounds = [None] * no_drivers_tmp
        action_q_vals = np.zeros((no_drivers_tmp, self.no_actions), dtype=np.float32)

        # --------------------------------------------------------------------------------------------------------------
        # TIRE CHANGE AND COMPOUND CHOICE DECISION ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        for idx_driver in range(no_drivers_tmp):
            # continue if driver does not participate anymore
            if not bool_driving[idx_driver]:
                continue

            # set NN input
            self.nn_model["interpreter"].set_tensor(self.nn_model["input_index"],
                                                    np.expand_dims(self.X_conv[idx_driver], axis=0))

            # invoke NN
            self.nn_model["interpreter"].invoke()

            # fetch NN output
            action_q_vals[idx_driver] = self.nn_model["interpreter"].get_tensor(self.nn_model["output_index"])[0]

            # use action with highest Q value
            action = action_q_vals[idx_driver].argmax()

            if action != 0:
                next_compounds[idx_driver] = param_dry_compounds[action - 1]

        # assure that every driver used two different compounds in a race (as soon as a raceprogress of 95% is exceeded)
        if raceprogress_prevlap > 0.95 and not all(used_2compounds):
            for idx_driver in range(no_drivers_tmp):
                if not used_2compounds[idx_driver] \
                        and bool_driving[idx_driver] \
                        and (next_compounds[idx_driver] is None
                             or next_compounds[idx_driver] == cur_compounds[idx_driver]):

                    # get indices of actions sorted from best to worst action (i.e. from highest to lowest Q value)
                    action_list_sorted = list(np.argsort(-action_q_vals[idx_driver]))

                    # loop through available actions
                    for action in action_list_sorted:
                        if action != 0 and param_dry_compounds[action - 1] != cur_compounds[idx_driver]:
                            # set new compound
                            next_compounds[idx_driver] = param_dry_compounds[action - 1]
                            break

                    print("WARNING: Had to enforce a pit stop or another tire compound for reinforcement VSE above"
                          " 95%% race progress (driver at index %i of reinforcement drivers)!" % idx_driver)

        return next_compounds


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
