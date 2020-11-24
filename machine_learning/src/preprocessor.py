import numpy as np
import math


class Preprocessor(object):
    """
    author:
    Alexander Heilmeier

    date:
    03.04.2020

    .. description::
    Preprocessing of ML features completely based on numpy.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__idxs_num",
                 "__idxs_cat",
                 "__idxs_buck",
                 "__col_info",
                 "__no_transf_cols",
                 "__cat_dict",
                 "__team_translation_dict")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, feature_types: list, bins_buck: list) -> None:
        # convert feature types list into indices
        self.idxs_num = [idx for idx in range(len(feature_types)) if feature_types[idx] == 'num']
        self.idxs_cat = [idx for idx in range(len(feature_types)) if feature_types[idx] == 'cat']
        self.idxs_buck = [idx for idx in range(len(feature_types)) if feature_types[idx] == 'buck']
        self.col_info = {}
        self.no_transf_cols = 0
        self.cat_dict = {}
        self.team_translation_dict = {"ForceIndia": "RacingPoint",
                                      "Sauber": "AlfaRomeo",
                                      "Marussia": "ManorMarussia",
                                      "LotusF1": "Renault",
                                      "ToroRosso": "AlphaTauri"}

        # check input
        if len(self.idxs_buck) != len(bins_buck):
            raise RuntimeError("Number of bucketized features does not fit number of inserted bins!")

        # handle bucketized columns (bins_buck holds only the separation marks, left and right are -inf, +inf)
        for idx_0, idx_buck in enumerate(self.idxs_buck):
            self.col_info[idx_buck] = {'bins': np.array(bins_buck[idx_0] + [math.inf]),
                                       'no_classes': len(bins_buck[idx_0]) + 1}

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_idxs_num(self) -> list: return self.__idxs_num
    def __set_idxs_num(self, x: list) -> None: self.__idxs_num = x
    idxs_num = property(__get_idxs_num, __set_idxs_num)

    def __get_idxs_cat(self) -> list: return self.__idxs_cat
    def __set_idxs_cat(self, x: list) -> None: self.__idxs_cat = x
    idxs_cat = property(__get_idxs_cat, __set_idxs_cat)

    def __get_idxs_buck(self) -> list: return self.__idxs_buck
    def __set_idxs_buck(self, x: list) -> None: self.__idxs_buck = x
    idxs_buck = property(__get_idxs_buck, __set_idxs_buck)

    def __get_col_info(self) -> dict: return self.__col_info
    def __set_col_info(self, x: dict) -> None: self.__col_info = x
    col_info = property(__get_col_info, __set_col_info)

    def __get_no_transf_cols(self) -> int: return self.__no_transf_cols
    def __set_no_transf_cols(self, x: int) -> None: self.__no_transf_cols = x
    no_transf_cols = property(__get_no_transf_cols, __set_no_transf_cols)

    def __get_cat_dict(self) -> dict: return self.__cat_dict
    def __set_cat_dict(self, x: dict) -> None: self.__cat_dict = x
    cat_dict = property(__get_cat_dict, __set_cat_dict)

    def __get_team_translation_dict(self) -> dict: return self.__team_translation_dict
    def __set_team_translation_dict(self, x: dict) -> None: self.__team_translation_dict = x
    team_translation_dict = property(__get_team_translation_dict, __set_team_translation_dict)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def translate_teams(self, teams: list) -> list:
        """Some teams changed their name within the seasons even though the team stayed the same. Therefore, handle
        these teams equally under the same name."""

        return [self.team_translation_dict[team] if team in self.team_translation_dict else team for team in teams]

    # CONVERT CATEGORICAL FEATURES WITH TYPE STRING TO TYPE NUMERIC ----------------------------------------------------
    def fit_cat_dict(self, X_cat_str: list, featurename: str) -> None:
        # translate teams
        if featurename == 'team':
            X_cat_str = self.translate_teams(teams=X_cat_str)

        self.cat_dict[featurename] = {}

        for entry in X_cat_str:
            if entry not in self.cat_dict[featurename]:
                self.cat_dict[featurename][entry] = len(self.cat_dict[featurename]) + 1

    def transform_cat_dict(self, X_cat_str: list, featurename: str) -> list:
        # translate teams
        if featurename == 'team':
            X_cat_str = self.translate_teams(teams=X_cat_str)

        return [self.cat_dict[featurename][entry] for entry in X_cat_str]

    def fit_transform_cat_dict(self, X_cat_str: list, featurename: str) -> list:
        self.fit_cat_dict(X_cat_str=X_cat_str, featurename=featurename)
        return self.transform_cat_dict(X_cat_str=X_cat_str, featurename=featurename)

    # PREPROCESSING OF THE DATA (SCALING, ONE-HOT ENCODING ETC.) -------------------------------------------------------
    def fit(self, X: np.ndarray) -> None:
        # check if inserted number of columns fits
        if not len(self.idxs_num) + len(self.idxs_cat) + len(self.idxs_buck) == X.shape[1]:
            raise RuntimeError("Wrong number of features!")

        # handle numerical columns
        means = np.mean(X[:, self.idxs_num], axis=0)
        stddevs = np.std(X[:, self.idxs_num], axis=0)

        for idx_0, idx_num in enumerate(self.idxs_num):
            self.col_info[idx_num] = {'mean': means[idx_0],
                                      'stddev': stddevs[idx_0]}

        # handle categorical columns
        for idx in self.idxs_cat:
            classes = np.unique(X[:, idx])
            self.col_info[idx] = {'bins': classes + 0.5,  # assuming that categorical data is at least 1.0 different
                                  'no_classes': classes.size}

        # determine number of columns in output
        self.no_transf_cols = (len(self.idxs_num)
                               + sum([self.col_info[idx]['no_classes'] for idx in self.idxs_cat])
                               + sum([self.col_info[idx]['no_classes'] for idx in self.idxs_buck]))

    def transform(self, X: np.ndarray, dtype_out: type = np.float32) -> np.ndarray:
        # check if already fitted
        if self.no_transf_cols == 0:
            raise RuntimeError("Seems like the preprocessor was not fitted yet!")

        # extend dimension if X is a single entry
        if X.ndim == 1:
            X_ = np.expand_dims(X, axis=0)
        else:
            X_ = X

        # check if inserted number of columns fits
        if not len(self.idxs_num) + len(self.idxs_cat) + len(self.idxs_buck) == X_.shape[1]:
            raise RuntimeError("Wrong number of features!")

        # create output array
        out = np.zeros((X_.shape[0], self.no_transf_cols), dtype=dtype_out)

        # loop through raw columns and handle them accordingly
        conv_idx = 0

        for raw_idx in range(X_.shape[1]):
            # numerical columns
            if raw_idx in self.idxs_num:
                out[:, conv_idx] = (X_[:, raw_idx] - self.col_info[raw_idx]['mean']) / self.col_info[raw_idx]['stddev']
                conv_idx += 1

            # categorical + bucketized columns
            elif raw_idx in self.idxs_cat or raw_idx in self.idxs_buck:
                # we have to work with bins (categorical data because it does not necessarily start at 0, bucketized
                # data anyway)
                bin_idxs = np.digitize(X_[:, raw_idx], bins=self.col_info[raw_idx]['bins'], right=True)

                # one hot encoding
                out[:, conv_idx:conv_idx + self.col_info[raw_idx]['no_classes']] = \
                    np.eye(self.col_info[raw_idx]['no_classes'])[bin_idxs]
                conv_idx += self.col_info[raw_idx]['no_classes']

            else:
                raise RuntimeError("Unknown column!")

        # reduce dimension if X is a single entry
        if X.ndim == 1:
            out = np.squeeze(out)

        return out

    def fit_transform(self, X: np.ndarray, dtype_out: type = np.float32) -> np.ndarray:
        self.fit(X=X)
        return self.transform(X=X, dtype_out=dtype_out)

    def reverse(self, X_conv: np.ndarray) -> np.ndarray:
        """Reverses the transform method to obtain X again. Attention: Bucketized columns cannot be reversed completely,
        since some information was lost during the transformation process!"""

        # check if already fitted
        if self.no_transf_cols == 0:
            raise RuntimeError("Seems like the preprocessor was not fitted yet!")

        # extend dimension if X_conv is a single entry
        if X_conv.ndim == 1:
            X_conv_ = np.expand_dims(X_conv, axis=0)
        else:
            X_conv_ = X_conv

        # check if inserted number of columns fits
        if not self.no_transf_cols == X_conv_.shape[1]:
            raise RuntimeError("Wrong number of columns in X_conv!")

        # create output array
        out = np.zeros((X_conv_.shape[0], len(self.col_info)))

        # loop through raw columns and handle them accordingly
        conv_idx = 0

        for raw_idx in range(out.shape[1]):
            # numerical columns
            if raw_idx in self.idxs_num:
                out[:, raw_idx] = (X_conv_[:, conv_idx] * self.col_info[raw_idx]['stddev']
                                   + self.col_info[raw_idx]['mean'])
                conv_idx += 1

            # categorical + bucketized columns
            elif raw_idx in self.idxs_cat or raw_idx in self.idxs_buck:
                # get index of True value in every row of the one-hot encoded feature
                idx_true = \
                    np.nonzero(X_conv_[:, conv_idx:conv_idx + self.col_info[raw_idx]['no_classes']].astype(np.bool))[1]

                # in the case of bucketized data the last bin entry is inf -> use previous entry + 1 instead
                if np.isinf(self.col_info[raw_idx]['bins'][-1]):
                    bins_tmp = np.copy(self.col_info[raw_idx]['bins'])
                    bins_tmp[-1] = bins_tmp[-2] + 1.0
                else:
                    bins_tmp = self.col_info[raw_idx]['bins']

                # store bin value (-0.5 because 0.5 was added when the bins were separated)
                out[:, raw_idx] = bins_tmp[idx_true] - 0.5
                conv_idx += self.col_info[raw_idx]['no_classes']

            else:
                raise RuntimeError("Unknown column!")

        # reduce dimension if X_conv is a single entry
        if X_conv.ndim == 1:
            out = np.squeeze(out)

        return out


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
