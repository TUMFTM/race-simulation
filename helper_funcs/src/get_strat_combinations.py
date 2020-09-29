import itertools


def get_strat_combinations(available_compounds: list,
                           min_no_pitstops: int = 1,
                           max_no_pitstops: int = 3,
                           enforce_diff_compounds: bool = True,
                           start_compound: str = None,
                           all_orders: bool = False) -> dict:
    """
    author:
    Alexander Heilmeier

    date:
    27.07.2020

    .. description::
    This function creates all possible combinations of tire compounds considering the inserted boundary conditions.

    .. inputs::
    :param available_compounds:     Available compounds that should be included in the combinations, e.g. ['A4', 'A5']
    :type available_compounds:      list
    :param min_no_pitstops:         Minimum number of pit stops
    :type min_no_pitstops:          int
    :param max_no_pitstops:         Maximum number of pit stops
    :type max_no_pitstops:          int
    :param enforce_diff_compounds:  Boolean flag to determine of combinations with a single compound should be deleted
    :type enforce_diff_compounds:   bool
    :param start_compound:          The start compound can be inserted such that it is included for the first stint
    :type start_compound:           str
    :param all_orders:              Boolean flag to determine if the function returns combinations (order of sequences
                                    does not matter) or the product (all possible sequences orders included)
    :type all_orders:               bool

    .. outputs::
    :return strategy_combinations:  Dict with possible strategy combinations for every number of stops, e.g.
                                    {1: [('A4', 'A5')], 2: [('A4', 'A5', 'A5'), ('A4', 'A4', 'A5')], ...}
    :rtype strategy_combinations:   dict
    """

    strategy_combinations = {}

    for cur_no_pitstops in range(min_no_pitstops, max_no_pitstops + 1):
        # combinations used since the chosen order of tire compounds does not matter for the final race time
        if not all_orders:
            strategy_combinations[cur_no_pitstops] = \
                list(itertools.combinations_with_replacement(available_compounds, r=cur_no_pitstops + 1))
        else:
            strategy_combinations[cur_no_pitstops] = \
                list(itertools.product(available_compounds, repeat=cur_no_pitstops + 1))

        # remove strategy combinations using only a single tire compound if enforced
        if enforce_diff_compounds:
            strategy_combinations[cur_no_pitstops] = [strat_tmp for strat_tmp in strategy_combinations[cur_no_pitstops]
                                                      if not len(set(strat_tmp)) == 1]

        # remove strategy combinations that do not include the starting tire compound if enforced and sort compounds
        # such that the combinations start with the starting compound, although this does not influence the total
        # race time (for plotting purposes)
        if start_compound:
            strategy_combinations[cur_no_pitstops] = [strat_tmp for strat_tmp in strategy_combinations[cur_no_pitstops]
                                                      if start_compound in strat_tmp]

            for idx_set in range(len(strategy_combinations[cur_no_pitstops])):
                # switch positions of first start compound entry and first entry if not equal
                if not strategy_combinations[cur_no_pitstops][idx_set][0] == start_compound:
                    set_list = list(strategy_combinations[cur_no_pitstops][idx_set])
                    idx_start_compound = set_list.index(start_compound)
                    set_list[0], set_list[idx_start_compound] = set_list[idx_start_compound], set_list[0]
                    strategy_combinations[cur_no_pitstops][idx_set] = tuple(set_list)

    return strategy_combinations
