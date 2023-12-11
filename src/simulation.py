from src.bayesnet import BayesNet
from src.distribution import ConditionalDistribution
import pandas as pd
import numpy as np


def generate_simulation_order(all_nodes: set, parents_dict: dict):
    """Generates valid simulation order for Bayes net given all nodes and parent mappings."""
    nodes_to_add = all_nodes.copy()
    num_nodes = len(nodes_to_add)
    nodes_to_simulate = []

    while nodes_to_add:
        node = nodes_to_add.pop()
        parents = parents_dict.get(node, None)

        # insert one beyond last parent, using -1 as index for no parents (or no parents in list)
        if parents is not None:
            parents_in_list = [
                nodes_to_simulate.index(p) if p in nodes_to_simulate else -1
                for p in parents
            ]
            max_parent = max(parents_in_list)
        else:
            max_parent = -1

        nodes_to_simulate.insert(max_parent + 1, node)

    return nodes_to_simulate


def simulate_bayes_net(
    simulation_order: list,
    conditional_distributions: [ConditionalDistribution],
    random_numbers: np.ndarray,
):
    """Simulates Bayes net to produce data-frame of random variates."""
    random_variates = pd.DataFrame()
    for var in simulation_order:
        if len(conditional_distributions[var]) == 1:
            # no condtionals, append to existing output
            random_variates = pd.concat(
                [
                    random_variates,
                    pd.DataFrame(
                        {
                            var: conditional_distributions[var][0].ppf(
                                random_numbers[var]
                            )
                        }
                    ),
                ],
                axis=1,
            )
        else:
            conditional_random_variates = pd.concat(
                [
                    pd.DataFrame({str(cd.conditionals): cd.ppf(random_numbers[var])})
                    for cd in conditional_distributions[var]
                ],
                axis=1,
            )

            # try:
            actual_events = pd.concat(
                [
                    pd.DataFrame(
                        {
                            str(cd.conditionals): generate_actual_events(
                                cd.conditionals, random_variates
                            )
                        }
                    )
                    for cd in conditional_distributions[var]
                ],
                axis=1,
            )

            random_variates = pd.concat(
                [
                    random_variates,
                    pd.DataFrame(
                        {var: (conditional_random_variates * actual_events).sum(axis=1)}
                    ),
                ],
                axis=1,
            )

    return random_variates


def generate_actual_events(
    conditionals: [tuple], random_variates: pd.DataFrame
) -> np.array:
    """
    Given a list of conditional (variable, value) pairs and a dataframe of random variates (1s and 0s),
    this function generates flags for whether the conditional is met
    e.g. given [('B', 0),('E', 0)], the return value is 1 where B and E are 0, and 0 elsewhere.

    Return Numpy array of 1s and 0s.
    """
    number_conditionals = len(conditionals)
    number_variates = random_variates.shape[0]
    outcomes = np.zeros((number_variates, number_conditionals)).astype(int)
    for i, (variable, value) in enumerate(conditionals):
        outcomes[:, i] = (random_variates[variable] == value) * 1

    return np.product(outcomes, axis=1)
