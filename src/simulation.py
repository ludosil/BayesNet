from src.bayesnet import BayesNet
from src.distribution import ConditionalDistribution
import pandas as pd
import numpy as np
from copy import deepcopy


def generate_simulation_order(bn: BayesNet):
    """
    use three lists - indepedent nodes, conditional nodes, order
    order is initially empty while conditional and independent nodes are initialised from network values
    while independent nodes not empty:
        pop off node and append to order - return if no conditionals left
        remove node from any parent mapping
        evaluate whether any conditional nodes are now (conditionally) independent
        append these to independent nodes
    when setting the original lists, apply sort to allow reproducibility
    print statements useful for debugging - keep for the moment
    """
    independent_nodes = list(bn.independent_nodes)
    conditional_nodes = list(bn.all_nodes - bn.independent_nodes)
    parents = deepcopy(bn.parents)
    node_order = []
    while independent_nodes:
        # print(f"{independent_nodes}:{conditional_nodes}:{node_order}")
        node = independent_nodes.pop()
        # print(f"popped off {node}")
        node_order.append(node)
        # print(f"{independent_nodes}:{conditional_nodes}:{node_order}")

        # print(f"can we remove parent {node} from {parents}?")
        for c, ps in parents.items():
            if node in ps:
                ps.remove(node)
        # print(f"post-update parents is {parents}")

        # print(f"are there any newly (conditionally) independent nodes?")
        conditionally_independent_nodes = sorted(
            [c for c, ps in parents.items() if ps == []]
        )
        for c in conditionally_independent_nodes:
            parents.pop(c)
            conditional_nodes.remove(c)
        # print(f"{conditionally_independent_nodes}\n")
        if conditionally_independent_nodes:
            independent_nodes = independent_nodes + conditionally_independent_nodes

    return node_order


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
