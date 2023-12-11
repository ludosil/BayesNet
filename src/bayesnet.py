from src.distribution import (
    Distribution,
    distribution_to_ppf,
    distribution_to_possible_values,
    ConditionalDistribution,
)
import graphviz
from itertools import product
import pandas as pd


class BayesNet:
    """Class for representing a Bayes net."""

    names: dict
    distributions: dict
    dependencies: dict
    parents: dict
    conditional_distributions: dict
    all_nodes: set
    independent_nodes: set
    leaf_nodes: set

    def __init__(
        self, summary_df: pd.DataFrame, conditional_probabilities_df: pd.DataFrame
    ):
        """
        Initialise summary info via parse function.
        Initialise conditional distributions via parse function.
        Check the network is self-consistent.
        """
        (
            self.all_nodes,
            self.names,
            self.distributions,
            self.dependencies,
            self.parents,
            self.independent_nodes,
            self.leaf_nodes,
        ) = parse_summary(summary_df)
        self.conditional_distributions = parse_conditional_distributions(
            conditional_probabilities_df
        )
        check_self_consistency(
            self.all_nodes,
            self.names,
            self.distributions,
            self.dependencies,
            self.parents,
            self.independent_nodes,
            self.leaf_nodes,
            self.conditional_distributions,
        )


def parse_summary(summary_df: pd.DataFrame):
    """Parse summary info for Bayes net."""

    # assert nodes unique and get all nodes
    assert summary_df["node"].is_unique
    all_nodes = set(summary_df["node"])

    # get names
    names = dict(zip(summary_df["node"], summary_df["name"]))

    # get distributions
    # output meaningful error message if this fails
    try:
        distributions_enumd = [
            getattr(Distribution, d) for d in summary_df["distribution"]
        ]
        distributions = dict(zip(summary_df["node"], distributions_enumd))
    except:
        raise AssertionError(
            f"Invalid distribution selected. Valid choices are: {[d.name for d in Distribution]}"
        )

    # get dependencies
    dependencies = {}
    for i, row in summary_df.iterrows():
        if row.dependencies == "None":
            dependencies[row.node] = []
        else:
            dependencies[row.node] = row.dependencies.split(";")
    dependencies = {c: p for c, p in dependencies.items() if p != []}

    # extract parents from dependencies
    parents = {node: [] for node in all_nodes}
    for c, ps in dependencies.items():
        if ps is not None:
            for p in ps:
                parents[p].append(c)
    parents = {p: c for p, c in parents.items() if c != []}

    # get independent and leaf nodes as implied from parents and dependencies
    independent_nodes = {node for node in all_nodes if node not in parents}
    leaf_nodes = {node for node in all_nodes if node not in dependencies}

    return (
        all_nodes,
        names,
        distributions,
        dependencies,
        parents,
        independent_nodes,
        leaf_nodes,
    )


def parse_conditional_distributions(conditional_probabilities_df: pd.DataFrame):
    """Parse each row into conditional distribution and return as dictionary mapping node to list of CDs."""

    unique_nodes = conditional_probabilities_df["node"].unique()
    conditional_distributions_from_df = {node: [] for node in unique_nodes}

    for _, row in conditional_probabilities_df.iterrows():
        conditional_distributions_from_df[row.node].append(
            parse_df_row_into_conditional_distribution(row)
        )

    return conditional_distributions_from_df


def parse_df_row_into_conditional_distribution(df_row):
    # get distribution
    # output meaningful error message if this fails
    try:
        distribution = getattr(Distribution, df_row.distribution)
    except:
        raise AssertionError(
            f"Invalid distribution selected. Valid choices are: {[d.name for d in Distribution]}"
        )

    # get parameter-value pairs
    # convert parameter value to float and return as dictionary
    parameters = [tuple(pv.split("=")) for pv in df_row.parameters.split(";")]
    parameters = {p: float(v) for p, v in parameters}

    # create a ppf using the supplied parameter(s)
    # this should pass/fail depending on whether the values are consistent with what scipy.stats expects
    # output a meaningful error message if this fails
    sci_py_distribution = distribution_to_ppf[distribution]

    def ppf(x):
        return (sci_py_distribution.ppf(x, **parameters)).astype(int)

    # get conditionals - either None or list of tuples which can be parsed as we did the parameters
    # convert '1' or '0' if supplied as conditional values - relevant for binary variable network
    if df_row.conditionals == "None":
        conditionals = None
    else:
        conditionals = [tuple(pv.split("=")) for pv in df_row.conditionals.split(";")]
        conditionals = [
            (n, int(v)) if v in ["0", "1"] else (n, v) for n, v in conditionals
        ]

    return ConditionalDistribution(
        distribution=distribution,
        parameters=parameters,
        conditionals=conditionals,
        ppf=ppf,
    )


def check_self_consistency(
    all_nodes,
    names,
    distributions,
    dependencies,
    parents,
    independent_nodes,
    leaf_nodes,
    conditional_distributions,
):
    assert set(names.keys()) == set(distributions.keys())
    assert set(names.keys()) == all_nodes

    for node in independent_nodes:
        assert len(conditional_distributions[node]) == 1
        assert conditional_distributions[node][0].conditionals is None

    for node in all_nodes - independent_nodes:
        # map each parent to a list  of tuples, then apply itertools.product to generate all the combinations
        # e.g. for A, we would have two lists
        # B -> [(B,0), (B,1)]
        # E -> [(E,0), (E,1)]
        parent_possible_values = [
            [(p, v) for v in distribution_to_possible_values[distributions[p]]]
            for p in parents[node]
        ]
        expected_conditionals = [list(x) for x in product(*parent_possible_values)]
        actual_conditionals = [
            cd.conditionals for cd in conditional_distributions[node]
        ]
        assert sorted(actual_conditionals) == sorted(expected_conditionals)

    for node in all_nodes:
        summary_distribution = distributions[node]
        assert all(
            [cd.distribution == summary_distribution]
            for cd in conditional_distributions[node]
        )


def visualise_bn(bn: BayesNet, use_full_names: bool):
    """Visualises Bayes net object."""
    g = graphviz.Digraph()
    for node, next_nodes in bn.dependencies.items():
        for nn in next_nodes:
            if use_full_names:
                g.edge(bn.names[node], bn.names[nn])
            else:
                g.edge(node, nn)
    return g
