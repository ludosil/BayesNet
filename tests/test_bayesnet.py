import pandas as pd
import numpy as np
from src.bayesnet import BayesNet
from src.simulation import generate_simulation_order, simulate_bayes_net
from src.estimate import generate_p_x_y_df_binary_network
import pytest


def test_e2e():
    """Run test examples E2E and check P(X|X) == 1."""
    num_sims = 100000
    random_seed = 0
    for label in ["alarm", "medical_diagnosis"]:
        with open(f"data/summary_{label}.csv", "r") as f:
            summary_df = pd.read_csv(f, na_filter=False)

        with open(f"data/conditional_probabilities_{label}.csv", "r") as f:
            conditional_distributions_df = pd.read_csv(f, na_filter=False)

        bn = BayesNet(summary_df, conditional_distributions_df)
        simulation_order = generate_simulation_order(bn)
        rng = np.random.default_rng(random_seed)
        random_numbers = pd.DataFrame(
            {var: rng.random(num_sims) for var in simulation_order}
        )
        random_variates = simulate_bayes_net(
            simulation_order, bn.conditional_distributions, random_numbers
        )
        pxy_df = generate_p_x_y_df_binary_network(random_variates, bn.names)

        # check all(PX|X) counts are identical
    for i, row in pxy_df.iterrows():
        event_x, event_y, count_x_and_y, count_y, _ = row
        if event_x == event_y:
            assert count_x_and_y == count_y


def test_generate_simulation_order_alarm():
    """Algorithm is deterministic given sorting of lists."""
    with open(f"data/summary_alarm.csv", "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)

    with open(f"data/conditional_probabilities_alarm.csv", "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    expected_simulation_order = ["E", "B", "A", "M", "J"]
    actual_simulation_order = generate_simulation_order(bn)
    assert actual_simulation_order == expected_simulation_order


def test_generate_simulation_order_medical_diagnosis():
    """Algorithm is deterministic given sorting of lists.
    [ind nodes] - [cond node]           # node removed
    [A,HNP,SM]    [B,DS,HC,HT,LC,SC,SN] # SM
    [A,HBP,B,LC]  [DS,HC,HT,SC,SN]      # LC
    [A,HBP,B]     [DS,HC,HT,SC,SN]      # B
    [A,HBP,HT,SC] [DS,HC,SN]            # SC
    [A,HBP,HT]    [DS,HC,SN]            # HT
    [A,HBP]       [DS,HC,SN]            # HBP
    [A,HC]        [DS,SN]               # HC
    [A,DS]        [SN]                  # DS
    [A]           [SN]                  # A
    [SN]          []                    # SN
    """
    with open(f"data/summary_medical_diagnosis.csv", "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)

    with open(f"data/conditional_probabilities_medical_diagnosis.csv", "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    expected_simulation_order = [
        "SM",
        "LC",
        "B",
        "SC",
        "HT",
        "HBP",
        "HC",
        "DS",
        "A",
        "SN",
    ]
    actual_simulation_order = generate_simulation_order(bn)
    assert actual_simulation_order == expected_simulation_order
