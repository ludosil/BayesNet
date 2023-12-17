import pandas as pd
import numpy as np
from src.bayesnet import BayesNet
from src.simulation import generate_simulation_order, simulate_bayes_net
from src.estimate import generate_p_x_y_df
import pytest

NUM_SIMS = 1000
RANDOM_SEED = 0


@pytest.fixture
def bn_medical_diagnosis_bernoulli():
    with open(f"data/summary_medical_diagnosis.csv", "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)

    with open(f"data/conditional_probabilities_medical_diagnosis.csv", "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    return bn


@pytest.fixture
def bn_alarm_bernoulli():
    with open(f"data/summary_alarm.csv", "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)

    with open(f"data/conditional_probabilities_alarm.csv", "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    return bn


@pytest.fixture
def bn_alarm_categorical():
    with open(f"data/summary_alarm_categorical.csv", "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)

    with open(f"data/conditional_probabilities_alarm_categorical.csv", "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    return bn


def test_e2e(bn_medical_diagnosis_bernoulli, bn_alarm_bernoulli, bn_alarm_categorical):
    """Run test examples E2E and check P(X|X) == 1."""
    num_sims = NUM_SIMS
    random_seed = RANDOM_SEED
    for bn in [
        bn_medical_diagnosis_bernoulli,
        bn_alarm_bernoulli,
        bn_alarm_categorical,
    ]:
        simulation_order = generate_simulation_order(bn)
        rng = np.random.default_rng(random_seed)
        random_numbers = pd.DataFrame(
            {var: rng.random(num_sims) for var in simulation_order}
        )
        random_variates = simulate_bayes_net(
            simulation_order, bn.conditional_distributions, random_numbers
        )
        pxy_df = generate_p_x_y_df(random_variates, bn)

    # check all(PX|X) counts are identical
    for i, row in pxy_df.iterrows():
        event_x, event_y, count_x_and_y, count_y, _ = row
        if event_x == event_y:
            assert count_x_and_y == count_y


def test_generate_simulation_order_alarm(bn_alarm_bernoulli):
    """Algorithm is deterministic given sorting of lists."""
    bn = bn_alarm_bernoulli
    expected_simulation_order = ["E", "B", "A", "M", "J"]
    actual_simulation_order = generate_simulation_order(bn)
    assert actual_simulation_order == expected_simulation_order


def test_generate_simulation_order_medical_diagnosis(bn_medical_diagnosis_bernoulli):
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
    bn = bn_medical_diagnosis_bernoulli
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


def test_equivalence_of_bernoulli_and_categorical_alarm_networks(
    bn_alarm_bernoulli, bn_alarm_categorical
):
    """
    Test equivalence of applying Bernoulli vs. categorical variables for the alarm network.
    Note that - to align ppf functions - random numbers are taken as rn and 1-rn, respectively.
    This is due to the convention we have applied when implementing the categorical distributions.
    """
    pxy_df = []
    for i, bn in enumerate([bn_alarm_bernoulli, bn_alarm_categorical]):
        num_sims = NUM_SIMS
        random_seed = RANDOM_SEED
        simulation_order = generate_simulation_order(bn)
        rng = np.random.default_rng(random_seed)
        random_numbers = pd.DataFrame(
            {var: rng.random(num_sims) for var in simulation_order}
        )
        if i == 1:
            random_numbers = 1 - random_numbers
        random_variates = simulate_bayes_net(
            simulation_order, bn.conditional_distributions, random_numbers
        )
        pxy_df.append(generate_p_x_y_df(random_variates, bn))

    pxy_df_bernoulli = pxy_df[0]
    pxy_df_categorical = pxy_df[1]

    # iterate through the Bernoulli output, map to the categorical equivalent, and assert rows are equal
    name_mapping = {
        "B": {"Burglary": 1, "NoBurglary": 0},
        "E": {"Earthquake": 1, "NoEarthquake": 0},
        "A": {"Alarm": 1, "NoAlarm": 0},
        "J": {"JohnCalls": 1, "JohnDoesNotCall": 0},
        "M": {"MaryCalls": 1, "MaryDoesNotCall": 0},
    }

    # iterate through name and value combinations
    # end goal is to compare like for like
    # e.g. row output for 'P(B=1|M=1)' vs. 'P(B=Burglary|M=MaryCalls)'
    for x in name_mapping:
        for y in name_mapping:
            for xval_cat in name_mapping[x]:
                xval_bern = name_mapping[x][xval_cat]
                for yval_cat in name_mapping[y]:
                    yval_bern = name_mapping[y][yval_cat]
                    row_cat = f"P({x}={xval_cat}|{y}={yval_cat})"
                    row_bern = f"P({x}={xval_bern}|{y}={yval_bern})"
                    assert all(
                        pxy_df_bernoulli.loc[row_bern, ["Count X & Y", "Count Y"]]
                        == pxy_df_categorical.loc[row_cat, ["Count X & Y", "Count Y"]]
                    )
