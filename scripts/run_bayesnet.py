import sys

sys.path.insert(0, ".")


import pandas as pd
import numpy as np
from src.bayesnet import BayesNet, visualise_bn
from src.simulation import generate_simulation_order, simulate_bayes_net

if __name__ == "__main__":
    if len(sys.argv[1:]) != 4:
        raise AssertionError(
            "Expected four arguments: \n \
                             1. filepath to summary CSV\n \
                             2. filepath to conditional distributions CSV\n \
                             3. random seed\n \
                             4. number of sims"
        )
    else:
        summary_fp = sys.argv[1]
        conditional_distributions_fp = sys.argv[2]
        random_seed = int(sys.argv[3])
        num_sims = int(sys.argv[4])

    with open(summary_fp, "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)
    with open(conditional_distributions_fp, "r") as f:
        conditional_distributions_df = pd.read_csv(f, na_filter=False)

    bn = BayesNet(summary_df, conditional_distributions_df)
    # visualise_bn(bn, use_full_names=True)
    simulation_order = generate_simulation_order(bn.all_nodes, bn.parents)
    rng = np.random.default_rng(random_seed)
    random_numbers = pd.DataFrame(
        {var: rng.random(num_sims) for var in simulation_order}
    )
    random_variates = simulate_bayes_net(
        simulation_order, bn.conditional_distributions, random_numbers
    )

    print(f"summary from {num_sims} simulations:\n{random_variates.sum()}")
