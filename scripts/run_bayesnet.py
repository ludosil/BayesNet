import sys

sys.path.insert(0, ".")

import os
import pandas as pd
import numpy as np
from src.bayesnet import BayesNet
from src.simulation import generate_simulation_order, simulate_bayes_net
from src.estimate import generate_p_x_y_df

if __name__ == "__main__":
    if len(sys.argv[1:]) != 6:
        raise AssertionError(
            "Expected six arguments: \n \
                             1. filepath to summary CSV\n \
                             2. filepath to conditional distributions CSV\n \
                             3. random seed\n \
                             4. number of sims\n \
                             5. filepath to random variates output CSV\n \
                             6. filepath to P(X|Y) output CSV"
        )
    else:
        summary_fp = sys.argv[1]
        conditional_distributions_fp = sys.argv[2]
        random_seed = int(sys.argv[3])
        num_sims = int(sys.argv[4])
        output_fp_rv = sys.argv[5]
        output_fp_pxy = sys.argv[6]

    with open(summary_fp, "r") as f:
        summary_df = pd.read_csv(f, na_filter=False)
    with open(conditional_distributions_fp, "r") as f:
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
    pxy_df = generate_p_x_y_df(random_variates, bn.names)

    # write random variates
    if os.path.exists(output_fp_rv):
        print(f"{output_fp_rv} exists - not overwritten")
    else:
        random_variates.to_csv(output_fp_rv)
        print(f"written output to {output_fp_rv}")

    # write pxy
    if os.path.exists(output_fp_pxy):
        print(f"{output_fp_pxy} exists - not overwritten")
    else:
        pxy_df.to_csv(output_fp_pxy)
        print(f"written output to {output_fp_pxy}")
