import sys

sys.path.insert(0, ".")

import os
import pandas as pd
import numpy as np
from yaml import safe_load
from src.bayesnet import BayesNet
from src.simulation import generate_simulation_order, simulate_bayes_net
from src.estimate import generate_p_x_y_df

CONFIG_YAML = "scripts/config.yaml"

if __name__ == "__main__":
    with open(CONFIG_YAML, "r") as f:
        x = safe_load(f)
    summary_fp = x["summary_fp"]
    conditional_distributions_fp = x["conditional_distributions_fp"]
    random_seed = x["random_seed"]
    num_sims = x["num_sims"]
    output_fp_rv = x["output_fp_rv"]
    output_fp_pxy = x["output_fp_pxy"]

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
    pxy_df = generate_p_x_y_df(random_variates, bn)

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
