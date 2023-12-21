from enum import Enum
from scipy.stats import bernoulli, rv_discrete
from dataclasses import dataclass
from typing import Callable
import numpy as np


class Distribution(Enum):
    BERNOULLI = 1
    CATEGORICAL = 2


class ConditionalDistribution:
    """Class for representing a conditional distribution."""

    distribution: Distribution
    parameters: dict
    conditionals: [tuple]
    ppf: Callable
    possible_values: list

    def __init__(self, distribution, parameters, conditionals):
        self.distribution = distribution
        self.parameters = parameters
        self.conditionals = conditionals

        if distribution == Distribution.BERNOULLI:
            # check only one parameter, p
            assert set(parameters.keys()) == {"p"}
            self.ppf = lambda x: bernoulli.ppf(x, **parameters).astype(int)
            self.possible_values = [0, 1]
        elif distribution == Distribution.CATEGORICAL:
            # extract into categories, categories as integers and probabilities
            cats = list(parameters.keys())
            cats_int = range(len(cats))
            probs = list(parameters.values())
            # check probabilities sum to 1 and are individually 0-1
            assert sum(probs) == 1
            assert all([p >= 0 and p < 1 for p in probs])
            # use internal function for integer distribution via rv_discrete
            # when call to ppf is made, use interal distribution's ppf then map back
            # note handling of iterable vs. non-iterable
            cats_dict = dict(zip(cats_int, cats))
            self._dist_int = rv_discrete(values=(cats_int, probs))

            def ppf(x):
                try:
                    iterator = iter(x)
                except TypeError:
                    return cats_dict[self._dist_int.ppf(x)]
                else:
                    return np.array(
                        [cats_dict[rv_int] for rv_int in self._dist_int.ppf(x)]
                    )

            self.ppf = ppf
            self.possible_values = cats

        else:
            raise AssertionError(
                f"Invalid distribution selected. Valid choices are: {[d.name for d in Distribution]}"
            )
