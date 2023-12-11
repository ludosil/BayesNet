from enum import Enum
from scipy.stats import bernoulli
from dataclasses import dataclass
from typing import Callable


class Distribution(Enum):
    BERNOULLI = 1


distribution_to_ppf = {Distribution.BERNOULLI: bernoulli}

distribution_to_possible_values = {Distribution.BERNOULLI: [0, 1]}


@dataclass
class ConditionalDistribution:
    """Class for representing a conditional distribution."""

    distribution: Distribution
    parameters: dict
    conditionals: [tuple]
    ppf: Callable
