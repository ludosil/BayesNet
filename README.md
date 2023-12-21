# BayesNet

This is a project to implement Bayesian networks - Bayes nets - in Python. Bayes nets are a type of probabilistic graphical model, or PGM. It is a simple and intuitive way to represent a joint probability distribution by specifying conditional probability distributions for the underlying variables in the network.

Hope you enjoy!



## Structure of repo

- requirements.txt
- src/: Python code
- notebooks/BayesNet.ipnb: Jupyter notebook
- scripts/: stand-alone scripts to run model
- tests/: unit tests for model
- excel/: model for alarm network
- data/: example data, including the alarm and medical diagnosis networks



## Getting started

Create a virtual environment using the requirements.txt file.

The [Jupyter notebook](notebooks/BayesNet.ipynb) is a good place to start. This sets out the modelling from scratch and gives insight into the design and implementation of the underlying code. This gist  [python-requirements.txt (github.com)](https://gist.github.com/alexmirrington/801971108e79456b1e7b13079abe19d9) is helpful to enable the notebook to be run on a virtual environment.

The [Excel model](excel/bayes_net_alarm.xlsx) sets out ideas on the Bayes net simulation, including the handling of random numbers, and the simulation of independent vs. conditional variables. It may be easier to look through this ahead of - or in conjunction with - the same content in the notebook.

The [scripts](scripts/) folder houses stand-alone scripts for the user to run the model. There are two main scripts: one runs from command-line arguments, the other via a config file, presently in [scripts/config.yaml](scripts/config.yaml). The six arguments - in order - are:

1. filepath to summary CSV
2. filepath to conditional distributions CSV
3. random seed
4. number of sims
5. filepath to random variates output CSV
6. filepath to P(X|Y) output CSV

The [tests](tests/) folder contains unit tests. Run these with 'python -m pytest tests/'.



## Data

The model generates a Bayes net from two CSV files, one that represents a summary view and one that sets out the conditional distributions. There are currently three examples in [data/](data/):

1. The alarm network - a well known example used to teach Bayes nets: https://github.com/jpmcarrilho/AIMA/blob/master/probability-4e.ipynb
2. The alarm network implemented via a more generic categorical representation. (1) is implemented as a Bernoulli (binary) variable network
3. A medical diagnosis network, taken from: Santos, Eugene & Shimony, S.E.. (1998). Deterministic approximation of marginal probabilities in Bayes nets. Systems, Man and Cybernetics, Part A: Systems and Humans, IEEE Transactions on. 28. 377 - 393. 10.1109/3468.686701. __Please note - probabilities are indicative only - ad-hoc estimation from Google search__

Both these examples can be run using the scripts above. Try creating a simple network yourself; simple Bayesian problems like Monty Hall and did-I-flip-the-two-sided-coin can be modelled as two-node problems.



## Extending the code

The string that keeps on getting longer....various items I would like to develop as follows.

Tidy-up

- Use of 'None', None, [] in the code and data. Maybe a few inconsistencies to iron out
- Use one of conditional probabilities vs. conditional distributions in naming conventions, not both
- Extend unit tests for further coverage
- Document the input format required
- Add regex checks prior to parsing of input files
- Where useful, add more informative error message e.g. which conditionals are missing in actual vs. expected check

~~Distributions~~

- ~~The code in [src/distribution.py](src/distribution.py) is a little scattered - an enumeration, two dictionaries, and a dataclass; this can be refactored into a single class that encapsulates all the functionality required. This should enable a generalised categorical distribution~~ - complete

~~Binary variables -> categorical variables~~

- ~~Linked to the distribution development is the one to generalise the model to handle categorical representations~~ - complete

User functionality to generate skeleton conditional distribution files

- The Bayes net is initialised from two files, one to represent the summary view and one for the conditional distributions. The former is compact while the latter is more detailed, with one row for every possible distribution. As the Bayes net grows in size, it gets more difficult, error-prone and tedious to set this out by hand. Given the summary file, we can generate a conditional distributions input file that is valid; the only item for the user to complete would be the probabilities

Bigger networks

- More nodes challenges the ability of the model to simulate the network and, practically, to create its representation in data. Toy models are fun, but real-world applications require a model and process that scales

