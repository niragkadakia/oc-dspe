# oc-dspe

Package to estimate nonlinear biophysical neural models from measured voltage traces using an algorithm combining optimal control theory and nonlinear optimization, "optimally-controlled dynamical state and parameter estimation."



## Installation

`OC-DSPE` is a standalone package in Python 2. The code has been tested on Python 2.7.11. The code requires the following packages:

1. Python 2 (tested on 2.7.9, probably will work on anything ≥ 2.7).
2. NumPy (tested on ≥ 1.12.0).
3. SciPy (tested on ≥ 0.18.1).
4. The variational annealing package [varanneal](https://github.com/niragkadakia/varanneal), which requires [PyAdolc](https://github.com/b45ch1/pyadolc)



## Data generation

To generate synthetic data as in the paper, use the `*gen_data*.py`scripts in the `scripts/` folder. These will generate synthetic datasets in the working folder, which you should move to a separate data folder. 

## Estimation

The estimation scripts using OC-DSPE are `lorenz_adj.py` for the Lorenz96 model, and `morris_lecar_adj.py` for the Morris-Lecar neuron model. There are many other related scripts that generate estimations with other conditions (extra channel in model, process noise, estimation all parameters versus just conductances). 

There are also related estimation scripts for using constrained least squares (named `*4dvar.py` ) and using the original dynamical state and parameter estimation routine `*dspe.py` . 
