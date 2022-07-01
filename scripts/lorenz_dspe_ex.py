"""
Dynamical state and parameter estimation with annealing, using 
automatic differentiation to code the derivatives. 

Created by Nirag Kadakia at 09:49 03-09-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import sys
from varanneal import va_ode
import sys, time

data_dir = '../../../data/oc-dspe'

seed = int(sys.argv[1])
skip = int(sys.argv[2])
u_range = float(sys.argv[3])
anneal_bin = int(sys.argv[4])

# Lorenz96 model dimension x2 for adjoint variables
D = 10

# Lorenz model in x,p of optimal control dynamical model
def l96(t, x, (p, stim)):

    gamma = p[:, 0]
    u = p[:, 1:]
    state_vec = np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x
    vec = (state_vec.T + gamma.T).T
    vec[:, ::skip] += (u.T*(stim - x[:, ::skip]).T).T

    return vec

Lidx = np.arange(0, D, skip)
RM = 1.0 / (0.5**2)
RF0 = 4.0e-4
alpha = 2
if anneal_bin == 0:
	beta_array = np.linspace(23, 23, 1)
elif anneal_bin == 1:
	beta_array = np.linspace(0, 23, 24)


data = np.load('%s/meas_data/dspe_ex_obs.npy' % data_dir)
times_data = data[:, 0]
dt_data = times_data[1] - times_data[0]
N_data = len(times_data)
data = data[:, 1:]
data = data[:, Lidx]
dt_model = dt_data
dt_model = dt_data
N_model = N_data

# Stimulus is measurement drive
stim =	np.load('%s/meas_data/dspe_ex_obs.npy' % data_dir)
stim = stim[:, 1:]
stim = stim[:, ::skip]

# Parameters now include time-dependent nudging terms; state space is D
nMeas = int(D - 1)//skip + 1
Pidx = [0]
Uidx = range(1, 1 + nMeas)
state_bounds = [[-15, 15]]*D
param_bounds =  [[1, 20]] + [[0, u_range]]*nMeas
bounds = state_bounds + param_bounds

# Initial conditions; initial forcing params set randomly to bounds
np.random.seed(seed)
X0 = (30.0*np.random.rand(N_model * D) - 15.0).reshape((N_model, D))
np.random.seed(seed)
P0 = [np.array([np.random.uniform(1, 20)])] + \
		[np.random.uniform(0, u_range, N_model)]*nMeas

# Run the annealing using L-BFGS-B
tstart = time.time()
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 
				'maxiter':1000000}
anneal1 = va_ode.Annealer()
anneal1.set_model(l96, D)
anneal1.set_data(data, t=times_data, stim=stim)
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, Uidx, 
				action='A_gaussian_quad_control', dt_model=dt_model, 
				init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B', 
				opt_args=BFGS_options, bounds=bounds, adolcID=0)

# Save the results
print (anneal1.P)
if anneal_bin == 0:
	anneal1.save_paths("../../../data/oc-dspe/dspe_ex/"\
						"paths_%.2f_%d_%d.npy" % (u_range, skip, seed))
	anneal1.save_params("../../../data/oc-dspe/dspe_ex/"\
						"params_%.2f_%d_%d.npy"	% (u_range, skip, seed))
	anneal1.save_action_errors("../../../data/oc-dspe/"\
						"dspe_ex/action_errors_%.2f_%d_%d.npy" % \
						(u_range, skip, seed))
elif anneal_bin == 1:
	anneal1.save_paths("../../../data/oc-dspe/dspe_ex/"\
						"paths_%.2f_%d_%d_4dvar.npy" % (u_range, skip, seed))
	anneal1.save_params("../../../data/oc-dspe/dspe_ex/"\
						"params_%.2f_%d_%d_4dvar.npy"	% (u_range, skip, seed))
	anneal1.save_action_errors("../../../data/oc-dspe/"\
						"dspe_ex/action_errors_%.2f_%d_%d_4dvar.npy" % \
						(u_range, skip, seed))
