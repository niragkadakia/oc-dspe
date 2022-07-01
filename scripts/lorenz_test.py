"""
Variational annealing with 4dvar. Run estimations
on Lorenz96 system of any dimension.

Created by Nirag Kadakia at 09:23 03-09-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import sys, time
sys.path.append('../../varanneal')
from varanneal import va_ode

data_dir = '../../../data/oc-dspe'

init = int(sys.argv[1])
seed = int(sys.argv[2])
skip = int(sys.argv[3])

# Lorenz96 model dimension x2 for adjoint variables
D = 10

def l96(t, x, (params, stim)):
	gamma = params[:, 0]
	
	# dx/dt: Nudged dynamical model in x,p
	vec = np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x
	vec = (vec.T + gamma.T).T
	
	return vec

Lidx = np.arange(0, D, skip)
RM = 1.0 / (0.5**2)
RF0 = 4.0e-4
alpha = 2
beta_array = np.linspace(0, 30, 31)

data = np.load('%s/meas_data/L10_est_dt=0.05_%s.npy' % (data_dir, init))
times_data = data[:, 0]
dt_data = times_data[1] - times_data[0]
N_data = len(times_data)
data = data[:, 1:]
data = data[:, Lidx]
dt_model = dt_data
dt_model = dt_data
N_model = N_data

# Stimulus is measurement drive
stim = np.load('%s/meas_data/L10_est_dt=0.05_%s.npy' % (data_dir, init))
stim = stim[:, 1:]
stim = stim[:, ::skip]

# Parameters are only the forcing in Lorenz system (gamma)
Pidx = [0]
Uidx = []
state_bounds = [[-15, 15]]*D
param_bounds =	[[1., 100.]]
bounds = state_bounds + param_bounds

# Initial conditions; measured vars will be initialized to data
np.random.seed(seed)
X0 = (30.0*np.random.rand(N_model * D) - 15.0).reshape((N_model, D))
np.random.seed(seed)
P0 = [np.array([np.random.uniform(1, 100)])]

# Run the annealing using L-BFGS-B
tstart = time.time()
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 
				'maxiter':1000000}
anneal1 = va_ode.Annealer()
anneal1.set_model(l96, D)
anneal1.set_data(data, t=times_data, stim=stim)
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, Uidx, 
				action='A_test', dt_model=dt_model, init_to_data=True, 
				disc='SimpsonHermite', method='L-BFGS-B', opt_args=BFGS_options, 
				bounds=bounds, adolcID=0)

# Save the results
anneal1.save_paths("../../../data/oc-dspe/4dvar/paths_%d_%d_%d.npy" 
					% (skip, init, seed))
anneal1.save_params("../../../data/oc-dspe/4dvar/params_%d_%d_%d.npy" 
					% (skip, init, seed))
anneal1.save_action_errors("../../../data/oc-dspe/4dvar/action"
					"_errors_%d_%d_%d.npy" % (skip, init, seed))
