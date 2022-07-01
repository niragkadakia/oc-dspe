"""
Morris Lecar model estimated using optimal control 
dynamical state and parameter estimation.

Created by Nirag Kadakia at 11:22 05-05-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys, time
from varanneal import va_ode


data_dir = '../../../data/oc-dspe'
I_type = str(sys.argv[1])
seed = int(sys.argv[2])
noise_sigma = float(sys.argv[3])


def ML_est(t, X, args):
	
	params, stim = args
	
	I = stim[:, 0]
	data = stim[:, 1]
	v = X[:, 0]
	w = X[:, 1]
	pv = X[:, 2]
	pw = X[:, 3]	

	inv_cap = 1./2.5
	g_fast = params[:, 0] #20
	g_slow = params[:, 1] #20
	g_leak = params[:, 2] #2
	E_Na = params[:, 3] #50 
	E_K = -params[:, 4] #-100
	E_leak = -params[:, 5] #-70
	phi_w = params[:, 6]  #.12
	beta_w = 0
	beta_m = -params[:, 7] # -1.2
	inv_gamma_m = 1./params[:, 8] #1./18
	inv_gamma_w = 1./params[:, 9] #1./10
	
	def m_inf(v):
		return 0.5*(1 + np.tanh((v - beta_m)*inv_gamma_m))
		
	def w_inf(v):
		return 0.5*(1 + np.tanh((v - beta_w)*inv_gamma_w))

	def inv_tau_w(v):
		return np.cosh((v - beta_w)/2*inv_gamma_w)
	
	dvdt = inv_cap*(I - g_fast*m_inf(v)*(v - E_Na) - g_slow*w*(v - E_K)\
			- g_leak*(v - E_leak)) - pv*(data - v)**2.0
	
	dwdt = phi_w*(w_inf(v) - w)*inv_tau_w(v)

	dp1 = -inv_cap*pv*(-0.5*g_fast*inv_gamma_m*(1 - np.tanh(inv_gamma_m*\
			(-beta_m + v))**2)*(-E_Na + v) - g_fast*\
			(0.5*np.tanh(inv_gamma_m*(-beta_m + v)) + 0.5)\
			 - g_leak - g_slow*w) + \
			pw*(-0.5*inv_gamma_w*phi_w*\
			(1 - np.tanh(inv_gamma_w*(-beta_w + v))**2)*\
			np.cosh(inv_gamma_w*(-beta_w/2 + v/2)) - \
			inv_gamma_w*phi_w*(-w + 0.5*np.tanh(inv_gamma_w*(-beta_w + v)) \
			+ 0.5)*np.sinh(inv_gamma_w*(-beta_w/2 + v/2))/2)
	dp1 += (data - v)*(1 - (pv)**2.0)

	dp2 = g_slow*inv_cap*pv*(-E_K + v) + phi_w*pw*\
			np.cosh(inv_gamma_w*(-beta_w/2 + v/2))

	return np.array([dvdt, dwdt, dp1, dp2]).T


# Annealing parameters
RM = 1/16/noise_sigma**2.0
RF0 = [1e-4, 1e0, 1e0, 1e0]
alpha = 2
beta_array = np.linspace(0, 30, 31)

# Truncate data by 1 timepoint for SimpsonHermite -- bug to be fixed
_data = np.load('%s/meas_data/ML_%s_data_sigma=%.1f.npy' %
				(data_dir, I_type, noise_sigma))[:-1, :]
times_data = _data[:, 0]
dt_data = times_data[1] - times_data[0]
dt_model = dt_data
N_data = len(times_data)
N_model = N_data

# Stimulus consists of both the injected stimulus and the measured voltage
stim = _data[:, [1, 2]]

# Measured data is just the measured voltage
data = _data[:, [2]]

# Parameters now include time-dependent nudging terms; state space is 2D
D = 4
Lidx = [0]
Pidx = range(10)
Uidx = []
state_bounds = [[-100, 100]] + [[0, 1]] + [[-100, 100]] + [[-100, 100]]
param_bounds =  [[0.01, 200]]*6 + [[0.01, 100]]*4
bounds = state_bounds + param_bounds
adj_var_scaling = np.array([100., 1.])

# Initial conditions; initial forcing params set randomly to bounds
np.random.seed(seed)
X0 = np.random.uniform(-100, 100, (N_model, D))
np.random.seed(seed)
P0 = [np.array([np.random.uniform(1, 200)])]*6 + \
		[np.array([np.random.uniform(0.1, 10)])]*2 +\
		[np.array([np.random.uniform(1, 100)])]*2 
		
		
# Run the annealing using L-BFGS-B
tstart = time.time()
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 
				'maxiter':1000000}
anneal1 = va_ode.Annealer()
anneal1.set_model(ML_est, D)
anneal1.set_data(data, t=times_data, stim=stim)
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, Uidx, 
				action='A_gaussian_adjoint', dt_model=dt_model, 
				init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B', 
				opt_args=BFGS_options, bounds=bounds, adolcID=0,
				adj_var_scaling=adj_var_scaling)

# Save the results
print ('seed =', seed)
print (anneal1.minpaths[:, anneal1.N_model*anneal1.D:])
anneal1.save_paths("../../../data/oc-dspe/ML/adj/%s/"
					"/all_sigma=%.1f_paths_%d.npy" % (I_type, noise_sigma, seed))
anneal1.save_params("../../../data/oc-dspe/ML/adj/%s/"
					"all_sigma=%.1f_params_%d.npy" % (I_type, noise_sigma, seed))
anneal1.save_action_errors("../../../data/oc-dspe/ML/"
					"adj/%s/all_sigma=%.1f_action_errors_%d.npy" \
					% (I_type, noise_sigma, seed))


