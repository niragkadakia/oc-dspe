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


def ML(X, t, params):
	
	V = X[0]
	w = X[1]
	
	cap = 2.5
	g_fast = 15
	g_slow = 20
	g_leak = 2
	E_Na = 50 
	E_K = -100
	E_leak = -70
	phi_w = 0.12
	beta_w = 0
	beta_m = -1.2
	gamma_m = 18
	gamma_w = 10
	
	def m_inf(V):
		return 0.5*(1 + np.tanh((V - beta_m)/gamma_m))
	def w_inf(V):
		return 0.5*(1 + np.tanh((V - beta_w)/gamma_w))
	def tau_w(V):
		return 1./np.cosh((V - beta_w)/(2*gamma_w))
	
	dVdt = 1./cap*(stim(t) - g_fast*m_inf(V)*(V - E_Na) - 
				g_slow*w*(V - E_K) - g_leak*(V - E_leak)) 
	dwdt = phi_w*(w_inf(V) - w)/tau_w(V)
		
	return [dVdt, dwdt]

def stim(t):
	return np.interp(t, Tt, stim_vec) 

# Generate the data	
Tt = np.arange(0, 100, 0.05)
stim_vec = np.zeros(len(Tt))
stim_vec[500:] = 100
x0 = [-70, 0.5]
res = odeint(ML, x0, Tt, args=([2.5, 20], ))

# Stimulus is injected current, then measured data
for sigma in [0.5, 2, 5, 10]:
	noisy_V = res[:, 0] + np.random.normal(0, sigma, res[:, 0].shape)
	data_to_save = np.vstack((Tt, stim_vec)).T
	data_to_save = np.vstack((data_to_save.T, noisy_V.T)).T
	np.save('ML_step_data_sigma=%.1f.npy' % sigma, data_to_save)
	data_to_save = np.vstack((Tt, res.T)).T
	np.save('ML_step_true.npy', data_to_save)
	
