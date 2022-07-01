"""
Morris Lecar model estimated using optimal control 
dynamical state and parameter estimation.

This data has an extra channel (persistent NaP) model and will
be estimated using the ML model without this channel.

Created by Nirag Kadakia at 11:22 06-24-2022
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def L96(x, t):
	dx = np.roll(x,1) * (np.roll(x,-1) - np.roll(x,2)) - x + 8
	return dx

def ML(X, t, params):
	
	V = X[0]
	w = X[1]
	
	cap = 2.5
	g_fast = 15
	g_slow = 20
	g_leak = 2
	g_NaP = 3
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
				g_slow*w*(V - E_K) - g_leak*(V - E_leak)
				- g_NaP*m_inf(V)*(V - E_Na)) 
	dwdt = phi_w*(w_inf(V) - w)/tau_w(V)
		
	return [dVdt, dwdt]

# Generate the chaotic stimulus
stim_scale = 15
np.random.seed(0)
x0 = np.random.uniform(-5, 5, 10)
stim_Tt = np.arange(0, 20, 0.005)
res = odeint(L96, x0, stim_Tt)
stim_vec_Tt = stim_Tt[::stim_scale]*stim_scale
stim_vec = res[::stim_scale, 0]*20 + 0.5
stim_vec[stim_vec < 0] *= -1

Tt = np.arange(0, 100, 0.05)
def stim(t):
	return np.interp(t, stim_vec_Tt, stim_vec) 

# Generate the data
x0 = [-70, 0.5]
res = odeint(ML, x0, Tt, args=([2.5, 20], ))

# Stimulus is injected current, then measured data
for sigma in [0.5, 2, 5, 10]:
	noisy_V = res[:, 0] + np.random.normal(0, sigma, res[:, 0].shape)
	data_to_save = np.vstack((Tt, stim(Tt))).T
	data_to_save = np.vstack((data_to_save.T, noisy_V.T)).T
	np.save('ML_chaotic_data_NaP_sigma=%.1f.npy' % sigma, data_to_save)
	data_to_save = np.vstack((Tt, res.T)).T
	np.save('ML_chaotic_NaP_true.npy', data_to_save)
	fig = plt.figure(figsize=(5, 8))
	plt.subplot(311)
	plt.plot(Tt, stim(Tt))
	plt.subplot(312)
	plt.plot(Tt, noisy_V)
	plt.subplot(313)
	plt.plot(res[:, 0], res[:, 1])
	plt.show()
	
