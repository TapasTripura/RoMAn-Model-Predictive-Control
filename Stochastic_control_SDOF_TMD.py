# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
-- SDOF-TMD system
    This code generates the data, performs the identification, and performs
    control of an SDOF Tuned-mass-damper system.
"""

# %% Load the libraries

import numpy as np
import fun_library
import fun_spikeslab
import fun_stochresp
import fun_plots
import fun_optimize

from scipy.optimize import minimize
from scipy.optimize import Bounds

import matplotlib.pyplot as plt

# %% Response simulation

ms, ks, cs = 20, 2000, 8
mu = 0.02 # mass ratio
sigma = 6
T, dt, Nsamp = 30, 0.01, 100
t = np.arange(0, T+dt, dt)
sysparam = [ms, ks, cs, mu, sigma]
tparam = [T, dt, Nsamp, t]
x0g = [0.5, 0, 0, 0]
xref = [0, 0, 0, 0] # Critical point
dxdatag, y1g, y2g, y3g, y4g, t = fun_stochresp.TMDsystem(x0g, sysparam, tparam)

# Plot the response
y1res = np.mean(np.array(y1g), axis = 0)
y2res = np.mean(np.array(y2g), axis = 0)
y3res = np.mean(np.array(y3g), axis = 0)
y4res = np.mean(np.array(y4g), axis = 0)
figure1 = fun_plots.plot_TMDlong(dxdatag, y1res, y2res, y3res, y4res, t)

# %% Data generation for identification

# Parameters: Model
T, dt, Nsamp = 1, 0.001, 500
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
x0= [[0.5, 0, 0, 0], [0.01, 0, 0, 0]]
dxdata, y1, y2, y3, y4, t = fun_stochresp.TMDgenerator(x0, sysparam, tparam)

# Plot for training data
figure2 = fun_plots.plot_TMDdata(dxdata, y1, y2, y3, y4, t)


# %% Identification 

[xlin1, xlin2, xlin3, xlin4] = dxdata[0]
str_vars = ['x1','x2','x3','x4']
xdata = [y1[0][:,:-1], y2[0][:,:-1], y3[0][:,:-1], y4[0][:,:-1]]
dx = np.column_stack((xlin1, xlin2, xlin3, xlin4))

polyorder, modfun, harmonic = 2, 0, 0
zstore_drift, Zmean_drift, theta_drift, mut_drift, sigt_drift = [], [], [], [], []
MCMC, burn_in = 1500, 500
for i in range(len(dx[0])):
    print('Drift: state-',i)
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse_stc(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    zstore_drift.append(z1t)
    Zmean_drift.append(z2t)
    theta_drift.append(z3t)
    mut_drift.append(z4t)
    sigt_drift.append(z5t)

for i in range(len(dx[0])):
    mut_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0
    Zmean_drift[i][np.where( Zmean_drift[i] < 0.5 )] = 0

# %%
xi_drift = np.transpose(np.array(mut_drift))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_drift)

# %%
xquad1 = dxdata[1]
xdata = [y1[1][:,:-1], y2[1][:,:-1], y3[1][:,:-1], y4[1][:,:-1]]
dx = np.vstack(xquad1)

zstore_diff, Zmean_diff, theta_diff, mut_diff, sigt_diff = [], [], [], [], []
for i in range(len(dx[0])):
    print('Diffusion: state-',i)
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse_stc(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    zstore_diff.append(z1t)
    Zmean_diff.append(z2t)
    theta_diff.append(z3t)
    mut_diff.append(z4t)
    sigt_diff.append(z5t)

for i in range(len(dx[0])):
    mut_diff[i][np.where(Zmean_diff[i] < 0.75)] = 0
    Zmean_diff[i][np.where( Zmean_diff[i] < 0.75 )] = 0
    
# %%
xi_diff = np.transpose(np.array(mut_diff))
xi_diff = np.column_stack((np.zeros([len(xi_diff),1]), np.sqrt(xi_diff), np.zeros([len(xi_diff),2])))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_diff)

# %%
# Plot for Identification results
figure3 = fun_plots.TMD_results(Zmean_drift, theta_drift, 1)
figure4 = fun_plots.TMD_results(Zmean_diff, theta_diff, 2)

# %% Verification for the training data

T, dt, Nsamp = 30, 0.01, 100
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
params = [xi_drift, xi_diff, polyorder, modfun, harmonic]
xdisp1, xvel1, xdisp2, xvel2, t = fun_stochresp.TMD_verify(x0g, tparam, params)
xtrain1 = np.mean(xdisp1, axis = 0)
xtrain2 = np.mean(xvel1, axis = 0)
xtrain3 = np.mean(xdisp2, axis = 0)
xtrain4 = np.mean(xvel2, axis = 0)

# %%

# Plot for verification results
figure5 = fun_plots.TMD_verify(y1res, y2res, y3res, y4res, \
                                    xtrain1, xtrain2, xtrain3, xtrain4, t)

# %% Prediction using the identified model

# Prediction of Actual system response
T, dt, Nsamp = 60, 0.01, 100
tval = np.arange(30, T+dt, dt)
tparam = [T, dt, Nsamp, t]
xval0 = [y1res[-1], y2res[-1], y3res[-1], y4res[-1]]
dxdataval, y1g, y2g, y3g, y4g, t = fun_stochresp.TMDsystem(xval0, sysparam, tparam)
y1val = np.mean(np.array(y1g), axis = 0)
y2val = np.mean(np.array(y2g), axis = 0)
y3val = np.mean(np.array(y3g), axis = 0)
y4val = np.mean(np.array(y4g), axis = 0)

# Prediction using identified model response
xpred0 = [xtrain1[-1], xtrain2[-1],xtrain3[-1], xtrain4[-1]]
xdisp1, xvel1, xdisp2, xvel2, t = fun_stochresp.TMD_verify(xpred0, tparam, params)
xpred1 = np.mean(xdisp1, axis = 0)
xpred2 = np.mean(xvel1, axis = 0)
xpred3 = np.mean(xdisp2, axis = 0)
xpred4 = np.mean(xvel2, axis = 0)

# %%
# Plot for prediction results
figure6 = fun_plots.TMD_predict(y1res, y2res, y3res, y4res, \
                                     y1val, y2val, y3val, y4val, \
                                     xpred1, xpred2, xpred3, xpred4, t, tval)

# %% Run for different prediction horizon's:

# phorizon = [2, 5, 8, 10, 12, 15, 20]
phorizon = [10]

xcontrol = []      # Stores horizon case history
ucontrol = []      # Stores horizon case history
tcontrol = []      # Stores horizon case history
rcontrol = []      # Stores horizon case history

for case in range(len(phorizon)):
    print('Case-', case)
    Ts          = 0.01             # Sampling time
    N           = phorizon[case]   # Control / prediction horizon (number of iterations)
    Duration    = 50               # Run control for 100 time units
    Nvar        = 2
    Q           = [1, 1, 0, 0]     # State weights
    R           = 0.01             # Control variation du weights
    Ncont       = 100
    Ru = 0.01                      # Control weights
    B = [0, 1]                     # Control vector (which state is controlled)
    C = np.eye(Nvar)               # Measurement matrix
    D = 0                          # Feedforward (none)
    x0n = xval0                    # Initial condition
    uopt0 = np.zeros(N)  
    LB = -50*np.ones([N,1])        # Lower bound of control input
    UB = 50*np.ones([N,1])         # Upper bound of control input
    bounds = Bounds(LB, UB)
       
    xHistory = x0n                 # Stores state history
    uHistory = uopt0[0]            # Stores control history
    tHistory = 0                   # Stores time history
    rHistory = np.array(xref)      # Stores reference
        
    for run in range(int(Duration/Ts)):
        if run % 2 == 0:
            print(run)
            
        xref1 = xref
        arguments1 = [x0n, Ts, N, Ncont, xi_drift, xi_diff, polyorder, modfun, harmonic]
        arguments2 = [x0n, Ts, N, Ncont, xref, uopt0[0], np.diag(Q), R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic]
        OBJFUN = lambda u: fun_optimize.TMD_Objective(u, arguments2)

        cons = ({'type': 'ineq', 'fun': lambda u: fun_optimize.TMD_Constraint(u, arguments1)})
        res = minimize(OBJFUN, uopt0, method='SLSQP', 
                       jac="2-point",
                       constraints=cons,
                       bounds=bounds)
        uopt0 = res.x
        
        params2 = [xi_drift, xi_diff, polyorder, modfun, harmonic]
        xtemp = np.zeros(len(x0n))
        for ensem in range(Nsamp):
            dydt = fun_optimize.TMD_control(x0n, Ts, uopt0[0], params2)
            xtemp = np.column_stack((xtemp, dydt))
        x0n = np.mean(xtemp[:,1:], axis = 1) # removing the first column for zeros
        
        xHistory = np.column_stack((xHistory,x0n))
        uHistory = np.append(uHistory, uopt0[0])
        tHistory = np.append(tHistory, run*Ts)
        rHistory = np.column_stack((rHistory,xref1))
    
    xcontrol.append(xHistory)      # Stores case history
    ucontrol.append(uHistory)      # Stores case history
    tcontrol.append(tHistory)      # Stores case history
    rcontrol.append(rHistory)      # Stores case history
   
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 26
    
    figure7 = plt.figure(figsize = (12, 14))
    # plt.suptitle('Prediction horizon: {}'.format(phorizon[case]))
    plt.subplot(3,1,1)
    plt.plot(t, y1res, 'r')
    plt.plot(t, y2res, 'g')
    plt.plot(30+tHistory, xHistory[0,:], 'r', label='$X_s$')
    plt.plot(30+tHistory, xHistory[1,:], 'g', label='$\dot{X}_s$')
    plt.axvline(x = 32, color='k', linestyle=':', linewidth=2)
    # plt.axvline(x = 30, color='k', linestyle=':', linewidth=2)
    plt.ylabel('Primary Structure')
    plt.legend()
    plt.xlim([0, 60]);
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(t, y3res, 'b')
    plt.plot(t, y4res, 'orange')
    plt.plot(30+tHistory, xHistory[2,:], 'b', label='$X_d$')
    plt.plot(30+tHistory, xHistory[3,:], 'orange', label='$\dot{X}_d$')
    plt.axvline(x = 32, color='k', linestyle=':', linewidth=2)
    # plt.axvline(x = 30, color='k', linestyle=':', linewidth=2)
    plt.legend(prop={"size":22})
    plt.ylabel('Auxiliary Structure')
    plt.xlim([0, 60]);
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(30+tHistory, uHistory, 'k')
    plt.plot(t, np.zeros(len(y1res)), 'k')
    plt.xlabel('Time (s)');
    plt.axvline(x = 32, color='k', linestyle=':')
    # plt.axvline(x = 30, color='k', linestyle=':')
    plt.ylabel('Control-$u(t)$')
    plt.text(10, -10, "Prediction", fontsize=24)
    plt.text(40, -10, "Control", fontsize=24)
    plt.xlim([0, 60]); plt.ylim([-12, 12]);
    plt.grid(True); 
    plt.margins(0)

    