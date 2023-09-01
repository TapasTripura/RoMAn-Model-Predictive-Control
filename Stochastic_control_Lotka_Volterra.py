# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
-- LOTKA-VOLTERRA system
    This code generates the data, performs the identification, and performs
    control of the Lotka-Volterrra system.
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from timeit import default_timer

np.random.seed(0)

# %% Response simulation

a, b, d, g = 0.5, 0.025, 0.005, 0.5
sigma1, sigma2 = 0.2, 0.2
T, dt, Nsamp = 100, 0.01, 100
t = np.arange(0, T+dt, dt)
sysparam = [a, b, d, g, sigma1, sigma2]
tparam = [T, dt, Nsamp, t]
x0g = [60, 50]
xref = [g/d, a/b] # Critical point
dxdatag, y1g, y2g, t = fun_stochresp.lotkasystem(x0g, sysparam, tparam)

# Plot the response
y1res = np.mean(np.array(y1g), axis = 0)
y2res = np.mean(np.array(y2g), axis = 0)
figure1 = fun_plots.plot_long(dxdatag, y1res, y2res, t)

# %% Data generation for identification

# Parameters: Model
T, dt, Nsamp = 1, 0.001, 500
td = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, td]
x0= [[60, 50], [0.60, 0.50]]
dxdata, y1, y2, t = fun_stochresp.lotkagenerator(x0, sysparam, tparam)

# Plot for training data
figure2 = fun_plots.plot_data(dxdata, y1, y2, td)


# %% Identification

[xlin1, xlin2] = dxdata[0]
x = np.mean(y1[0], axis = 0) # displacement data
x = np.column_stack((x, np.mean(y2[0], axis = 0))) # velocity data

str_vars = ['x','y']
xdata = [y1[0][:,:-1], y2[0][:,:-1]]
dx = np.column_stack((xlin1, xlin2))

polyorder, modfun, harmonic = 3, 0, 0
zstore_drift, Zmean_drift, theta_drift, mut_drift, sigt_drift = [], [], [], [], []
MCMC, burn_in = 1500, 500
for i in range(len(x[0])):
    print('Drift: state-',i)
    t1 = default_timer()
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse_stc(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    t2 = default_timer()
    print('Time-{}'.format(t2-t1))
    zstore_drift.append(z1t)
    Zmean_drift.append(z2t)
    theta_drift.append(z3t)
    mut_drift.append(z4t)
    sigt_drift.append(z5t)

for i in range(len(x[0])):
    mut_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0
xi_drift = np.transpose(np.array(mut_drift))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_drift)

[xquad1, xquad2] = dxdata[1]
xdata = [y1[1][:,:-1], y2[1][:,:-1]]
dx = np.column_stack((xquad1, xquad2))

figure3 = fun_plots.lotka_results(Zmean_drift, theta_drift, 1)

zstore_diff, Zmean_diff, theta_diff, mut_diff, sigt_diff = [], [], [], [], []
for i in range(len(x[0])):
    print('Diffusion: state-',i)
    t1 = default_timer()
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse_stc(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    t2 = default_timer()
    print('Time-{}'.format(t2-t1))
    zstore_diff.append(z1t)
    Zmean_diff.append(z2t)
    theta_diff.append(z3t)
    mut_diff.append(z4t)
    sigt_diff.append(z5t)

for i in range(len(x[0])):
    mut_diff[i][np.where(Zmean_diff[i] < 0.5)] = 0
xi_diff = np.transpose(np.array(mut_diff))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_diff)

# Plot for Identification results
figure3 = fun_plots.lotka_results(Zmean_drift, theta_drift, 1)
figure4 = fun_plots.lotka_results(Zmean_diff, theta_diff, 2)

# %% Verification for the training data

T, dt, Nsamp = 100, 0.01, 100
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
params = [xi_drift, xi_diff, polyorder, modfun, harmonic]
xdisp, xvel, t = fun_stochresp.lotka_verify(x0g, tparam, params)
xtrain1 = np.mean(xdisp, axis = 0)
xtrain2 = np.mean(xvel, axis = 0)

# %%

# Plot for verification results
figure5 = fun_plots.lotka_verify(y1res, y2res, xtrain1, xtrain2, t)

# %% Prediction using the identified model

# Prediction of Actual system response
T, dt, Nsamp = 200, 0.01, 100
tval = np.arange(100, T+dt, dt)
tparam = [T, dt, Nsamp, t]
xval0 = [y1res[-1], y2res[-1]]
dxdataval, y1g, y2g, t = fun_stochresp.lotkasystem(xval0, sysparam, tparam)
y1val = np.mean(np.array(y1g), axis = 0)
y2val = np.mean(np.array(y2g), axis = 0)

# Prediction using identified model response
xpred0 = [xtrain1[-1], xtrain2[-1]]
xdisp, xvel, t = fun_stochresp.lotka_verify(xpred0, tparam, params)
xpred1 = np.mean(xdisp, axis = 0)
xpred2 = np.mean(xvel, axis = 0)

# %%
# Plot for prediction results
figure6 = fun_plots.lotka_predict(y1res, y2res, y1val, y2val, xpred1, xpred2, t, tval)

# %% Run for different prediction horizon's:

# phorizon = [2, 5, 8, 10, 12, 15, 20]
phorizon = [10]

xcontrol = []      # Stores horizon case history
ucontrol = []      # Stores horizon case history
tcontrol = []      # Stores horizon case history
rcontrol = []      # Stores horizon case history
time = []

for case in range(len(phorizon)):
    print('Case-', case)
    Ts          = 0.1               # Sampling time
    N           = phorizon[case]    # Control / prediction horizon (number of iterations)
    Duration    = 100               # Run control for 100 time units
    Nvar        = 2
    Q           = [1, 1]            # State weights
    R           = 0.5               # Control variation du weights
    Ncont       = 100
    Ru = 0.5                        # Control weights
    B = [0, 1]                      # Control vector (which state is controlled)
    C = np.eye(Nvar)                # Measurement matrix
    D = 0                           # Feedforward (none)
    x0n = xpred0 #xval0             # Initial condition
    uopt0 = np.zeros(N)  
    LB = -100*np.ones([N,1])        # Lower bound of control input
    UB = 100*np.ones([N,1])         # Upper bound of control input
    bounds = Bounds(LB, UB)
       
    xHistory = x0n                  # Stores state history
    uHistory = uopt0[0]             # Stores control history
    tHistory = 0                    # Stores time history
    rHistory = np.array(xref)       # Stores reference
        
    for run in range(int(Duration/Ts)):
        t1 = default_timer()
        if run % 50 == 0:
            print(run)
        if run*Ts > 30:            # Turn control on
            if run*Ts == 30+Ts:
                print('Start control:')
            
            xref1 = xref
            arguments1 = [x0n, Ts, N, Ncont, xi_drift, xi_diff, polyorder, modfun, harmonic]
            arguments2 = [x0n, Ts, N, Ncont, xref, uopt0[0], np.diag(Q), R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic]
            OBJFUN = lambda u: fun_optimize.lotka_Objective(u, arguments2)
    
            cons = ({'type': 'ineq', 'fun': lambda u: fun_optimize.lotka_Constraint(u, arguments1)})
            res = minimize(OBJFUN, uopt0, method='SLSQP', 
                           jac="2-point",
                           constraints=cons,
                           bounds=bounds)
            uopt0 = res.x
        else:
            uopt0 = np.zeros(N)
            xref1 = [0, 0]
        t2 = default_timer()
        time.append(t2-t1)
        print('Time_iteration-{}, Time-{}'.format(run, t2-t1))
        
        params2 = [xi_drift, xi_diff, polyorder, modfun, harmonic]
        xtemp = np.zeros(len(x0n))
        for ensem in range(Nsamp):
            dydt = fun_optimize.lotka_control(x0n, Ts, uopt0[0], params2)
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
    
    figure7 = plt.figure(figsize = (14, 10))
    gs = gridspec.GridSpec(1, 8)
    gs.update(wspace = 0.0, hspace = 0.3)
    
    ax1 = plt.subplot(gs[0])
    ax1.spines['right'].set_visible(False)
    plt.plot(td, np.mean(y1[0], axis=0), 'b')
    plt.plot(td, np.mean(y2[0], axis=0), 'g')
    plt.ylim([-30, 300]); plt.xlim([0, 1])
    plt.ylabel('Population Size')
    plt.plot(td, np.zeros(len(td)), 'k')
    plt.text(0.5, 100, "Training", rotation=90, fontsize=24)
    ax1.add_patch(Rectangle((0, -30), 1, 330,color="grey",alpha=0.25))
    plt.grid(True)
    
    ax2 = plt.subplot(gs[1:])
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])
    plt.plot(tval-100, y1res, 'b'); 
    plt.plot(tval-100, y2res, 'g');
    plt.plot(tval-100, xtrain1, 'r--');
    plt.plot(tval-100, xtrain2, '--', color='brown');
    plt.plot(tval-100, np.zeros(len(xpred1)), 'k')
    
    plt.plot(100+tHistory, xHistory[0,:], 'r', label='Prey')
    plt.plot(100+tHistory, xHistory[1,:], 'brown', label='Predator')
    plt.plot(100+tHistory, uHistory, 'k',label='Control-u(t)')
    
    plt.text(50, -20, "Prediction", fontsize=24)
    ax2.add_patch(Rectangle((1, -30), 128, 330,color="yellow",alpha=0.15))
    plt.text(150, -20, "Control", fontsize=24)
    ax2.add_patch(Rectangle((130, -30), 70, 330,color="g",alpha=0.15))
    plt.ylim([-30, 300]); plt.xlim([1, 200]);
    # plt.title('Prediction horizon: {}'.format(phorizon[case]))
    plt.grid(True)
    
    plt.xlabel('Time (s)');
    plt.legend(prop={"size":22})
    plt.axvline(x= 1, color='k', linestyle=':', linewidth=2)
    plt.axvline(x= 130, color='k', linestyle=':', linewidth=2)
    plt.margins(0)
    
