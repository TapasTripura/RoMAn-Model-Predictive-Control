# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
-- LORENZ oscillator
    This code generates the data, performs the identification, and performs
    control of the Lorenz oscillator.
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

alpha, rho, beta = 10, 28, 8/3
sigma1, sigma2, sigma3 = 4, 4, 4
T, dt, Nsamp = 30, 0.01, 400
t = np.arange(0, T+dt, dt)
sysparam = [alpha, rho, beta, sigma1, sigma2, sigma3]
tparam = [T, dt, Nsamp, t]
x0g = [-8, 8, 27]
xref1 = [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)), (rho-1)] # Critical point 1
xref2 = [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)), (rho-1)] # Critical point 2
dxdatag, y1g, y2g, y3g, t = fun_stochresp.lorenzsystem(x0g, sysparam, tparam)

# Plot the response
y1res = np.mean(np.array(y1g), axis = 0)
y2res = np.mean(np.array(y2g), axis = 0)
y3res = np.mean(np.array(y3g), axis = 0)

figure1 = fun_plots.plot_lorenzlong(dxdatag, y1res, y2res, y3res, t)

# %% Data generation for identification

# Parameters: Model
T, dt, Nsamp = 1, 0.001, 400
td = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, td]
x0= [[-8, 8, 27], [-0.008, -0.008, 0]]
dxdata, y1, y2, y3, t = fun_stochresp.lorenzgenerator(x0, sysparam, tparam)

# Plot for training data
figure2 = fun_plots.plot_lorenzdata(dxdata, y1, y2, y3, td)


# %% Identification 

[xlin1, xlin2, xlin3] = dxdata[0]
x = np.mean(y1[0], axis = 0) # x-data
x = np.column_stack((x, np.mean(y2[0], axis = 0))) # y-data
x = np.column_stack((x, np.mean(y3[0], axis = 0))) # y-data

str_vars = ['x','y','z']
xdata = [y1[0][:,:-1], y2[0][:,:-1], y3[0][:,:-1]]
dx = np.column_stack((xlin1, xlin2, xlin3))

polyorder, modfun, harmonic = 2, 0, 0
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
    Zmean_drift[i][np.where( Zmean_drift[i] < 0.5 )] = 0
xi_drift = np.transpose(np.array(mut_drift))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_drift)

# %%

[xquad1, xquad2, xquad3] = dxdata[1]
xdata = [y1[1][:,:-1], y2[1][:,:-1], y3[1][:,:-1]]
dx = np.column_stack((xquad1, xquad2, xquad3))

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
    # The PIP creterion:
    mut_diff[i][np.where( Zmean_diff[i] < 0.5 )] = 0
    Zmean_diff[i][np.where( Zmean_diff[i] < 0.5 )] = 0
    # The Relative Value of theta value creterion:
    mutind = np.where( mut_diff[i] < (np.max( np.abs(mut_diff[i]) )*0.1 ) )
    mut_diff[i][mutind] = 0
    Zmean_diff[i][mutind] = 0
    # The stadard deviation creteria:
    sigtemp = sigt_diff[i]
    sigtemp = np.matmul(sigtemp, sigtemp.T)
    sigtemp = np.array(np.diag(sigtemp))
    sigtemp[mutind] = 0
    if np.sum(np.array(np.where(sigtemp > 4))) > 0:
        mut_diff[i][np.where(sigtemp > 4)] = 0
        Zmean_diff[i][np.where(sigtemp > 4)] = 0
xi_diff = np.transpose(np.array(mut_diff))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi_diff)

# Plot for Identification results
figure3 = fun_plots.lorenz_results(Zmean_drift, theta_drift, 1) # 1 for drift
figure4 = fun_plots.lorenz_results(Zmean_diff, theta_diff, 2) # 2 for diffusion

# %% Verification for the training data

T, dt, Nsamp = 30, 0.01, 400
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
params = [xi_drift, xi_diff, polyorder, modfun, harmonic]
xstate, ystate, zstate, t = fun_stochresp.lorenz_verify(x0g, tparam, params)
xtrain1 = np.mean(xstate, axis = 0)
xtrain2 = np.mean(ystate, axis = 0)
xtrain3 = np.mean(zstate, axis = 0)

# %%

# Plot for verification results
figure5 = fun_plots.lorenz_verify(y1res, y2res, y3res, xtrain1, xtrain2, xtrain3, t)

# %% Prediction using the identified model

# Prediction of Actual system response
T, dt, Nsamp = 60, 0.01, 400
tval = np.arange(30, T+dt, dt)
tparam = [T, dt, Nsamp, t]
xval0 = [y1res[-1], y2res[-1], y3res[-1]]
dxdataval, y1g, y2g, y3g, t = fun_stochresp.lorenzsystem(xval0, sysparam, tparam)
y1val = np.mean(np.array(y1g), axis = 0)
y2val = np.mean(np.array(y2g), axis = 0)
y3val = np.mean(np.array(y3g), axis = 0)

# Prediction using identified model response
xpred0 = [xtrain1[-1], xtrain2[-1], xtrain3[-1]]
xstate, ystate, zstate, t = fun_stochresp.lorenz_verify(xpred0, tparam, params)
xpred1 = np.mean(xstate, axis = 0)
xpred2 = np.mean(ystate, axis = 0)
xpred3 = np.mean(zstate, axis = 0)

# Plot for prediction results
figure6 = fun_plots.lorenz_predict(y1res, y2res, y3res, y1val, y2val, \
                                        y3val, xpred1, xpred2, xpred3, t, tval)

# %% Run for different prediction horizon's:

# phorizon = [2, 5, 8, 10, 12, 15, 20]
phorizon = [5]

xcontrol = []      # Stores case history
ucontrol = []      # Stores case history
tcontrol = []      # Stores case history
rcontrol = []      # Stores case history
time = []

for case in range(len(phorizon)):
    print('Case-', case)
    Ts          = 0.01              # Sampling time
    N           = phorizon[case]    # Control / prediction horizon (number of iterations)
    Duration    = 30                # Run control for 100 time units
    Nvar        = 3
    Q           = [1, 1, 1]         # State weights
    R           = 0.001             # Control variation du weights
    Ncont       = 400
    Ru = 0.001                      # Control weights
    B = [1, 0, 0]                   # Control vector (which state is controlled)
    C = np.eye(Nvar)                # Measurement matrix
    D = 0                           # Feedforward (none)
    x0n = xval0                     # Initial condition
    uopt0 = np.zeros(N)  
    LB = -100*np.ones([N,1])        # Lower bound of control input
    UB = 100*np.ones([N,1])         # Upper bound of control input
    bounds = Bounds(LB, UB)
    np.random.seed(2021)
       
    xHistory = x0n                  # Stores state history
    uHistory = uopt0[0]             # Stores control history
    tHistory = 0                    # Stores time history
    rHistory = np.array(xref2)      # Stores reference
        
    for run in range(int(Duration/Ts)):
        t1 = default_timer()
        force = np.random.normal(0,1, [3,Ncont])

        if run % 50 == 0:
            print(run)
        
        # Control is from begining:
        xrefc = xref2
        arguments2 = [x0n, Ts, N, Ncont, force, xrefc, uopt0[0], np.diag(Q), R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic]
        OBJFUN = lambda u: fun_optimize.lorenz_Objective(u, arguments2)

        res = minimize(OBJFUN, uopt0, method='SLSQP',
                       jac="2-point",
                       bounds=bounds)
        uopt0 = res.x
        
        t2 = default_timer()
        time.append(t2-t1)
        print('Time_iteration-{}, Time-{}'.format(run, t2-t1))
        
        params2 = [xi_drift, xi_diff, polyorder, modfun, harmonic]
        xtemp = np.zeros(len(x0n))
        for ensem in range(Ncont):
            dydt = fun_optimize.lorenz_control(x0n, force[:,ensem], Ts, uopt0[0], params2)
            xtemp = np.column_stack((xtemp, dydt))
        x0n = np.mean(xtemp[:,1:], axis = 1) # removing the first column for zeros
        
        xHistory = np.column_stack((xHistory,x0n))
        uHistory = np.append(uHistory, uopt0[0])
        tHistory = np.append(tHistory, run*Ts)
        rHistory = np.column_stack((rHistory,xrefc))
        
    xcontrol.append(xHistory)      # Stores case history
    ucontrol.append(uHistory)      # Stores case history
    tcontrol.append(tHistory)      # Stores case history
    rcontrol.append(rHistory)      # Stores case history
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 28
    
    figure7 = plt.figure(figsize = (16, 12))
    gs = gridspec.GridSpec(3, 9)
    gs.update(wspace = 0.0, hspace = 0.3)
    
    ax1 = plt.subplot(gs[0:2, 0])
    ax1.spines['right'].set_visible(False)
    plt.plot(td, -1*np.mean(y1[0], axis=0), 'b')
    plt.plot(td, -1*np.mean(y2[0], axis=0), 'm')
    plt.plot(td, np.mean(y3[0], axis=0), 'r')
    plt.ylim([-20, 40]); plt.xlim([0, 1]);
    plt.ylabel('System states')
    plt.plot(td, np.zeros(len(td)), 'k')
    plt.text(0.5, -15, "Training", rotation=90, fontsize=28)
    ax1.add_patch(Rectangle((0, -20), 1, 60, color="grey",alpha=0.15))
    plt.grid(True)
    
    ax2 = plt.subplot(gs[0:2, 1:])
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])    
    plt.plot(t, y1res, 'b'); 
    plt.plot(t, y2res, 'g');
    plt.plot(t, y3res, 'g');
    plt.plot(t, xtrain1, 'r--');
    plt.plot(t, xtrain2, '--', color='brown');
    plt.plot(t, xtrain3, '--', color='brown');
    
    plt.plot(30+tHistory, xHistory[0,:], label='X(t)')
    plt.plot(30+tHistory, xHistory[1,:], label='Y(t)')
    plt.plot(30+tHistory, xHistory[2,:], label='Z(t)')
    plt.text(13, -15, "Prediction", fontsize=28)
    plt.text(40, -15, "Control", fontsize=28)
    ax2.add_patch(Rectangle((1, -30), 29, 70,color="y",alpha=0.1))
    ax2.add_patch(Rectangle((30, -30), 30, 70,color="g",alpha=0.1))
    plt.axvline(x = 1, color='k', linestyle=':', linewidth=2)
    plt.axvline(x = 30, color='k', linestyle=':', linewidth=2)
    plt.legend(prop={"size":28}, loc='center right')
    plt.ylim([-20, 40]); plt.xlim([1, 60]); 
    plt.grid(True)
    
    ax3 = plt.subplot(gs[2, 0])
    ax3.spines['right'].set_visible(False)
    plt.plot(td, np.zeros(len(td)), 'k')
    plt.ylim([-60, 60]); plt.xlim([0, 1])
    plt.ylabel('Control-$u(t)$')
    ax3.add_patch(Rectangle((0, -60), 1, 120,color="grey",alpha=0.15))
    plt.grid(True)
    
    ax4 = plt.subplot(gs[2, 1:])
    ax4.spines['left'].set_visible(False)
    ax4.set_yticklabels([])
    plt.plot(t, np.zeros(len(y1res)), 'k')
    plt.plot(30+tHistory, uHistory)
    plt.xlabel('Time (s)'); 
    plt.axvline(x = 1, color='k', linestyle=':', linewidth=2)
    plt.axvline(x = 30, color='k', linestyle=':', linewidth=2)
    ax4.add_patch(Rectangle((1, -60), 29, 120,color="y",alpha=0.1))
    ax4.add_patch(Rectangle((30, -60), 30, 120,color="g",alpha=0.1))
    plt.ylim([-60, 60]); plt.xlim([1, 60]); 
    plt.grid(True)
    plt.margins(0)

    