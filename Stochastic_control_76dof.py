# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
-- 76DOF-slender system
    This code generates the data, performs the identification, and performs
    control of an 76DOF slender system.
"""

# %% Load the libraries

import numpy as np
import fun_spikeslab
import fun_library
import fun_stochresp
import fun_plots
import fun_optimize

from scipy import signal
from scipy.optimize import minimize
from scipy.optimize import Bounds

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from timeit import default_timer

# %% Response simulation
np.random.seed(0)
T, dt, Nsamp = 1, 0.0005, 400
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
x0g = 0.1
xref = [0, 0, 0, 0] # Critical point
dxdatag, y1g, y2g, y3g, y4g, t = fun_stochresp.TMD76system(x0g, tparam)

# Plot the response
y1res = np.mean(np.array(y1g), axis = 0)
y2res = np.mean(np.array(y2g), axis = 0)
y3res = np.mean(np.array(y3g), axis = 0)
y4res = np.mean(np.array(y4g), axis = 0)
figure1 = fun_plots.plot_TMD76long(dxdatag, y1res, y2res, y3res, y4res, t)

# %% Data generation for identification

# Parameters: Model
T, dt, Nsamp = 1, 0.0005, 200
t = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, t]
x0= [[0.1], [0]]
dxdata, y1, y2, y3, y4, t = fun_stochresp.TMD76generator(x0, tparam)

# Plot for training data
figure2 = fun_plots.plot_TMD76data(dxdata, y1, y2, y3, y4, t)


# %% Identification using SINDYc

[xlin1, xlin2, xlin3, xlin4] = dxdata[0]
str_vars = ['x1','x2','x3','x4']
xdata = [y1[0][:,:-1], y2[0][:,:-1], y3[0][:,:-1], y4[0][:,:-1]]
dx = np.column_stack((xlin1, xlin2, xlin3, xlin4))

polyorder, modfun, harmonic = 2, 0, 0
zstore_drift, Zmean_drift, theta_drift, mut_drift, sigt_drift = [], [], [], [], []
MCMC, burn_in = 50, 20
for i in range(len(dx[0])):
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
    t1 = default_timer()
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse_stc(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    t2 = default_timer()
    print('Time-{}'.format(t2-t1))
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
figure3 = fun_plots.TMD76_results(Zmean_drift, theta_drift, 1)
figure4 = fun_plots.TMD76_results(Zmean_diff, theta_diff, 2)

# %% Verification for the training data
np.random.seed(0)
T, dt, Nsamp = 1, 0.0005, 400
tval = np.arange(0, T+dt, dt)
tparam = [T, dt, Nsamp, tval]
params = [xi_drift, xi_diff, polyorder, modfun, harmonic]
xdisp1, xvel1, xdisp2, xvel2, _ = fun_stochresp.TMD76_verify(x0g, tparam, params)
xtrain1 = np.mean(xdisp1, axis = 0)
xtrain2 = np.mean(xvel1, axis = 0)
xtrain3 = np.mean(xdisp2, axis = 0)
xtrain4 = np.mean(xvel2, axis = 0)

# %%
y1resd = signal.detrend(y1res)
y2resd = signal.detrend(y2res)
y3resd = signal.detrend(y3res)
y4resd = signal.detrend(y4res)

xtrain1d = signal.detrend(xtrain1)
xtrain2d = signal.detrend(xtrain2)
xtrain3d = signal.detrend(xtrain3)
xtrain4d = signal.detrend(xtrain4)

# %%
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 28
plt.rcParams['font.weight'] = "normal"

figure5 = plt.figure(figsize =([14, 20]))
gs = gridspec.GridSpec(3, 2)
gs.update(wspace = 0.5, hspace = 0.3)

ax1 = plt.subplot(gs[0, 0])
ax1.plot(t, y1resd, '--', color='grey');     
ax1.plot(tval, xtrain1d, 'r:'); 
ax1.margins(0)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('$X_1$(t)', fontsize=36)
ax1.legend(['Actual','Identified'], prop={"size":26})

ax2 = plt.subplot(gs[0, 1])
ax2.plot(t, y2resd, '--', color='grey');     
ax2.plot(tval, xtrain2d, 'b:');
ax2.margins(0)
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('$X_2$(t)', fontsize=36)
ax2.legend(['Actual','Identified'], prop={"size":26})

ax3 = plt.subplot(gs[1, 0])
ax3.plot(t, y3resd, '--', color='grey');     
ax3.plot(tval, xtrain3d, 'g:');
ax3.margins(0)
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('$X_3$(t)', fontsize=36)
ax3.legend(['Actual','Identified'], prop={"size":26})

ax4 = plt.subplot(gs[1, 1])
ax4.plot(t, y4resd, '--', color='grey');     
ax4.plot(tval, xtrain4d, 'm:');
ax4.margins(0)
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('$X_4$(t)', fontsize=36)
ax4.legend(['Actual','Identified'], prop={"size":26})

ax5 = plt.subplot(gs[2, 0])
ax5.plot(y1resd, y2resd)
ax5.plot(xtrain1d, xtrain2d, '--')
ax5.set_xlabel('$X_1$(t)'); ax5.set_ylabel('$X_2$(t)', fontsize=36)
ax5.legend(['True','(Predict)'], prop={"size":26})
ax5.margins(0)

ax6 = plt.subplot(gs[2, 1])
ax6.plot(y3resd, y4resd)
ax6.plot(xtrain3d, xtrain4d, '--')
ax6.set_xlabel('$X_3$(t)'); ax6.set_ylabel('$X_4$(t)', fontsize=36)
ax6.legend(['True','(Predict)'], prop={"size":26})
ax6.margins(0)
plt.show()


# %% Prediction using the identified model

# Prediction of Actual system response
np.random.seed(0)
T, dt, Nsamp = 2, 0.0005, 400
tpred = np.arange(1, T+dt, dt)
tparam = [T, dt, Nsamp, tpred]
xval0 = [y1res[-1], y2res[-1], y3res[-1], y4res[-1]]
dxdataval, y1g, y2g, y3g, y4g, _ = fun_stochresp.TMD76system(y1res[-1], tparam)
y1val = np.mean(np.array(y1g), axis = 0)
y2val = np.mean(np.array(y2g), axis = 0)
y3val = np.mean(np.array(y3g), axis = 0)
y4val = np.mean(np.array(y4g), axis = 0)

# Prediction using identified model response
np.random.seed(0)
T, dt, Nsamp = 2, 0.0005, 50
tpred = np.arange(1, T+dt, dt)
tparam = [T, dt, Nsamp, tpred]
xpred0 = [xtrain1[-1], xtrain2[-1],xtrain3[-1], xtrain4[-1]]
xdisp1, xvel1, xdisp2, xvel2, _ = fun_stochresp.TMD76_verify(xtrain1[-1], tparam, params)
xpred1 = np.mean(xdisp1, axis = 0)
xpred2 = np.mean(xvel1, axis = 0)
xpred3 = np.mean(xdisp2, axis = 0)
xpred4 = np.mean(xvel2, axis = 0)

# %%
y1vald = signal.detrend(y1val)
y2vald = signal.detrend(y2val)
y3vald = signal.detrend(y3val)
y4vald = signal.detrend(y4val)

xpred1d = signal.detrend(xpred1)
xpred2d = signal.detrend(xpred2)
xpred3d = signal.detrend(xpred3)
xpred4d = signal.detrend(xpred4)

# Plot for prediction results
figure6 = fun_plots.TMD76_verify(y1vald, y2vald, y3vald, y4vald, \
                                     xpred1d, xpred2d, xpred3d, xpred4d, tpred, tpred)

# %% Run for different prediction horizon's:

# phorizon = [2, 5, 8, 10, 12, 15, 20]
phorizon = [8]

xcontrol = []      # Stores horizon case history
ucontrol = []      # Stores horizon case history
tcontrol = []      # Stores horizon case history
rcontrol = []      # Stores horizon case history
time = []

for case in range(len(phorizon)):
    print('Case-', case)
    Ts          = 0.0005             # Sampling time
    N           = phorizon[case]         # Control / prediction horizon (number of iterations)
    Duration    = 10            # Run control for 100 time units
    Q           = [1, 1, 0, 0]           # State weights
    R           = 0.0001         # Control variation du weights
    Ncont       = 200
    Ru = 0.0001                 # Control weights
    B = [0, 1]                     # Control vector (which state is controlled)
    x0n = xpred0             # Initial condition
    uopt0 = np.zeros(N)  
    LB = -1e15*np.ones([N,1])        # Lower bound of control input
    UB = 1e15*np.ones([N,1])         # Upper bound of control input
    bounds = Bounds(LB, UB)
       
    xHistory = x0n      # Stores state history
    uHistory = uopt0[0]  # Stores control history
    tHistory = 0         # Stores time history
    rHistory = np.array(xref)      # Stores reference
        
    for run in range(int(Duration/Ts)):
        t1 = default_timer()
        force = np.random.normal(0,1, Ncont)
        if run % 50 == 0:
            print(run)
            
        if run % 10 == 0:
            xref1 = xref
            arguments1 = [x0n, Ts, N, Ncont, force, xi_drift, xi_diff, polyorder, modfun, harmonic]
            arguments2 = [x0n, Ts, N, Ncont, force, xref, uopt0[0], np.diag(Q), R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic]
            OBJFUN = lambda u: fun_optimize.TMD76_Objective(u, arguments2)
    
            cons = ({'type': 'ineq', 'fun': lambda u: fun_optimize.TMD76_Constraint(u, arguments1)})
            res = minimize(OBJFUN, uopt0, method='SLSQP', 
                           jac="2-point",
                            constraints=cons,
                           bounds=bounds)
        uopt0 = res.x
        t2 = default_timer()
        time.append(t2-t1)
        print('Time_iteration-{}, Time-{}'.format(run, t2-t1))
        
        params2 = [xi_drift, xi_diff, polyorder, modfun, harmonic]
        xtemp = np.zeros(len(x0n))
        for ensem in range(Ncont):
            dydt = fun_optimize.TMD76_control(x0n, force[ensem], Ts, uopt0[0], params2)
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
    xHistoryd = xHistory
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 32
    
    figure7 = plt.figure(figsize = (20, 10))
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace = 0.0, hspace = 0.3)

    ax1 = plt.subplot(gs[0:2])
    plt.plot(t, y1res, 'r')
    plt.plot(t, y2res, 'g')
    plt.plot(1+tHistory[:5990], xHistory[0,:5990]*0.5, 'r', label='$X_s$')
    plt.plot(1+tHistory[:5990], xHistory[1,:5990]*0.5, 'g', label='$\dot{X}_s$')
    plt.axvline(x = 1, color='k', linestyle=':', linewidth=2)
    plt.ylabel('Primary \n Structure')
    plt.legend(prop={"size":30}, ncol=2)
    plt.xlim([0, 4]);
    ax1.add_patch(Rectangle((0, -0.7), 1, 14, color="grey", alpha=0.2))
    ax1.add_patch(Rectangle((1, -0.7), 9, 14, color="green", alpha=0.1))
    plt.grid(True)
    
    ax2 = plt.subplot(gs[2])
    plt.plot(t, y3res, 'b')
    plt.plot(t, y4res, 'm')
    plt.plot(1+tHistory[:5990], xHistory[2,:5990]*0.5, 'b', label='$X_d$')
    plt.plot(1+tHistory[:5990], xHistory[3,:5990]*0.5, 'm', label='$\dot{X}_d$')
    plt.axvline(x = 1, color='k', linestyle=':', linewidth=2)
    plt.legend(prop={"size":30}, ncol=2, loc=1)
    plt.ylabel('Auxiliary \n Structure')
    plt.xlim([0, 4]);
    ax2.add_patch(Rectangle((0, -3), 1, 14, color="grey", alpha=0.2))
    ax2.add_patch(Rectangle((1, -3), 9, 14, color="green", alpha=0.1))
    plt.grid(True)
    
    ax3 = plt.subplot(gs[3])
    plt.plot(1+tHistory, uHistory, 'k')
    plt.plot(t, np.zeros(len(y1res)), 'k')
    plt.xlabel('Time (s)');
    plt.axvline(x = 1, color='k', linestyle=':')
    plt.ylabel('Control \n $u(t)$')
    plt.text(0.05, 3, "Training", fontsize=30)
    plt.text(2.5, 3, "Control", fontsize=30)
    plt.xlim([0, 4]); plt.ylim([-100, 300]);
    ax3.add_patch(Rectangle((0, -100), 1, 30, color="grey", alpha=0.2))
    ax3.add_patch(Rectangle((1, -100), 9, 30, color="green", alpha=0.1))
    plt.grid(True); 
    plt.margins(0)
