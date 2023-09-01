# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
This code is for deterministic (as a headstart to MPC) Lotka-Volterra system.
"""

# %% Load the libraries

import numpy as np
from scipy.integrate import odeint
import fun_spikeslab
import fun_library
from scipy.stats import gaussian_kde

from scipy.optimize import minimize
from scipy.optimize import Bounds

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timeit import default_timer
 
# %% Response simulation

uf = lambda t, A: (A*(np.sin(1*t)+np.sin(0.1*t)))**2
def lotka(y,t,params):
    A = 2
    a, b, d, g = params
    u = (A*(np.sin(1*t)+np.sin(0.1*t)))**2
    dydt = [(a*y[0]-b*y[0]*y[1]), 
            (d*y[0]*y[1]-g*y[1] + u)]
    return dydt

# Parameters: Model
a = 0.5
b = 0.025
d = 0.005
g = 0.5
n = 2
x0= [60, 50]
xref = [g/d, a/b] # Critical point

dt = 0.1
T = 200
t = np.arange(0, T+dt, dt)
params = [a, b, d, g]
x = odeint(lotka, x0, t, args=(params,))
u = uf(t,2) # Force in the system,


# %% Split into training and validation data set

Ntrain = 1001
xval = x[Ntrain:,:]
x = x[1:Ntrain,:]

uval = u[Ntrain:]
u = u[1:Ntrain]

tval = t[Ntrain-1:]
t = t[1:Ntrain]

tspanval = tval
tspan = t

# %% Plot for training data and response

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

figure1 = plt.figure(figsize =([12, 8]))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.25, hspace = 0.3)
ax1 = plt.subplot(gs[:, 0])
ax1.plot(t, x[:, 0])
ax1.plot(t, x[:, 1])
ax1.margins(0)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size')
ax1.legend(['Prey','Predator'])

ax2 = plt.subplot(gs[0, 1])
ax2.plot(x[:, 0], x[:, 1])
ax2.set_xlabel('Prey'); ax2.set_ylabel('Predator')
ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
ax2.margins(0)

ax3 = plt.subplot(gs[1, 1])
ax3.plot(t, u)
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Force, F(t)')
ax3.margins(0)
plt.show()


# %% Identification 

str_vars = ['x','y','u']
dx = np.zeros([len(x)-5, len(x[0])+1])
for i in range(2,len(x)-3):
    for k in range(len(x[0])):
        dx[i-2, k] = (1/(12*dt))*(-x[i+2,k] +8*x[i+1,k] -8*x[i-1,k] +x[i-2,k])

xdata = np.column_stack((x[2:-3,:], u[2:-3])) # Removing first and last three elements (Central diff.),
dx[:,len(x[0])] = 0 # the derivative of u-vector is zero,

polyorder, modfun, harmonic = 3, 0, 0
zstore, Zmean, theta, mut, sigt = [],[],[],[],[]
MCMC, burn_in = 100, 50
for i in range(len(x[0])):
    print('state-',i)
    t1 = default_timer()
    z1t, z2t, z3t, z4t, z5t = fun_spikeslab.sparse(dx[:,i], xdata, polyorder, modfun, harmonic, MCMC, burn_in)
    t2 = default_timer()
    print('Time-{}'.format(t2-t1))
    zstore.append(z1t)
    Zmean.append(z2t)
    theta.append(z3t)
    mut.append(z4t)
    sigt.append(z5t)

for i in range(len(x[0])):
    mut[i][np.where(Zmean[i] < 0.5)] = 0
xi = np.transpose(np.array(mut))
xi = np.column_stack((xi, np.zeros(len(xi))))
fun_library.library_list(str_vars, polyorder, modfun, harmonic, xi)
  
# %% Plot for Identification results

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

figure2 = plt.figure(figsize =([16, 10]))
gs = gridspec.GridSpec(2, 3)
gs.update(wspace = 0.3, hspace = 0.35)
ax1 = plt.subplot(gs[0, :])
xr = np.array(range(len(Zmean[0])))
ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='blue', basefmt="k")
ax1.stem(xr+0.1, Zmean[1], use_line_collection = True, linefmt='red', basefmt="k", markerfmt ='rD')
ax1.axhline(y= 0.5, color='r', linestyle='-.')
ax1.set_ylabel('PIP ', fontweight='bold');
ax1.set_xlabel('Library Functions', fontweight='bold');
ax1.set_title('(a)', fontweight='bold')
ax1.legend(['PIP = 0.5', 'Prey', 'Predator'])
ax1.grid(True); plt.ylim(0,1.05)
ax1.margins(0)

ax2 = plt.subplot(gs[1, 0])
xy = np.vstack([theta[0][1,:], theta[0][5,:]])
z = gaussian_kde(xy)(xy)
ax2.scatter(theta[0][1,:], theta[1][5,:], c=z, s=100)
ax2.set_xlabel(' 'r'$\theta (x)$', fontweight='bold'); 
ax2.set_ylabel(' 'r'$\theta (xy)$', fontweight='bold');
ax2.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
ax2.set_title('(a) Prey', fontweight='bold')
ax2.grid(True); 
# ax2.margins(0)

ax3 = plt.subplot(gs[1, 1])
xy = np.vstack([theta[1][2,:], theta[1][3,:]])
z = gaussian_kde(xy)(xy)
ax3.scatter(theta[1][2,:], theta[1][3,:], c=z, s=100)
ax3.set_xlabel(' 'r'$\theta (y)$', fontweight='bold'); 
ax3.set_ylabel(' 'r'$\theta (u)$', fontweight='bold');
ax3.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
ax3.set_title('(b) Predator', fontweight='bold')
ax3.grid(True); 
# ax3.margins(0)

ax4 = plt.subplot(gs[1, 2])
xy = np.vstack([theta[1][2,:], theta[1][5,:]])
z = gaussian_kde(xy)(xy)
ax4.scatter(theta[1][2,:], theta[1][5,:], c=z, s=100)
ax4.set_xlabel('Drift- 'r'$\theta (x)$', fontweight='bold'); 
ax4.set_ylabel(' 'r'$\theta (xy)$', fontweight='bold');
ax4.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
ax4.set_title('(c) Predator', fontweight='bold')
ax4.grid(True); 
# ax4.margins(0)

plt.show()

# %% Verification for the training data

def sparsecontrol(y, t, params):
    ahat, polyn, modfun, harmonic = params
    A = 2
    u = uf(t, A)
    y = np.row_stack((np.array(np.append(y, u))))
    D, nl = fun_library.library(y, polyn, modfun, harmonic)
    dydt = np.dot(D,ahat).reshape(-1)
    
    return dydt

ahat = np.array(mut).T
params = [ahat, polyorder, modfun, harmonic]
xtrain = odeint(sparsecontrol, x0, t, args=(params,))


# %% Plot for verification results

figure3 = plt.figure(figsize =([12, 8]))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.25, hspace = 0.3)
ax1 = plt.subplot(gs[:, 0])
ax1.plot(t, x[:, 0]); ax1.plot(t, x[:, 1]);
ax1.plot(t, xtrain[:, 0], 'r--'); ax1.plot(t, xtrain[:, 1], 'b--');
ax1.margins(0)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size')
ax1.legend(['Prey','Predator','Prey(predt)','Predator(predt)'], prop={"size":16})

ax2 = plt.subplot(gs[0, 1])
ax2.plot(x[:, 0], x[:, 1])
ax2.plot(xtrain[:, 0], xtrain[:, 1], '--')
ax2.set_xlabel('Prey'); ax2.set_ylabel('Predator')
ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
ax2.legend(['True','(Predict)'])
ax2.margins(0)

ax3 = plt.subplot(gs[1, 1])
ax3.plot(t, u)
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Force, F(t)')
ax3.margins(0)
plt.show()

# %% Prediction using the identified model

def sparsecontrol(y, t, params):
    ahat, polyn, modfun, harmonic = params
    A = 2
    u = (A*(np.sin(1*t)+np.sin(0.1*t)))**2
    y = np.row_stack((np.array(np.append(y, u))))
    D, nl = fun_library.library(y, polyn, modfun, harmonic)
    dydt = np.dot(D,ahat).reshape(-1)
    
    return dydt

xval0 = xtrain[-1,:]
ahat = np.array(mut).T
params = [ahat, polyorder, modfun, harmonic]
xpredict = odeint(sparsecontrol, xval0, tval, args=(params,))

# %% Plot for verification results

figure4 = plt.figure(figsize =([12, 8]))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.25, hspace = 0.3)
ax1 = plt.subplot(gs[:, 0])
# Training phase
ax1.plot(t, x[:, 0], 'grey'); 
ax1.plot(t, x[:, 1], 'grey');
# Actual prediction
ax1.plot(tval[1:], xval[:, 0]); 
ax1.plot(tval[1:], xval[:, 1]);
# Prediction
ax1.plot(tval[1:], xpredict[:-1, 0], 'r--');
ax1.plot(tval[1:], xpredict[:-1, 1], 'b--');
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size')
ax1.legend(['Prey(train)','Predator(train)','Prey(true)','Predator(true)', \
            'Prey(predt)','Predator(predt)'], prop={"size":16})
ax1.axvline(x= 100, color='k', linestyle=':')
ax1.margins(0)

ax2 = plt.subplot(gs[0, 1])
ax2.plot(x[:, 0], x[:, 1], 'grey')
ax2.plot(xval[:, 0], xval[:, 1], 'orange')
ax2.plot(xpredict[:, 0], xpredict[:, 1], 'b--')
ax2.set_xlabel('Prey'); ax2.set_ylabel('Predator')
ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
ax2.margins(0)

ax3 = plt.subplot(gs[1, 1])
ax3.plot(t, u, 'k')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Force, F(t)')
ax3.margins(0)
plt.show()

# %%

def rk4ivp(fun, allvalues):
    x, u, dt, n, t, ahat, polyorder, modfun, harmonic = allvalues
    arguments = [ahat, polyorder, modfun, harmonic]
    for i in range(n):
        k1 = np.array(fun(t, x, u, arguments))
        k2 = np.array(fun(t+dt/2, x+(dt*k1)/2, u, arguments))
        k3 = np.array(fun(t+dt/2, x+(dt*k2)/2, u, arguments))
        k4 = np.array(fun(t+dt, x+(dt*k3), u, arguments))
        
        x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + dt   
    return x

def lotkacontrol_discrt(t, y, u, params):
    a, b, d, g = params  
    dydt = [(a*y[0]-b*y[0]*y[1]), 
            (d*y[0]*y[1]-g*y[1] + u)]
    return dydt

def sparsecontrol_discrt(t, y, u, arguments):
    ahat, polyn, modfun, harmonic = arguments    
    y = np.row_stack((np.array(np.append(y, u))))
    
    D, nl = fun_library.library(y, polyn, modfun, harmonic)
    dydt = np.dot(D,ahat).reshape(-1)
    return dydt

def lotkaConstraint(u, params):
    # Constraint function of nonlinear MPC for Lotka-Volterra system
    # Inputs:
    #   u:      optimization variable, from time k to time k+N-1 
    #   x:      current state at time k
    #   dt:     controller sample time
    #   N:      prediction horizon
    # Output:
    #   c:      inequality constraints applied across prediction horizon
    #   ceq:    equality constraints (empty)
    
    x, dt, N, ahat, polyorder, modfun, harmonic = params
    zMin = 10 # Predator population size always > min population size of 10
    c = np.zeros(N)
    n = 1   # one step integration
    t = 0   # the time argument is zero
    xk = x  # Apply N population size constraints across prediction horizon, \
            # from time k+1 to k+N
    uk = u[0]
    for ct in range(N):
        allvalues = [xk, uk, dt, n, t, ahat, polyorder, modfun, harmonic]
        
        # obtain new cart position at next prediction step
        xk1 = rk4ivp(sparsecontrol_discrt, allvalues)
        
        c[ct] = xk1[1] - zMin # -x2k + zMin < 0, constraint for x2 > zMin
        xk = xk1
        if ct < (N-1):  # updating the controll forces,
            uk = u[ct+1]
    return c

def lotkaObjective(u, params):
    # Cost function of nonlinear MPC for Lotka-Volterra system
    # Inputs:
    #   u:      optimization variable, from time k to time k+N-1 
    #   x:      current state at time k
    #   dt:     controller sample time
    #   N:      prediction horizon
    #   xref:   state references, constant from time k+1 to k+N
    #   u0:     previous controller output at time k-1
    # Output:
    #   J:      objective function cost

    x, dt, N, xref, u0, Q, R, Ru, ahat, polyorder, modfun, harmonic = params
    # Set initial plant states, controller output and cost
    n = 1   # one step integration
    xk = x
    uk = u[0]
    t = 0
    J = 0
    for ct in range(N):     # Loop through each prediction step
        allvalues = [xk, uk, dt, n, t, ahat, polyorder, modfun, harmonic]
        
        # Obtain plant state at next prediction step
        xk1 = rk4ivp(sparsecontrol_discrt, allvalues)
        
        # Accumulate state tracking cost from x(k+1) to x(k+N)
        J = J + np.matmul(np.matmul(np.transpose(xk1-xref),Q), (xk1-xref))
        # Accumulate MV rate of change cost from u(k) to u(k+N-1)
        if ct == 0:
            J = J + np.dot(np.dot(np.transpose(uk-u0),R), (uk-u0)) + \
                np.dot(np.dot(np.transpose(uk),Ru), uk)
        else:
            J = J + np.dot(np.dot(np.transpose(uk-u[ct-1]),R), (uk-u[ct-1])) + \
                np.dot(np.dot(np.transpose(uk),Ru), uk)
            
        # Update xk and uk for the next prediction step
        xk = xk1
        if ct<(N-1):
            uk = u[ct+1]
    return J

# %% Run for different prediction horizon's:
    
# phorizon = [5, 10, 15]
phorizon = [10]
for case in range(len(phorizon)):
    print('Case-', case)
    Ts          = 0.1               # Sampling time
    N           = phorizon[case]    # Control / prediction horizon (number of iterations)
    Duration    = 100               # Run control for 100 time units
    Nvar        = 2
    Q           = [1, 0]            # State weights
    R           = 0.5               # Control variation du weights
    Ru = 0.5                        # Control weights
    B = [0, 1]                      # Control vector (which state is controlled)
    C = np.eye(Nvar)                # Measurement matrix
    D = 0                           # Feedforward (none)
    x0n = xpredict[0,:]             # Initial condition
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
        # if run % 50 == 0:
        #     print(run)
        if run*Ts > 30:            # Turn control on
            if run*Ts == 30+Ts:
                print('Start control:')
            
            xref1 = xref
            arguments1 = [x0n, Ts, N, ahat, polyorder, modfun, harmonic]
            arguments2 = [x0n, Ts, N, xref1, uopt0[0], np.diag(Q), R, Ru, ahat, polyorder, modfun, harmonic]
            OBJFUN = lambda u: lotkaObjective(u, arguments2)
    
            cons = ({'type': 'ineq', 'fun': lambda u: lotkaConstraint(u, arguments1)})
            res = minimize(OBJFUN, uopt0, method='SLSQP', 
                           jac="2-point",
                           constraints=cons,
                           bounds=None)
            uopt0 = res.x
        else:
            uopt0 = np.zeros(N)
            xref1 = [0, 0]
            
        t2 = default_timer()
        print('Time_iteration-{}, {}'.format(run, t2-t1))
        
        params2 = [x0n, uopt0[0], Ts, 1, 0, a, b, d, g]
        x0n = rk4ivp(lotkacontrol_discrt, params2)
        xHistory = np.column_stack((xHistory,x0n))
        uHistory = np.append(uHistory, uopt0[0])
        tHistory = np.append(tHistory, run*Ts)
        rHistory = np.column_stack((rHistory,xref1))

    plt.figure(1)
    plt.figure(figsize = (10, 8))
    plt.plot(100+tHistory, xHistory[0,:], label='Prey-$X_1(t)$')
    plt.plot(100+tHistory, xHistory[1,:], label='Predator-$X_2(t)$')
    plt.plot(100+tHistory, uHistory, label='Control-$u(t)$')
    plt.axvline(x = 30, color='r', linestyle=':')
    plt.legend()
    plt.ylim([-25, 250]); plt.xlim([100, 200])
    plt.grid(True)
    plt.margins(0)
