# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
This code contains the functions
    -- to generates the discrete dynamics from the identified systems, 
    -- to create the constraints, and,
    -- to generate the objective function for the control.
"""

import numpy as np
import fun_library

"""
The Lotka-Votera system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" 
# The discrete stochastic dynamics
def lotka_control(xinit, dt, u, params):
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    
    x0 = np.array(xinit)
    np.random.seed(2021)
    dW1, dW2 = np.sqrt(dt)*np.random.normal(0,1,2)
    
    D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
    fa = np.dot(D, xi_drift).reshape(-1)
    fb = np.dot(D, xi_diff).reshape(-1)
    fb = np.sqrt(fb)
    
    sol1 = x0[0] + (fa[0])*dt + fb[0]*dW1
    sol2 = x0[1] + (fa[1] +u)*dt + fb[1]*dW2
    x0 = np.array([sol1, sol2])
    return x0

# The original discrete stochastic dynamics
def lotka_control_resp(xinit, dt, u, sysparams):
    a, b, d, g, sigma1, sigma2 = sysparams
    
    x0 = np.array(xinit)
    np.random.seed(2021)
    dW1, dW2 = np.sqrt(dt)*np.random.normal(0,1,2)
    
    a1 = a*x0[0] - b*x0[0]*x0[1] 
    a2 = d*x0[0]*x0[1] - g*x0[1]
    
    sol1 = x0[0] + (a1)*dt + sigma1*dW1
    sol2 = x0[1] + (a2 + u)*dt + sigma2*dW2
    x0 = np.array([sol1, sol2])
    return x0

# The constarint function for LOTKA-VOTERA system
def lotka_Constraint(u, params):
    # Constraint function of nonlinear MPC for Lotka-Volterra system
    # Inputs:
    #   u:      optimization variable, from time k to time k+N-1 
    #   x:      current state at time k
    #   dt:     controller sample time
    #   N:      prediction horizon
    # Output:
    #   c:      inequality constraints applied across prediction horizon
    #   ceq:    equality constraints (empty)
    
    x, dt, N, Nsamp, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    zMin = 10 # Predator population size always > min population size of 10
    c = np.zeros(N)
    xk = x  # Apply N population size constraints across prediction horizon, \
            # from time k+1 to k+N
    uk = u[0]
    for ct in range(N):
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # obtain new expected cart position at next prediction step
        xk1 = np.zeros(len(xk))
        for ensem in range(Nsamp):
            dydt = lotka_control(xk, dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        c[ct] = xk1[1] - zMin # x2k - zMin > 0, constraint for x2 > zMin
        xk = xk1
        if ct < (N-1):  # updating the controll forces,
            uk = u[ct+1]
    return c

# The objective function for LOTKA-VOTERA system
def lotka_Objective(u, params):
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

    x, dt, N, Nsamp, xref, u0, Q, R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    # Set initial plant states, controller output and cost
    xk = x
    uk = u[0]
    J = 0   # the objective cost function
    for ct in range(N):     # Loop through each prediction step
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # Obtain plant state at next prediction step
        xk1 = np.zeros(len(xk))
        jstar = np.zeros(1)
        for ensem in range(Nsamp):
            dydt = lotka_control(xk, dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))  
            xstar = np.vstack(dydt-xref)
            xmp = np.matmul(np.matmul(np.transpose(xstar),Q), xstar) # the norm
            jstar = np.append(jstar, xmp)
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        # Accumulate state tracking cost from x(k+1) to x(k+N)
        J = J + np.mean(jstar[1:])  # adding the expected Q-norm
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


"""
The Lorenz system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" 
# The discrete stochastic dynamics
def lorenz_control(xinit, force, dt, u, params):
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    
    x0 = np.array(xinit)
    # np.random.seed(2021)
    dW1, dW2, dW3 = np.sqrt(dt)*force
    
    D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
    fa = np.dot(D, xi_drift).reshape(-1)
    fb = np.dot(D, xi_diff).reshape(-1)
    fb = np.sqrt(fb)
    
    sol1 = x0[0] + (fa[0]+u)*dt + fb[0]*dW1
    sol2 = x0[1] + (fa[1]+u)*dt + fb[1]*dW2
    sol3 = x0[2] + (fa[2]+u)*dt + fb[2]*dW3

    x0 = np.array([sol1, sol2, sol3])
    return x0

# The original discrete stochastic dynamics
def lorenz_control_resp(xinit, force, dt, u, sysparams):
    alpha, rho, beta, sigma1, sigma2, sigma3 = sysparams
    
    x0 = np.array(xinit)
    # np.random.seed(2021)
    dW1, dW2, dW3 = np.sqrt(dt)*force
    
    a1 = alpha*(x0[1] - x0[0]) #+ u
    a2 = x0[0]*(rho - x0[2]) - x0[1] 
    a3 = x0[0]*x0[1] - beta*x0[2]
    
    sol1 = x0[0] + (a1+u)*dt + sigma1*dW1
    sol2 = x0[1] + (a2+u)*dt + sigma2*dW2
    sol3 = x0[2] + (a3+u)*dt + sigma3*dW3

    x0 = np.array([sol1, sol2, sol3])
    return x0

# The objective function for LOTKA-VOTERA system
def lorenz_Objective(u, params):
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

    x, dt, N, Nsamp, force, xref, u0, Q, R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    # Set initial plant states, controller output and cost
    xk = x
    uk = u[0]
    J = 0   # the objective cost function
    for ct in range(N):     # Loop through each prediction step
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # Obtain plant state at next prediction step
        xk1 = np.zeros(len(xk))
        jstar = np.zeros(1)
        for ensem in range(Nsamp):
            dydt = lorenz_control(xk, force[:,ensem], dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))  
            xstar = np.vstack(dydt-xref)
            xmp = np.matmul(np.matmul(np.transpose(xstar),Q), xstar) # the norm
            jstar = np.append(jstar, xmp)
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        # Accumulate state tracking cost from x(k+1) to x(k+N)
        J = J + np.mean(jstar[1:])  # adding the expected Q-norm
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




"""
The Tuned Mass Damper system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" 
# The discrete stochastic dynamics
def TMD_control(xinit, dt, u, params):
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    
    x0 = np.array(xinit)
    np.random.seed(2021)
    dW = np.sqrt(dt)*np.random.normal(0,1)
    
    D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
    fa = np.dot(D, xi_drift).reshape(-1)
    fb = np.dot(D, xi_diff).reshape(-1)
    fb = np.sqrt(fb)
    
    sol1 = x0[0] + (fa[0])*dt 
    sol2 = x0[1] + (fa[1] + u*x0[1] + u*x0[3] )*dt + fb[1]*dW
    sol3 = x0[2] + (fa[2])*dt
    sol4 = x0[3] + (fa[3] + u*x0[1] + u*x0[3] )*dt
    x0 = np.array([sol1, sol2, sol3, sol4])
    return x0

# The original discrete stochastic dynamics
def TMD_control_resp(xinit, dt, u, sysparams):
    ms, ks, cs, md, kd, cd, sigma = sysparams
    
    x0 = np.array(xinit)
    np.random.seed(2021)
    dW = np.sqrt(dt)*np.random.normal(0,1)
    
    a1 = x0[1]
    a2 = (-(cs+cd)*x0[1] +cd*x0[3] -(ks+kd)*x0[0] + kd*x0[2])/ms
    a3 = x0[3]
    a4 = (cd*x0[1] -cd*x0[3] +kd*x0[0] -kd*x0[2])/md
    
    sol1 = x0[0] + (a1)*dt
    sol2 = x0[1] + (a2 + u*x0[1] + u*x0[3])*dt + sigma*dW
    sol3 = x0[2] + (a3)*dt
    sol4 = x0[3] + (a4 + u*x0[1] + u*x0[3] )*dt
    x0 = np.array([sol1, sol2, sol3, sol4])
    return x0

# The constarint function for LOTKA-VOTERA system
def TMD_Constraint(u, params):
    # Constraint function of nonlinear MPC for Lotka-Volterra system
    # Inputs:
    #   u:      optimization variable, from time k to time k+N-1 
    #   x:      current state at time k
    #   dt:     controller sample time
    #   N:      prediction horizon
    # Output:
    #   c:      inequality constraints applied across prediction horizon
    #   ceq:    equality constraints (empty)
    
    x, dt, N, Nsamp, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    zMin = 0.02 # Predator population size always > min population size of 10
    c = np.zeros(N)
    xk = x  # Apply N population size constraints across prediction horizon, \
            # from time k+1 to k+N
    uk = u[0]
    for ct in range(N):
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # obtain new expected cart position at next prediction step
        xk1 = np.zeros(len(xk))
        for ensem in range(Nsamp):
            dydt = TMD_control(xk, dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        c[ct] = -xk1[1] + zMin # -x2k + zMin > 0, constraint for x2 < zMin
        xk = xk1
        if ct < (N-1):  # updating the controll forces,
            uk = u[ct+1]
    return c

# The objective function for LOTKA-VOTERA system
def TMD_Objective(u, params):
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

    x, dt, N, Nsamp, xref, u0, Q, R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    # Set initial plant states, controller output and cost
    xk = x
    uk = u[0]
    J = 0   # the objective cost function
    for ct in range(N):     # Loop through each prediction step
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # Obtain plant state at next prediction step
        xk1 = np.zeros(len(xk))
        jstar = np.zeros(1)
        for ensem in range(Nsamp):
            dydt = TMD_control(xk, dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))  
            xstar = np.vstack(dydt-xref)
            xmp = np.matmul(np.matmul(np.transpose(xstar),Q), xstar) # the norm
            jstar = np.append(jstar, xmp)
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        # Accumulate state tracking cost from x(k+1) to x(k+N)
        J = J + np.mean(jstar[1:])  # adding the expected Q-norm
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


"""
The 76 DOF system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" 
# The discrete stochastic dynamics
def TMD76_control(xinit, force, dt, u, params):
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    
    x0 = np.array(xinit)
    # np.random.seed(2021)
    dW = np.sqrt(dt)*force
    
    D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
    fa = np.dot(D, xi_drift).reshape(-1)
    fb = np.dot(D, xi_diff).reshape(-1)
    fb = np.sqrt(fb)
    
    sol1 = x0[0] + (fa[0])*dt 
    sol2 = x0[1] + (fa[1] + u)*dt + fb[1]*dW
    sol3 = x0[2] + (fa[2])*dt
    sol4 = x0[3] + (fa[3] + u)*dt
    x0 = np.array([sol1, sol2, sol3, sol4])
    return x0

# The original discrete stochastic dynamics
def TMD76_control_resp(xinit, force, dt, u, sysparams):
    ms, ks, cs, md, kd, cd, sigma = sysparams
    
    x0 = np.array(xinit)
    # np.random.seed(2021)
    dW = np.sqrt(dt)*force
    
    a1 = x0[1]
    a2 = (-(cs+cd)*x0[1] +cd*x0[3] -(ks+kd)*x0[0] + kd*x0[2])/ms
    a3 = x0[3]
    a4 = (cd*x0[1] -cd*x0[3] +kd*x0[0] -kd*x0[2])/md
    
    sol1 = x0[0] + (a1)*dt
    sol2 = x0[1] + (a2 - u*x0[1] + u*x0[3])*dt + sigma*dW
    sol3 = x0[2] + (a3)*dt
    sol4 = x0[3] + (a4 + u*x0[1] - u*x0[3])*dt
    x0 = np.array([sol1, sol2, sol3, sol4])
    return x0

# The constarint function for LOTKA-VOTERA system
def TMD76_Constraint(u, params):
    # Constraint function of nonlinear MPC for Lotka-Volterra system
    # Inputs:
    #   u:      optimization variable, from time k to time k+N-1 
    #   x:      current state at time k
    #   dt:     controller sample time
    #   N:      prediction horizon
    # Output:
    #   c:      inequality constraints applied across prediction horizon
    #   ceq:    equality constraints (empty)
    
    x, dt, N, Nsamp, force, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    zMin = 0.01 # Predator population size always > min population size of 10
    c = np.zeros(N)
    xk = x  # Apply N population size constraints across prediction horizon, \
            # from time k+1 to k+N
    uk = u[0]
    for ct in range(N):
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # obtain new expected cart position at next prediction step
        xk1 = np.zeros(len(xk))
        for ensem in range(Nsamp):
            dydt = TMD76_control(xk, force[ensem], dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        c[ct] = -xk1[1] + zMin # -x2k + zMin > 0, constraint for x2 < zMin
        xk = xk1
        if ct < (N-1):  # updating the controll forces,
            uk = u[ct+1]
    return c

# The objective function for LOTKA-VOTERA system
def TMD76_Objective(u, params):
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

    x, dt, N, Nsamp, force, xref, u0, Q, R, Ru, xi_drift, xi_diff, polyorder, modfun, harmonic = params
    # Set initial plant states, controller output and cost
    xk = x
    uk = u[0]
    J = 0   # the objective cost function
    for ct in range(N):     # Loop through each prediction step
        allvalues = [xi_drift, xi_diff, polyorder, modfun, harmonic] 
        
        # Obtain plant state at next prediction step
        xk1 = np.zeros(len(xk))
        jstar = np.zeros(1)
        for ensem in range(Nsamp):
            dydt = TMD76_control(xk, force[ensem], dt, uk, allvalues)
            xk1 = np.column_stack((xk1, dydt))  
            xstar = np.vstack(dydt-xref)
            xmp = np.matmul(np.matmul(np.transpose(xstar),Q), xstar) # the norm
            jstar = np.append(jstar, xmp)
        xk1 = np.mean(xk1[:,1:], axis = 1) # removing the first column for zeros
        
        # Accumulate state tracking cost from x(k+1) to x(k+N)
        J = J + np.mean(jstar[1:])  # adding the expected Q-norm
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
