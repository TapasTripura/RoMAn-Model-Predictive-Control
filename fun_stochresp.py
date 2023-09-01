# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
This code generates the synthetic stochastic responses for the systems.
"""

import scipy.io as sio
import numpy as np
import fun_library
np.random.seed(0)

"""
LOTKA-VOTERA dynamical system excited by random noise
----------------------------------------------------------------------
"""
def lotkagenerator(xinit, sysparam, tparam):
    dx, y1, y2 = [], [], []
    for i in range(len(xinit)):
        print('SDE Component:-', i)
        x0 = xinit[i]
        dxdata, y1data, y2data, t = lotkasystem(x0, sysparam, tparam)
        dx.append(dxdata)
        y1.append(y1data)
        y2.append(y2data)
    dxmodf = [dx[0][:2], dx[1][2:]] # first 2-variable is for linear variation
                                    # second 2-variable is for quadratic variation
    return dxmodf, y1, y2, t
    
def lotkasystem(xinit, sysparam, tparam):
    # parameters of LOTKA-VOTERA oscillator in Equation
    # ---------------------------------------------
    a, b, d, g, sigma1, sigma2 = sysparam
    T, dt, Nsamp, t = tparam

    sig = np.array([[sigma1, 0],[0, sigma2]])
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    # np.random.seed(2021)
    y1, y2, xz1, xz2, xzs1, xzs2 = [], [], [], [], [], []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    print('Response is being generated...')
    for ensemble in range(Nsamp):
        # if ensemble % 10 == 0:
        #     print(ensemble)
        x0 = np.array(xinit)
        x = np.vstack(x0)  
        for n in range(len(t)-1):
            dW1, dZ1 = np.dot(delmat, np.random.normal(0,1,2))
            dW2, dZ2 = np.dot(delmat, np.random.normal(0,1,2))

            a1 = a*x0[0] - b*x0[0]*x0[1] 
            a2 = d*x0[0]*x0[1] - g*x0[1]
            
            # For Taylor 1.5 integration scheme uncomment here 
            # L0a1 = a1*(a - b*x0[1]) + a2*(-b*x0[0])
            # L0a2 = a1*(d*x0[1]) + a2*(-g + d*x0[0])
            # L1a1 = sig[0,0]*(a - b*x0[1]) 
            # L1a2 = sig[0,0]*(d*x0[1]) 
            # L2a1 = sig[1,1]*(-b*x0[0])
            # L2a2 = sig[1,1]*(-g + d*x0[0])
            # sol1 = x0[0] + a1*dt + sig[0,0]*dW1 + 0.5*L0a1*(dt**2) + L1a1*dZ1 + L2a1*dZ2
            # sol2 = x0[1] + a2*dt + sig[1,1]*dW2 + 0.5*L0a2*(dt**2) + L1a2*dZ2 + L2a2*dZ2
            
            # For Taylor 1.5 integration scheme, comment here
            sol1 = x0[0] + a1*dt + sig[0,0]*dW1
            sol2 = x0[1] + a2*dt + sig[1,1]*dW2 
            
            x0 = np.array([sol1, sol2])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        
        zint1 = x[0,0:-1]
        zint2 = x[1,0:-1]
        xfinal1 = x[0,1:]
        xfinal2 = x[1,1:]
        
        xlin1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xlin2 = (xfinal2 - zint2)
        xquad1 = np.multiply(xlin1, xlin1) # '[x(t)-z]^2' vector
        xquad2 = np.multiply(xlin2, xlin2)
        
        xz1.append(xlin1)
        xz2.append(xlin2)
        xzs1.append(xquad1)
        xzs2.append(xquad2)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    xzs1 = pow(dt,-1)*np.mean(np.array(xzs1), axis = 0)
    xzs2 = pow(dt,-1)*np.mean(np.array(xzs2), axis = 0)
    
    dx = [xz1, xz2, xzs1, xzs2]
    y1 = np.array(y1)
    y2 = np.array(y2)
    return dx, y1, y2, t

def lotka_verify(xinit, tparam, params):
    # parameters of LOTKA-VOTERA oscillator in Equation
    # ---------------------------------------------
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    T, dt, Nsamp, t = tparam
    
    # np.random.seed(2021)
    y1, y2 = [], []
    print('Response is being generated...')
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        if ensemble % 15 == 0:
            print(ensemble)
        x0 = np.array(xinit)
        x = np.vstack(x0)
        for n in range(len(t)-1):
            dW1, dW2 = np.sqrt(dt)*np.random.normal(0,1,2)
            
            D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + a[0]*dt + b[0]*dW1
            sol2 = x0[1] + a[1]*dt + b[1]*dW2 
            x0 = np.array([sol1, sol2])
    
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    return y1, y2, t
    

"""
Lorenz dynamical system excited by random noise
--------------------------------------------------------------------------
"""
def lorenzgenerator(xinit, sysparam, tparam):
    dx, y1, y2, y3 = [], [], [], []
    for i in range(len(xinit)):
        print('SDE Component:-', i)
        x0 = xinit[i]
        dxdata, y1data, y2data, y3data, t = lorenzsystem(x0, sysparam, tparam)
        dx.append(dxdata)
        y1.append(y1data)
        y2.append(y2data)
        y3.append(y3data)
    dxmodf = [dx[0][:3], dx[1][3:]]
    return dxmodf, y1, y2, y3, t

def lorenzsystem(xinit, sysparam, tparam):
    # parameters of Lorenz oscillator in Equation
    # ---------------------------------------------
    alpha, rho, beta, sigma1, sigma2, sigma3 = sysparam
    T, dt, Nsamp, t = tparam
    
    sig = np.array([[sigma1, 0, 0],[0, sigma2, 0],[0, 0, sigma3]])
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    y1, y2, y3 = [], [], []
    xz1, xz2, xz3 = [], [], []
    xzs1, xzs2, xzs3 = [], [], []
    
    # Simulation Starts Here ::
    # -------------------------------------------------------
    print('Response is being generated...')
    for ensemble in range(Nsamp):
        x0 = np.array(xinit)
        x = np.vstack(x0)  
        for n in range(len(t)-1):
            dW1, dZ1 = np.dot(delmat, np.random.normal(0,1,2))
            dW2, dZ2 = np.dot(delmat, np.random.normal(0,1,2))
            dW3, dZ3 = np.dot(delmat, np.random.normal(0,1,2))
    
            a1 = alpha*(x0[1] - x0[0]) 
            a2 = x0[0]*(rho - x0[2]) - x0[1] 
            a3 = x0[0]*x0[1] - beta*x0[2]
            
            # For Taylor 1.5 integration scheme uncomment here 
            # L0a1 = a1*(-alpha) + a2*(alpha)
            # L0a2 = a1*(rho - x0[2]) + a2*(-1) + a3*(-x0[0])
            # L0a3 = a1*x0[1] + a2*x0[0] + a3*(-beta)
            # L1a1 = sig[0,0]*(-alpha) 
            # L1a2 = sig[0,0]*(rho - x0[2])
            # L1a3 = sig[0,0]*x0[1]
            # L2a1 = sig[1,1]*(alpha)
            # L2a2 = sig[1,1]*(-1)
            # L2a3 = sig[1,1]*(x0[0])
            # L3a1 = 0
            # L3a2 = sig[2,2]*(-x0[0])
            # L3a3 = sig[2,2]*(-beta)
            # sol1 = x0[0] + a1*dt + sig[0,0]*dW1 + 0.5*L0a1*(dt**2) + L1a1*dZ1 + L2a1*dZ2 + L3a1*dZ3
            # sol2 = x0[1] + a2*dt + sig[1,1]*dW2 + 0.5*L0a2*(dt**2) + L1a2*dZ1 + L2a2*dZ2 + L3a2*dZ3
            # sol3 = x0[2] + a3*dt + sig[2,2]*dW3 + 0.5*L0a3*(dt**2) + L1a3*dZ1 + L2a3*dZ2 + L3a3*dZ3
            
            # For Taylor 1.5 integration scheme comment here 
            sol1 = x0[0] + a1*dt + sig[0,0]*dW1
            sol2 = x0[1] + a2*dt + sig[1,1]*dW2
            sol3 = x0[2] + a3*dt + sig[2,2]*dW3
            
            x0 = np.array([sol1, sol2, sol3])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        
        zint1 = x[0,0:-1]
        zint2 = x[1,0:-1]
        zint3 = x[2,0:-1]
        xfinal1 = x[0,1:]
        xfinal2 = x[1,1:]
        xfinal3 = x[2,1:]
        
        xlin1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xlin2 = (xfinal2 - zint2)
        xlin3 = (xfinal3 - zint3)
        xquad1 = np.multiply(xlin1, xlin1) # '[x(t)-z]^2' vector
        xquad2 = np.multiply(xlin2, xlin2)
        xquad3 = np.multiply(xlin3, xlin3)
        
        xz1.append(xlin1)
        xz2.append(xlin2)
        xz3.append(xlin3)
        xzs1.append(xquad1)
        xzs2.append(xquad2)
        xzs3.append(xquad3)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    xz3 = pow(dt,-1)*np.mean(np.array(xz3), axis = 0)
    xzs1 = pow(dt,-1)*np.mean(np.array(xzs1), axis = 0)
    xzs2 = pow(dt,-1)*np.mean(np.array(xzs2), axis = 0)
    xzs3 = pow(dt,-1)*np.mean(np.array(xzs3), axis = 0)
    
    dx = [xz1, xz2, xz3, xzs1, xzs2, xzs3]
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    return dx, y1, y2, y3, t

def lorenz_verify(xinit, tparam, params):
    # parameters of Lorenz oscillator in Equation
    # ---------------------------------------------
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    T, dt, Nsamp, t = tparam
    
    # np.random.seed(2021)
    y1, y2, y3 = [], [], []
    
    print('Response is being generated...')
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        if ensemble % 15 == 0:
            print(ensemble)
        x0 = np.array(xinit)
        x = np.vstack(x0)
        for n in range(len(t)-1):
            dW1, dW2, dW3 = np.sqrt(dt)*np.random.normal(0,1,3)
            
            D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + a[0]*dt + b[0]*dW1
            sol2 = x0[1] + a[1]*dt + b[1]*dW2
            sol3 = x0[2] + a[2]*dt + b[2]*dW3
            x0 = np.array([sol1, sol2, sol3])
    
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    return y1, y2, y3, t



"""
NEW SYSTEM : Tuned Mass Damper (TMD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def TMDgenerator(xinit, sysparam, tparam):
    dx, y1, y2, y3, y4 = [], [], [], [], []
    for i in range(len(xinit)):
        print('SDE Component:-', i)
        x0 = xinit[i]
        dxdata, y1data, y2data, y3data, y4data, t = TMDsystem(x0, sysparam, tparam)
        dx.append(dxdata)
        y1.append(y1data)
        y2.append(y2data)
        y3.append(y3data)
        y4.append(y4data)
    dxmodf = [dx[0][:4], dx[1][-1]] # first 2-variable is for linear variation
                                    # second 2-variable is for quadratic variation
    return dxmodf, y1, y2, y3, y4, t

def TMDsystem(xinit, sysparam, tparam):
    # parameters of Tuned Mass Damper in Equation
    # ---------------------------------------------
    ms, ks, cs, mu, sigma  = sysparam
    wd = np.sqrt(ks/ms)

    md = mu*ms  # Damper mass
    fopt = 1/(1+mu)
    zopt = np.sqrt((3*mu)/(8*(1+mu)))
    kd = pow((fopt*wd),2)*md    # Damper stiffness
    cda = 2*zopt*fopt*wd*md     # Damper damping
    
    # solution by Taylor 1.5 strong scheme Run
    # -------------------------------------------------------
    T, dt, Nsamp, t = tparam
    
    sig = np.vstack([0, sigma/ms, 0, 0])
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    y1, y2, y3, y4 = [], [], [], []
    xz1, xz2, xz3, xz4 = [], [], [], []
    xzs = []
    
    # Simulation Starts Here ::
    # -------------------------------------------------------
    print('Response is being generated...')
    for ensemble in range(Nsamp):
        x0 = np.array(xinit)
        x = np.vstack(x0)  
        for n in range(len(t)-1):
            dW, dZ = np.dot(delmat, np.random.normal(0,1,2))
    
            cd = cda 
            a1 = x0[1]
            a2 = (-(cs+cd)*x0[1] +cd*x0[3] -(ks+kd)*x0[0] + kd*x0[2])/ms
            a3 = x0[3]
            a4 = (cd*x0[1] -cd*x0[3] +kd*x0[0] -kd*x0[2])/md
            
            # For Taylor 1.5 integration scheme uncomment here 
            # L0a1 = a2
            # L0a2 = a1*(-(ks+kd)/ms) + a2*(-(cs+cd)/ms) + a3*(kd/ms) + a4*(cd/ms)
            # L0a3 = a4
            # L0a4 = a1*(kd/md) + a2*(cd/md) + a3*(-kd/md) + a4*(-cd/md)
            # L1a1 = sig[1]
            # L1a2 = sig[1]*(-(cs+cd)/ms)
            # L1a3 = 0
            # L1a4 = sig[1]*(cd/md)
    
            # sol1 = x0[0] + a1*dt + 0.5*L0a1*(dt**2) + L1a1*dZ
            # sol2 = x0[1] + a2*dt + sig[1]*dW + 0.5*L0a2*(dt**2) + L1a2*dZ
            # sol3 = x0[2] + a3*dt + 0.5*L0a3*(dt**2) + L1a3*dZ
            # sol4 = x0[3] + a4*dt + 0.5*L0a4*(dt**2) + L1a4*dZ
            
            # For Taylor 1.5 integration scheme comment here 
            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + sig[1,0]*dW 
            sol3 = x0[2] + a3*dt 
            sol4 = x0[3] + a4*dt 
            
            x0 = np.array([sol1, sol2, sol3, sol4])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
        
        zint1 = x[0,0:-1]
        zint2 = x[1,0:-1]
        zint3 = x[2,0:-1]
        zint4 = x[3,0:-1]
        xfinal1 = x[0,1:]
        xfinal2 = x[1,1:]
        xfinal3 = x[2,1:]
        xfinal4 = x[3,1:]
        
        xlin1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xlin2 = (xfinal2 - zint2)
        xlin3 = (xfinal3 - zint3)
        xlin4 = (xfinal4 - zint4)
        xquad = np.multiply(xlin2, xlin2) # '[x(t)-z]^2' vector
        
        xz1.append(xlin1)
        xz2.append(xlin2)
        xz3.append(xlin3)
        xz4.append(xlin4)
        xzs.append(xquad)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    xz3 = pow(dt,-1)*np.mean(np.array(xz3), axis = 0)
    xz4 = pow(dt,-1)*np.mean(np.array(xz4), axis = 0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    
    dx = [xz1, xz2, xz3, xz4, xzs]
    return dx, y1, y2, y3, y4, t


def TMD_verify(xinit, tparam, params):
    # parameters of Tuned Mass Damper in Equation
    # ---------------------------------------------
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    T, dt, Nsamp, t = tparam
    
    # np.random.seed(2021)
    y1, y2, y3, y4 = [], [], [], []
    
    print('Response is being generated...')
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        if ensemble % 15 == 0:
            print(ensemble)
        x0 = np.array(xinit)
        x = np.vstack(x0)
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.normal(0,1)
            
            D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + a[0]*dt
            sol2 = x0[1] + a[1]*dt + b[1]*dW
            sol3 = x0[2] + a[2]*dt
            sol4 = x0[3] + a[3]*dt
            x0 = np.array([sol1, sol2, sol3, sol4])
    
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    return y1, y2, y3, y4, t



"""
76 DOF SYSTEM : Adaptive Tuned Mass Damper (ATMD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def TMD76generator(xinit, tparam):
    dx, y1, y2, y3, y4 = [], [], [], [], []
    for i in range(len(xinit)):
        print('SDE Component:-', i)
        x0 = xinit[i]
        dxdata, y1data, y2data, y3data, y4data, t = TMD76system(x0, tparam)
        dx.append(dxdata)
        y1.append(y1data)
        y2.append(y2data)
        y3.append(y3data)
        y4.append(y4data)
    dxmodf = [dx[0][:4], dx[1][-1]] # first 2-variable is for linear variation
                                    # second 2-variable is for quadratic variation
    return dxmodf, y1, y2, y3, y4, t

def TMD76system(xinit, tparam):
    # Some usefull functions:
    def tridiag(mat):
        matrix = np.zeros(mat.shape)
        for i in range(mat.shape[0]):
            if i == 0:
                matrix[i,i] = mat[i,i]
                matrix[i,i+1] = mat[i,i+1]
            elif i == mat.shape[0]-1:
                matrix[i,i] = mat[i,i]
                matrix[i,i-1] = mat[i,i-1]
            else:
                matrix[i,i] = mat[i,i]
                matrix[i,i+1] = mat[i,i+1]
                matrix[i,i-1] = mat[i,i-1]
        return matrix

    def kcmat(k,dof):
        mat = np.zeros([dof, dof])
        for i in range(dof):
            if i == 0:
                mat[i, 0] = (k[0]+k[1])
                mat[i, 1] = -k[1]
            elif i == dof-1:
                mat[i, i-1] = -k[-1]
                mat[i, i] = k[-1]
            else:
                mat[i, i-1] = -k[i]
                mat[i, i] = (k[i]+k[i+1])
                mat[i, i+1] = -k[i+1]
        return mat

    def drift(m,c,k,dof):
        mat = np.zeros([2*dof, 2*dof])
        for i in range(2*dof):
            if i % 2 == 0:
                mat[i, 1+i] = 1
            elif i == 1:
                mat[i, 0] = -(k[0]+k[1])/m[0]
                mat[i, 1] = -(c[0]+c[1])/m[0]
                mat[i, 2] = k[1]/m[0]
                mat[i, 3] = c[1]/m[0]
            elif i == 2*dof-1:
                mat[i, i-3] = k[-1]/m[-1]
                mat[i, i-2] = c[-1]/m[-1]
                mat[i, i-1] = -k[-1]/m[-1]
                mat[i, i] = -c[-1]/m[-1]
            else:
                mat[i, i-3] = k[i//2]/m[i//2]
                mat[i, i-2] = c[i//2]/m[i//2]
                mat[i, i-1] = -(k[i//2]+k[i//2+1])/m[i//2]
                mat[i, i] = -(c[i//2]+c[i//2+1])/m[i//2]
                mat[i, i+1] = k[i//2+1]/m[i//2]
                mat[i, i+2] = c[i//2+1]/m[i//2]
        return mat

    def diffusion(sigma, dof):
        mat = np.zeros([2*dof])
        for i in range(2*dof):
            if i == 1:
                mat[i] = sigma[0]
            elif i % 2 == 1:
                mat[i] = sigma[i//2]
        return mat

    """ The 76 storey building properties """
    data = sio.loadmat('data/B76_inp.mat')
    M76 = data['M76']
    K76 = data['K76']
    C76 = data['C76']

    """
    Tuned Mass Damper excited by random noise
    --------------------------------------------------------------------------
    """
    def normalize(mode):
        nmode = np.zeros(mode.shape)
        for i in range(len(mode[0])):
            nmode[:,i] = mode[:,i]/mode[-1,i]
        return nmode

    # parameters of Actual system
    # ---------------------------------------------
    ms = 153000
    eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(M76),K76))

    MT = np.matmul( np.matmul(eigvec.T, M76), eigvec)
    KT = np.matmul( np.matmul(eigvec.T, K76), eigvec)
    CT = np.matmul( np.matmul(eigvec.T, C76), eigvec)

    MT[np.where(MT<1)] = 0
    KT[np.where(MT<1)] = 0
    CT[np.where(MT<1)] = 0

    MM = MT
    KK = kcmat(np.diag(KT),76)
    CC = kcmat(np.diag(CT),76)

    eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(MM),KK))

    wd = np.sqrt(min(eigval))/2/np.pi
    phi = normalize(eigvec)
    phi = np.squeeze(phi[:, np.where(eigval == min(eigval))])
    md = 0.00327*ms

    modal_m = np.dot( np.dot(phi.T, MM), phi)
    mu = md/modal_m  # mass ratio

    fopt = 1/(1+mu)
    zopt = np.sqrt((3*mu)/(8*(1+mu)))
    kd = pow((fopt*2*np.pi*wd),2)*md*1e6
    cda = 2*zopt*fopt*(wd*2*np.pi)*md*1e3

    M77 = np.zeros([77,77])
    K77 = np.zeros([77,77])
    C77 = np.zeros([77,77])
    M77[:76,:76] = MM
    M77[-1,-1] = md

    K77[:76,:76] = KK
    K77[-2,-2] = KK[-2,-2]+kd
    K77[-2,-1] = -kd
    K77[-1,-2] = -kd
    K77[-1,-1] = kd

    C77[:76,:76] = CC
    C77[-2,-2] = CC[-2,-2]+cda
    C77[-2,-1] = -cda
    C77[-1,-2] = -cda
    C77[-1,-1] = cda

    drift77 = np.row_stack(( np.column_stack(( np.zeros(M77.shape), np.eye(M77.shape[0]) )), \
                      np.column_stack(( -np.matmul(np.linalg.inv(1e3*M77),K77), -np.matmul(np.linalg.inv(1e3*M77),C77) )) ))

    # solution by Euler-Maruyama scheme
    # -------------------------------------------------------
    T, dt, Nsamp, t = tparam

    sigma = 1*np.ones(M76.shape[0])
    # sigma = np.divide(sigma, np.diag(MM))
    sig = np.zeros(2*M77.shape[0])
    sig[77:153] = sigma

    y1, y2, y3, y4 = [], [], [], []
    xz1, xz2, xz3, xz4 = [], [], [], []
    xzs = []
    tr = 0

    # Simulation Starts Here ::
    # -------------------------------------------------------
    print('Response is being generated...')
    for ensemble in range(Nsamp):
        if ensemble % 50 == 0:
            print(ensemble)
        x0 = np.zeros(2*M77.shape[0])
        x0[75] = np.array(xinit)
        x = np.vstack(x0)  
        for n in range(len(t)-1):
            dW = np.zeros(2*M77.shape[0])
            dW[77:153] = np.sqrt(dt)*np.random.randn(M76.shape[0])
            
            a = np.dot(drift77, x0)
            sol = x0 + a*dt + np.multiply(sig, dW) 
            x0 = sol
            x = np.column_stack((x, x0))
        y1.append(x[75,tr:])   # Last floor displacement
        y2.append(x[152,tr:])  # Last floor velocity
        y3.append(x[76,tr:])   # TMD displacement
        y4.append(x[153,tr:])  # TMD velocity
        
        zint1 = x[75,tr:-1]
        zint2 = x[152,tr:-1]
        zint3 = x[76,tr:-1]
        zint4 = x[153,tr:-1]
        xfinal1 = x[75,tr+1:]
        xfinal2 = x[152,tr+1:]
        xfinal3 = x[76,tr+1:]
        xfinal4 = x[153,tr+1:]
        
        xlin1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xlin2 = (xfinal2 - zint2)
        xlin3 = (xfinal3 - zint3)
        xlin4 = (xfinal4 - zint4)
        xquad = np.multiply(xlin2, xlin2) # '[x(t)-z]^2' vector
        
        xz1.append(xlin1)
        xz2.append(xlin2)
        xz3.append(xlin3)
        xz4.append(xlin4)
        xzs.append(xquad)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    xz3 = pow(dt,-1)*np.mean(np.array(xz3), axis = 0)
    xz4 = pow(dt,-1)*np.mean(np.array(xz4), axis = 0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)

    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    time = t[tr:]
    
    dx = [xz1, xz2, xz3, xz4, xzs]
    return dx, y1, y2, y3, y4, time


def TMD76_verify(xinit, tparam, params):
    # parameters of Tuned Mass Damper in Equation
    # ---------------------------------------------
    xi_drift, xi_diff, polyorder, modfun, harmonic = params
    T, dt, Nsamp, t = tparam
    
    # np.random.seed(2021)
    y1, y2, y3, y4 = [], [], [], []
    
    print('Response is being generated...')
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        if ensemble % 10 == 0:
            print(ensemble)
        
        x0 = np.zeros(xi_drift.shape[1])
        x0[0] = xinit
        x = np.vstack(x0)
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.normal(0,1)
            
            D, nl = fun_library.library(np.vstack(x0), polyorder, modfun, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + a[0]*dt
            sol2 = x0[1] + a[1]*dt + b[1]*dW
            sol3 = x0[2] + a[2]*dt
            sol4 = x0[3] + a[3]*dt
            x0 = np.array([sol1, sol2, sol3, sol4])
    
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
        
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    return y1, y2, y3, y4, t
