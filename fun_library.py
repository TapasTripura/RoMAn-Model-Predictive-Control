# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
This code generates the library for sparse Bayesian inference.
"""

import numpy as np
import pandas as pd
from scipy import linalg as LA
from sklearn.metrics import mean_squared_error as MSE

"""
The Dictionary creation part:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def library(xt, polyn, modfun, harmonic):
    # The input data matrix xt = [m, N]
    # where, m = no. of states, N = no. of time integration steps,
    if polyn == 0:
        polyn = 1
    # poly order 0
    ind = 0
    n = len(xt[0]) # getting the no. of the time steps in data
    D = np.ones([n,1])
    if polyn >= 1:
        # poly order 1
        for i in range(len(xt)): # len(xt) = no. of states in the data
            ind = ind+1
            new = np.vstack(xt[i,:])
            D = np.append(D, new, axis=1)
    if polyn >= 2: 
        # ploy order 2
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                ind = ind+1
                new = np.multiply(xt[i,:], xt[j,:])
                new = np.vstack(new)
                D = np.append(D, new, axis=1) 
    if polyn >= 3:    
        # ploy order 3
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    ind = ind+1
                    new = np.multiply(np.multiply(xt[i,:], xt[j,:]), xt[k,:])
                    new = np.vstack(new)
                    D = np.append(D, new, axis=1) 
    if polyn >= 4:
        # ploy order 4
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in range(k,len(xt)):
                        ind = ind+1
                        new = np.multiply(np.multiply(xt[i,:], xt[j,:]), xt[k,:])
                        new = np.multiply(new, xt[l,:])
                        new = np.vstack(new)
                        D = np.append(D, new, axis=1) 
    if polyn >= 5:
        # ploy order 5
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in  range(k,len(xt)):
                        for m in  range(l,len(xt)):
                            ind = ind+1
                            new = np.multiply(xt[i,:], xt[j,:])
                            new = np.multiply(new, xt[k,:])
                            new = np.multiply(new, xt[l,:])
                            new = np.multiply(new, xt[m,:])
                            new = np.vstack(new)
                            D = np.append(D, new, axis=1) 
    if polyn >= 6:
        # ploy order 6
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in  range(k,len(xt)):
                        for m in  range(l,len(xt)):
                            for n in  range(m,len(xt)):
                                ind = ind+1
                                new = np.multiply(xt[i,:], xt[j,:])
                                new = np.multiply(new, xt[k,:])
                                new = np.multiply(new, xt[l,:])
                                new = np.multiply(new, xt[m,:])
                                new = np.multiply(new, xt[n,:])
                                new = np.vstack(new)
                                D = np.append(D, new, axis=1) 
    if modfun == 1:
        # for the signum or sign operator
        for i in range(len(xt)):
            ind = ind+1
            new = np.vstack(np.sign(xt[i,:]))+0.0001
            D = np.append(D, new, axis=1)
        # for the modulus operator
        for i in range(len(xt)):
            ind = ind+1
            new = np.vstack(abs(xt[i,:]))
            D = np.append(D, new, axis=1)
        # for the tensor operator
        for i in range(len(xt)):
            for j in  range(len(xt)):
                ind = ind+1
                new = np.multiply(xt[i,:],abs(xt[j,:]))
                new = np.vstack(new)
                D = np.append(D, new, axis=1)
            
    if harmonic == 1:
        # for sin(x)
        for i in range(len(xt)):
            ind = ind+1
            new = np.vstack(np.sin(xt[i,:]))
            D = np.append(D, new, axis=1)
        # for cos(x)
        for i in range(len(xt)):
            ind = ind+1
            new = np.vstack(np.cos(xt[i,:]))
            D = np.append(D, new, axis=1)
    ind = len(D[0])
    
    return D, ind


"""
# Bayesian Interference:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def BayInt(D, xdt):
    # for the dictionary:
    muD = np.mean(D,0)
    sdvD1 = np.std(D,0)
    sdvD = np.diag(sdvD1)
    Ds = np.dot((D - np.ones([len(D),1])*muD), LA.inv(sdvD))
    
    # for the observed data:
    muxdt = np.mean(xdt)
    xdts = np.vstack(xdt) - np.ones([len(D),1])*muxdt
    xdts = np.reshape(xdts, -1)
    
    return Ds, xdts, muD, sdvD

"""
# Residual variance:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def res_var(D, xdts):
    theta1 = np.dot(LA.pinv(D), xdts)
    error = xdts - np.matmul(D, theta1)
    err_var = np.var(error)
    
    return err_var

"""
# Initial latent vector finder:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def latent(nl, D, xdts):
    # Forward finder:
    zint = np.zeros(nl)
    theta = np.matmul(LA.pinv(D), xdts)
    index = np.array(np.where(zint != 0))[0]
    index = np.reshape(index,-1) # converting to 1-D array,
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    for i in range(0, nl):
        index = i
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint[index] = 1
        else:
            zint[index] = 0
    
    # Backward finder:
    index = np.array(np.where(zint != 0))
    index = np.reshape(index,-1) # converting to 1-D array,
    # gg = index.flatten()
    # gg = np.ravel(index)
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    ind = 0
    for i in range(nl-1, -1, -1):
        index = ind
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[ind+1] <= err[ind]:
            zint[index] = 1
        else:
            zint[index] = 0
        ind = ind + 1
    
    # for the states
    zint[[0, 1]] = [1, 1]
    return zint


"""
# For listing the library functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def library_list(xt, polyn, modfun, harmonic, theta):
    np.set_printoptions(precision=3)

    n = len(xt)
    D = ['1']
    
    if polyn >= 1:
        # poly order 1
        for i in range(len(xt)):
            D.append(f"{xt[i]}")
     
    if polyn >= 2: 
        # ploy order 2
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                D.append(f"{xt[i]}{xt[j]}")
    
    if polyn >= 3:    
        # ploy order 3
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    D.append(f"{xt[i]}{xt[j]}{xt[k]}")
    
    if polyn >= 4:
        # ploy order 4
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in range(k,len(xt)):
                        D.append(f"{xt[i]}{xt[j]}{xt[k]}{xt[l]}")
                        
    if polyn >= 5:
        # ploy order 5
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in  range(k,len(xt)):
                        for m in  range(l,len(xt)):
                            D.append(f"{xt[i]}{xt[j]}{xt[k]}{xt[l]}{xt[m]}")
    
    if polyn >= 6:
        # ploy order 6
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in  range(k,len(xt)):
                        for m in  range(l,len(xt)):
                            for n in  range(m,len(xt)):
                                D.append(f"{xt[i]}{xt[j]}{xt[k]}{xt[l]}{xt[m]}{xt[n]}")
                                
    if modfun == 1:
        # for the signum or sign operator
        for i in range(len(xt)):
            D.append( ('sign(').__add__(str(xt[i])).__add__(')') )
        
        # for the modulus operator
        for i in range(len(xt)):
            D.append( ('|').__add__(str(xt[i])).__add__('|') )
          
        # for the tensor operator
        for i in range(len(xt)):
            for j in  range(len(xt)):
                D.append( xt[i].__add__('|').__add__(str(xt[j])).__add__('|') )
            
    if harmonic == 1:
        # for sin(x)
        for i in range(len(xt)):
            D.append( ('sin(').__add__(str(xt[i])).__add__(')') )
            
        # for cos(x)
        for i in range(len(xt)):
            D.append( ('cos(').__add__(str(xt[i])).__add__(')') )
    
    pstrout = ['Functions']
    for i in range(n):
        pstrout.append((xt[i]).__add__('dot'))
    
    df = pd.DataFrame(columns=pstrout)
    for i in range(len(D)):
        df.loc[i] = [D[i]] + list(np.round(theta[i,:], 3))
        
    print(df)
    