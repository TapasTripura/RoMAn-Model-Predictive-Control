# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
This code performs the sparse Bayesian inference using the Spike and Slab prior
    and darws the samples from posterior using Gibbs sampler.
"""

import numpy as np
from numpy import linalg as LA
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
import fun_library
from numpy.random import multivariate_normal as mvrv
from scipy.special import loggamma as LG


"""
Theta: Multivariate Normal distribution
"""
def sigmu(z, D, vs, xdts):
    index = np.array(np.where(z != 0))
    index = np.reshape(index,-1) # converting to 1-D array, 
    Dr = D[:,index] 
    Aor = np.eye(len(index)) # independent prior
    # Aor = np.dot(len(Dr), LA.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = LA.inv(np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA.inv(Aor)))
    mu = np.matmul(np.matmul(BSIG,Dr.T),xdts)
    return mu, BSIG, Aor, index

"""
P(Y|zi=(0|1),z-i,vs)
"""
def pyzv(D, ztemp, vs, N, xdts, asig, bsig):
    rind = np.array(np.where(ztemp != 0))[0]
    rind = np.reshape(rind, -1) # converting to 1-D array,   
    Sz = sum(ztemp)
    Dr = D[:, rind] 
    Aor = np.eye(len(rind)) # independent prior
    # Aor = np.dot(N, LA.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA.inv(Aor))
    
    (sign, logdet0) = LA.slogdet(LA.inv(Aor))
    (sign, logdet1) = LA.slogdet(LA.inv(BSIG))
    
    PZ = LG(asig + 0.5*N) -0.5*N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
        + asig*np.log(bsig) - LG(asig) + 0.5*logdet0 + 0.5*logdet1
    denom1 = np.eye(N) - np.matmul(np.matmul(Dr, LA.inv(BSIG)), Dr.T)
    denom = (0.5*np.matmul(np.matmul(xdts.T, denom1), xdts))
    PZ = PZ - (asig+0.5*N)*(np.log(bsig + denom))
    return PZ

"""
P(Y|zi=0,z-i,vs)
"""
def pyzv0(xdts, N, asig, bsig):
    PZ0 = LG(asig + 0.5*N) - 0.5*N*np.log(2*np.pi) + asig*np.log(bsig) - LG(asig) \
        + np.log(1) - (asig+0.5*N)*np.log(bsig + 0.5*np.matmul(xdts.T, xdts))
    return PZ0


"""
Sparse regression with Normal Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def sparse(xdts, ydata, polyorder, modfun, harmonic, MCMC, burn_in):
    # Library creation:
    D, nl = fun_library.library(np.transpose(ydata), polyorder, modfun, harmonic)

    # Residual variance:
    err_var = fun_library.res_var(D, xdts)

    """
    # Gibbs sampling:
    """
    # Hyper-parameters
    ap, bp = 0.1, 1 # for beta prior for p0
    av, bv = 0.5, 0.5 # inverge gamma for vs
    asig, bsig = 1e-4, 1e-4 # invese gamma for sig^2

    # Parameter Initialisation:
    p0 = np.zeros(MCMC)
    vs = np.zeros(MCMC)
    sig = np.zeros(MCMC)
    p0[0] = 0.1
    vs[0] = 10
    sig[0] = err_var

    N = len(xdts)

    # Initial latent vector
    zval = np.zeros(nl)
    zint  = fun_library.latent(nl, D, xdts)
    zstore = np.transpose(np.vstack([zint]))
    zval = zint

    zval0 = zval
    vs0 = vs[0]
    mu, BSIG, Aor, index = sigmu(zval0, D, vs0, xdts)
    Sz = sum(zval)

    # Sample theta from Normal distribution
    thetar = mvrv(mu, np.dot(sig[0], BSIG))
    thetat = np.zeros(nl)
    thetat[index] = thetar
    theta = np.vstack(thetat)

    for i in range(1, MCMC):
        if i % 100 == 0:
            print('MCMC-', i)
        # sample z from the Bernoulli distribution:
        zr = np.zeros(nl) # instantaneous latent vector (z_i):
        zr = zval
        for j in range(nl):
            ztemp0 = zr
            ztemp0[j] = 0
            if np.mean(ztemp0) == 0:
                PZ0 = pyzv0(xdts, N, asig, bsig)
            else:
                vst0 = vs[i-1]
                PZ0 = pyzv(D, ztemp0, vst0, N, xdts, asig, bsig)
            
            ztemp1 = zr
            ztemp1[j] = 1      
            vst1 = vs[i-1]
            PZ1 = pyzv(D, ztemp1, vst1, N, xdts, asig, bsig)
            
            zeta = PZ0 - PZ1  
            zeta = p0[i-1]/( p0[i-1] + np.exp(zeta)*(1-p0[i-1]))
            zr[j] = bern(1, p = zeta, size = None)
        
        zval = zr
        zstore = np.append(zstore, np.vstack(zval), axis = 1)
        
        # sample sig^2 from inverse Gamma:
        asiggamma = asig+0.5*N
        temp = np.matmul(np.matmul(mu.T, LA.inv(BSIG)), mu)
        bsiggamma = bsig+0.5*(np.dot(xdts.T, xdts) - temp)
        sig[i] = 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        
        # sample vs from inverse Gamma:
        avvs = av+0.5*Sz
        bvvs = bv+(np.matmul(np.matmul(thetar.T, LA.inv(Aor)), thetar))/(2*sig[i])
        vs[i] = 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        
        # sample p0 from Beta distribution:
        app0 = ap+Sz
        bpp0 = bp+nl-Sz # Here, P=nl (no. of functions in library)
        p0[i] = beta(app0, bpp0)
        # or, np.random.beta()
        
        # Sample theta from Normal distribution:
        vstheta = vs[i]
        mu, BSIG, Aor, index = sigmu(zval, D, vstheta, xdts)
        Sz = sum(zval)
        thetar = mvrv(mu, np.dot(sig[i], BSIG))
        thetat = np.zeros(nl)
        thetat[index] = thetar
        theta = np.append(theta, np.vstack(thetat), axis = 1)

    # Marginal posterior inclusion probabilities (PIP):
    zstoredrift = zstore[:, burn_in:]
    Zmeandrift = np.mean(zstoredrift, axis=1)

    # Post processing:
    thetadrift = theta[:, burn_in:]
    mutdrift = np.mean(thetadrift, axis=1)
    sigtdrift = np.cov(thetadrift, bias = False)
    
    return zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift


"""
Sparse regression with Ensemble Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def sparse_stc(xdts, ydata, polyorder, modfun, harmonic, MCMC, burn_in):
        
    # Expected Dictionary Creation:
    if len(ydata) == 2:
        y1, y2 = ydata
        libr = []
        for j in range(len(y1)):
            data = np.row_stack((y1[j,:], y2[j,:]))
            Dtemp, nl = fun_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
        
    elif len(ydata) == 3:
        y1, y2, y3 = ydata
        libr = []
        for j in range(len(y1)):
            data = np.row_stack((y1[j,:], y2[j,:], y3[j,:]))
            Dtemp, nl = fun_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
    elif len(ydata) == 4:
        y1, y2, y3, y4 = ydata
        libr = []
        for j in range(len(y1)):
            data = np.row_stack((y1[j,:], y2[j,:], y3[j,:], y4[j,:]))
            Dtemp, nl = fun_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
    print('Library created...')
    
    # Residual variance:
    err_var = fun_library.res_var(D, xdts)
    
    """
    # Gibbs sampling:
    """
    # Hyper-parameters
    ap, bp = 0.1, 1 # for beta prior for p0
    av, bv = 0.5, 0.5 # inverge gamma for vs
    asig, bsig = 1e-4, 1e-4 # invese gamma for sig^2
    
    # Parameter Initialisation:
    p0 = np.zeros(MCMC)
    vs = np.zeros(MCMC)
    sig = np.zeros(MCMC)
    p0[0] = 0.1
    vs[0] = 10
    sig[0] = err_var
    
    N = len(xdts)
    
    # Initial latent vector
    zval = np.zeros(nl)
    zint  = fun_library.latent(nl, D, xdts)
    zstore = np.transpose(np.vstack([zint]))
    zval = zint
    
    zval0 = zval
    vs0 = vs[0]
    mu, BSIG, Aor, index = sigmu(zval0, D, vs0, xdts)
    Sz = sum(zval)
    
    # Sample theta from Normal distribution
    thetar = mvrv(mu, np.dot(sig[0], BSIG))
    thetat = np.zeros(nl)
    thetat[index] = thetar
    theta = np.vstack(thetat)
    
    for i in range(1, MCMC):
        if i % 50 == 0:
            print('MCMC-', i)
        # sample z from the Bernoulli distribution:
        zr = np.zeros(nl) # instantaneous latent vector (z_i):
        zr = zval
        for j in range(nl):
            ztemp0 = zr
            ztemp0[j] = 0
            if np.mean(ztemp0) == 0:
                PZ0 = pyzv0(xdts, N, asig, bsig)
            else:
                vst0 = vs[i-1]
                PZ0 = pyzv(D, ztemp0, vst0, N, xdts, asig, bsig)
            
            ztemp1 = zr
            ztemp1[j] = 1      
            vst1 = vs[i-1]
            PZ1 = pyzv(D, ztemp1, vst1, N, xdts, asig, bsig)
            
            zeta = PZ0 - PZ1  
            zeta = p0[i-1]/( p0[i-1] + np.exp(zeta)*(1-p0[i-1]))
            zr[j] = bern(1, p = zeta, size = None)
        
        zval = zr
        zstore = np.append(zstore, np.vstack(zval), axis = 1)
        
        # sample sig^2 from inverse Gamma:
        asiggamma = asig+0.5*N
        temp = np.matmul(np.matmul(mu.T, LA.inv(BSIG)), mu)
        bsiggamma = bsig+0.5*(np.dot(xdts.T, xdts) - temp)
        sig[i] = 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        
        # sample vs from inverse Gamma:
        avvs = av+0.5*Sz
        bvvs = bv+(np.matmul(np.matmul(thetar.T, LA.inv(Aor)), thetar))/(2*sig[i])
        vs[i] = 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        
        # sample p0 from Beta distribution:
        app0 = ap+Sz
        bpp0 = bp+nl-Sz # Here, P=nl (no. of functions in library)
        p0[i] = beta(app0, bpp0)
        # or, np.random.beta()
        
        # Sample theta from Normal distribution:
        vstheta = vs[i]
        mu, BSIG, Aor, index = sigmu(zval, D, vstheta, xdts)
        Sz = sum(zval)
        thetar = mvrv(mu, np.dot(sig[i], BSIG))
        thetat = np.zeros(nl)
        thetat[index] = thetar
        theta = np.append(theta, np.vstack(thetat), axis = 1)
    
    # Marginal posterior inclusion probabilities (PIP):
    zstoredrift = zstore[:, burn_in:]
    Zmeandrift = np.mean(zstoredrift, axis=1)
    
    # Post processing:
    thetadrift = theta[:, burn_in:]
    mutdrift = np.mean(thetadrift, axis=1)
    sigtdrift = np.cov(thetadrift, bias = False)
    
    return zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift
