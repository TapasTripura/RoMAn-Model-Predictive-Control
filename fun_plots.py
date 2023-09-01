# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Robust model agnostic predictive control algorithm for randomly excited
    dynamical systems, Probabilistic Engineering Mechanics.
-- Tapas Tripura, Souvik Chakraborty, IIT Delhi.
   
    This code contains the figure plot settings.
"""

import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 32

"""
Plot for LOTKA-VOTERA system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# System response
def plot_long(dxdata, y1, y2, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1)
    ax1.plot(t, y2)
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size');
    ax1.legend(['Prey','Predator']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(y1, y2)
    ax2.set_xlabel('Prey'); ax2.set_ylabel('Predator')
    ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=20, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.margins(0); ax2.grid(True);
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, dxdata[0])
    ax3.plot(time, dxdata[1])
    ax3.margins(0); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Linear Variation, |X(t)-z|')
    ax3.legend(['Prey','Predator']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(time, dxdata[2])
    ax4.plot(time, dxdata[3])
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax4.margins(0); ax4.legend(['Prey','Predator']); ax4.grid(True);
    plt.show()  
    return figure

# Plot for training data
def plot_data(dxdata, y1, y2, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, np.mean(np.array(y1[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y2[0]), axis = 0))
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size');
    ax1.legend(['Prey','Predator']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, np.mean(np.array(y1[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y2[1]), axis = 0))
    ax2.margins(0); 
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Population Size');
    ax2.legend(['Prey','Predator']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, dxdata[0][0])
    ax3.plot(time, dxdata[0][1])
    ax3.margins(0); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Linear Variation, |X(t)-z|')
    ax3.legend(['Prey','Predator']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(time, dxdata[1][0])
    ax4.plot(time, dxdata[1][1])
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax4.margins(0); ax4.legend(['Prey','Predator']); ax4.grid(True);
    plt.show()
    return figure

# Basis function identification results
def lotka_results(Zmean, theta, index):
    # index : 1 for drift identification plot
    #         2 for diffusion identification plot
    
    if index == 1:
        figure = plt.figure(figsize =([16, 10]))
        gs = gridspec.GridSpec(2, 2)
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
        # ax1.margins(0)
    
        ax2 = plt.subplot(gs[1, 0])
        xy = np.vstack([theta[0][1,:], theta[0][4,:]])
        z = gaussian_kde(xy)(xy)
        ax2.scatter(theta[0][1,:], theta[0][4,:], c=z, s=100)
        ax2.set_xlabel(' 'r'$\theta (x)$', fontweight='bold'); 
        ax2.set_ylabel(' 'r'$\theta (xy)$', fontweight='bold');
        ax2.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax2.set_title('(a) Prey', fontweight='bold')
        ax2.grid(True); 
        # ax2.margins(0)
    
        ax3 = plt.subplot(gs[1, 1])
        xy = np.vstack([theta[1][2,:], theta[1][4,:]])
        z = gaussian_kde(xy)(xy)
        ax3.scatter(theta[1][2,:], theta[1][4,:], c=z, s=100)
        ax3.set_xlabel(' 'r'$\theta (y)$', fontweight='bold'); 
        ax3.set_ylabel(' 'r'$\theta (xy)$', fontweight='bold');
        ax3.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax3.set_title('(b) Predator', fontweight='bold')
        ax3.grid(True); 
        # ax3.margins(0)
        plt.show()
        
    elif index == 2:
        figure = plt.figure(figsize =([16, 10]))
        gs = gridspec.GridSpec(2, 2)
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
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2=sns.distplot(theta[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel('Diffusion- 'r'$\theta (x_1^2)$', fontweight='bold'); 
        ax2.set_title('(b)', fontweight='bold');
        ax2.grid(True); plt.xlim([0,0.1])

        ax3 = plt.subplot(gs[1, 1])
        ax3=sns.distplot(theta[1][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax3.set_xlabel('Diffusion- 'r'$\theta (x_2^2)$', fontweight='bold'); 
        ax3.set_title('(c)', fontweight='bold');
        ax3.grid(True); plt.xlim([0.02,0.06])
        plt.show()
    return figure

# Plot for verification of identified system:
def lotka_verify(y1res, y2res, xtrain1, xtrain2, t):
    figure = plt.figure(figsize =([14, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[:, 0])
    ax1.plot(t, y1res); ax1.plot(t, xtrain1, '--'); 
    ax1.plot(t, y2res); ax1.plot(t, xtrain2, '-.');
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size')
    ax1.legend(['Prey(True)','Predicted(True)','Prey(Pred)','Predicted(Pred)'], prop={"size":24})
    ax1.grid(True)
    ax1.margins(0)
    
    ax3 = plt.subplot(gs[:, 1])
    ax3.plot(y1res, y2res)
    ax3.plot(xtrain1, xtrain2, '--')
    ax3.set_xlabel('Prey'); ax3.set_ylabel('Predator')
    ax3.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=24, bbox=dict(facecolor='w', alpha=0.5),ha='right')
    ax3.legend(['Actual','Predicted'])
    ax3.margins(0)
    ax3.grid(True)
    plt.show()
    return figure

# Plot for prediction using identified system:
def lotka_predict(y1res, y2res, y1val, y2val, xpred1, xpred2, t, tval):
    figure = plt.figure(figsize =([12, 8]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[:, 0])
    # Training phase
    ax1.plot(t, y1res, 'grey'); 
    ax1.plot(t, y2res, 'grey');
    # Actual prediction
    ax1.plot(tval, y1val); 
    ax1.plot(tval, y2val);
    # Prediction
    ax1.plot(tval, xpred1, 'r--');
    ax1.plot(tval, xpred2, 'b--');
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Population Size')
    ax1.legend(['Prey(train)','Predator(train)','Prey(true)','Predator(true)', \
                'Prey(predt)','Predator(predt)'], prop={"size":16})
    ax1.axvline(x= 100, color='k', linestyle=':')
    ax1.margins(0)
    
    ax2 = plt.subplot(gs[:, 1])
    ax2.plot(y1res, y2res, 'grey')
    ax2.plot(y1val, y2val, 'orange')
    ax2.plot(xpred1, xpred2, 'b--')
    ax2.set_xlabel('Prey'); ax2.set_ylabel('Predator')
    ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    plt.show()
    return figure


"""
Plot for LORENZ system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# System response
def plot_lorenzlong(dxdata, y1, y2, y3, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([20, 20]))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1)
    ax1.plot(t, y2)
    ax1.plot(t, y3)
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System States');
    ax1.legend(['X','Y','Z']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, dxdata[0])
    ax2.plot(time, dxdata[1])
    ax2.plot(time, dxdata[2])
    ax2.margins(0); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Linear Variation, |X(t)-z|')
    ax2.legend(['X','Y','Z']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(time, dxdata[3])
    ax3.plot(time, dxdata[4])
    ax3.plot(time, dxdata[5])
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax3.margins(0); ax3.legend(['X','Y','Z']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(y1, y2)
    ax4.set_xlabel('X'); ax4.set_ylabel('Y')
    # ax4.text(-8.458, -8.485, "Stable Point\n(-8.458, -8.485)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='left')
    ax4.text(8.458, 8.485, "Stable Point\n(8.458, 8.485)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='right')
    ax4.margins(0); ax4.grid(True);
    
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(y2, y3)
    ax5.set_xlabel('Y'); ax5.set_ylabel('Z')
    # ax5.text(-8.485, 27, "Stable Point\n(-8.485, 27)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='right')
    ax5.text(8.485, 27, "Stable Point\n(8.485, 27)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='right')
    ax5.margins(0); ax5.grid(True);
    
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(y1, y3)
    ax6.set_xlabel('X'); ax6.set_ylabel('Z')
    # ax6.text(-8.485, 27, "Stable Point\n(-8.485, 27)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='right')
    ax6.text(8.485, 27, "Stable Point\n(8.485, 27)", fontsize=20, bbox=dict(facecolor='g', alpha=0.2),ha='right')
    ax6.margins(0); ax6.grid(True);
    plt.show()  
    return figure

# Plot for training data
def plot_lorenzdata(dxdata, y1, y2, y3, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, np.mean(np.array(y1[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y2[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y3[0]), axis = 0))
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System States');
    ax1.legend(['X','Y','Z']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, np.mean(np.array(y1[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y2[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y3[1]), axis = 0))
    ax2.margins(0); 
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('System States');
    ax2.legend(['X','Y','Z']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, dxdata[0][0])
    ax3.plot(time, dxdata[0][1])
    ax3.plot(time, dxdata[0][2])
    ax3.margins(0); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Linear Variation, |X(t)-z|')
    ax3.legend(['X','Y','Z']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(time, dxdata[1][0])
    ax4.plot(time, dxdata[1][1])
    ax4.plot(time, dxdata[1][2])
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax4.margins(0); ax4.legend(['X','Y','Z']); ax4.grid(True);
    plt.show()
    return figure

# Basis function identification results
def lorenz_results(Zmean, theta, index):
    # index : 1 for drift identification plot
    #         2 for diffusion identification plot
    
    if index == 1:
        figure = plt.figure(figsize =([20, 14]))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace = 0.3, hspace = 0.35)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='blue', basefmt="k")
        ax1.stem(xr+0.1, Zmean[1], use_line_collection = True, linefmt='red', basefmt="k", markerfmt ='rD')
        ax1.stem(xr+0.15, Zmean[2], use_line_collection = True, linefmt='green', basefmt="k", markerfmt ='gD')
        ax1.axhline(y= 0.5, color='r', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', 'X', 'Y', 'Z'])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
    
        ax2 = plt.subplot(gs[1, 0])
        xy = np.vstack([theta[0][1,:], theta[0][2,:]])
        z = gaussian_kde(xy)(xy)
        ax2.scatter(theta[0][1,:], theta[0][2,:], c=z, s=100)
        ax2.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax2.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        ax2.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax2.set_title('(a) First state', fontweight='bold')
        ax2.grid(True); 
        # ax2.margins(0)
    
        ax3 = plt.subplot(gs[1, 1])
        xy = np.vstack([theta[1][1,:], theta[1][2,:]])
        z = gaussian_kde(xy)(xy)
        ax3.scatter(theta[1][1,:], theta[1][2,:], c=z, s=100)
        ax3.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax3.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        ax3.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax3.set_title('(b) Second state', fontweight='bold')
        ax3.grid(True); 
        # ax3.margins(0)
        
        ax4 = plt.subplot(gs[1, 2])
        xy = np.vstack([theta[1][1,:], theta[1][6,:]])
        z = gaussian_kde(xy)(xy)
        ax4.scatter(theta[1][1,:], theta[1][6,:], c=z, s=100)
        ax4.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax4.set_ylabel(' 'r'$\theta ({x_1}{x_3})$', fontweight='bold');
        ax4.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax4.set_title('(c) Second state', fontweight='bold')
        ax4.grid(True); 
        # ax3.margins(0)
        
        ax5 = plt.subplot(gs[2, 0])
        xy = np.vstack([theta[1][2,:], theta[1][6,:]])
        z = gaussian_kde(xy)(xy)
        ax5.scatter(theta[1][2,:], theta[1][6,:], c=z, s=100)
        ax5.set_xlabel(' 'r'$\theta (x_2)$', fontweight='bold'); 
        ax5.set_ylabel(' 'r'$\theta ({x_1}{x_3})$', fontweight='bold');
        ax5.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax5.set_title('(d) Second state', fontweight='bold')
        ax5.grid(True); 
        # ax3.margins(0)
        
        ax6 = plt.subplot(gs[2, 1])
        xy = np.vstack([theta[2][3,:], theta[2][5,:]])
        z = gaussian_kde(xy)(xy)
        ax6.scatter(theta[2][3,:], theta[2][5,:], c=z, s=100)
        ax6.set_xlabel(' 'r'$\theta (x_3)$', fontweight='bold'); 
        ax6.set_ylabel(' 'r'$\theta ({x_1}{x_2})$', fontweight='bold');
        ax6.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax6.set_title('(e) Third state', fontweight='bold')
        ax6.grid(True); 
        # ax3.margins(0)
        plt.show()
        
    elif index == 2:
        figure = plt.figure(figsize =([16, 10]))
        gs = gridspec.GridSpec(2, 3)
        gs.update(wspace = 0.3, hspace = 0.35)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='blue', basefmt="k")
        ax1.stem(xr+0.1, Zmean[1], use_line_collection = True, linefmt='red', basefmt="k", markerfmt ='rD')
        ax1.stem(xr+0.15, Zmean[2], use_line_collection = True, linefmt='green', basefmt="k", markerfmt ='gD')
        ax1.axhline(y= 0.5, color='r', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', 'X', 'Y', 'Z'])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2=sns.distplot(theta[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel('Diffusion- 'r'$\theta (x_1^2)$', fontweight='bold'); 
        ax2.set_title('(b)', fontweight='bold');
        ax2.grid(True); #plt.xlim([0,0.1])

        ax3 = plt.subplot(gs[1, 1])
        ax3=sns.distplot(theta[1][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax3.set_xlabel('Diffusion- 'r'$\theta (x_2^2)$', fontweight='bold'); 
        ax3.set_title('(c)', fontweight='bold');
        ax3.grid(True); #plt.xlim([0.02,0.06])
        
        ax4 = plt.subplot(gs[1, 2])
        ax4=sns.distplot(theta[2][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax4.set_xlabel('Diffusion- 'r'$\theta (x_3^2)$', fontweight='bold'); 
        ax4.set_title('(d)', fontweight='bold');
        ax4.grid(True); #plt.xlim([0.02,0.06])
        plt.show()
    return figure

# Plor for verification of identified system:
def lorenz_verify(y1res, y2res, y3res, xtrain1, xtrain2, xtrain3, t):
    figure = plt.figure(figsize =([10, 10]))
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1res, 'grey');     ax1.plot(t, xtrain1, 'r--'); 
    ax1.margins(0); ax1.grid(True)
    ax1.set_ylabel('$X$(t)')
    ax1.legend(['Actual','Identified'], prop={"size":20})
    ax1.set_xlim([0,30])
    
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(t, y2res, 'grey'); ax2.plot(t, xtrain2, 'g--');
    ax2.margins(0); ax2.grid(True)
    ax2.set_ylabel('$Y$(t)')
    ax2.legend(['Actual','Identified'], prop={"size":20})
    ax2.set_xlim([0,30])
    
    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(t, y3res, 'grey'); ax3.plot(t, xtrain3, 'b--');
    ax3.margins(0); ax3.grid(True)
    ax3.set_ylabel('$Z$(t)')
    ax3.legend(['Actual','Identified'], prop={"size":20})
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim([0,30])
    plt.show()
    return figure

# Plot for prediction of identified system:
def lorenz_predict(y1res, y2res, y3res, y1val, y2val, y3val, xpred1, xpred2, xpred3, t, tval):
    figure = plt.figure(figsize =([12, 8]))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, :])
    # Training phase
    ax1.plot(t, y1res, 'grey'); 
    ax1.plot(t, y2res, 'grey');
    ax1.plot(t, y3res, 'grey');
    # Actual prediction
    ax1.plot(tval, y1val); 
    ax1.plot(tval, y2val);
    ax1.plot(tval, y3val);
    # Prediction
    ax1.plot(tval, xpred1, 'r--');
    ax1.plot(tval, xpred2, 'b--');
    ax1.plot(tval, xpred3, 'g--');
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System states')
    ax1.legend(['X(train)','Y(train)','Z(train)','X(true)','Y(true)','Z(true)', \
                'X(predt)','Y(predt)','Z(predt)'], prop={"size":16})
    ax1.axvline(x= 100, color='k', linestyle=':')
    ax1.margins(0)
    
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(y1res, y2res, 'grey')
    ax2.plot(y1val, y2val, 'orange')
    ax2.plot(xpred1, xpred2, 'b--')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    # ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    ax2 = plt.subplot(gs[1, 1])
    ax2.plot(y1res, y3res, 'grey')
    ax2.plot(y1val, y3val, 'orange')
    ax2.plot(xpred1, xpred3, 'b--')
    ax2.set_xlabel('X'); ax2.set_ylabel('Z')
    # ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    ax2 = plt.subplot(gs[1, 2])
    ax2.plot(y2res, y3res, 'grey')
    ax2.plot(y2val, y3val, 'orange')
    ax2.plot(xpred2, xpred3, 'b--')
    ax2.set_xlabel('Y'); ax2.set_ylabel('Z')
    # ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    plt.show()
    return figure


"""
Plot for TMD system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# System response
def plot_TMDlong(dxdata, y1, y2, y3, y4, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace = 0.5, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y2)
    ax1.plot(t, y1)
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('SDOF system');
    ax1.legend(['$X_s$','$\dot{X}_s$']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, y4)
    ax2.plot(t, y3)
    ax2.margins(0); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('TMD');
    ax2.legend(['$X_d$','$\dot{X}_d$']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(y1, y2)
    ax3.set_xlabel('$X_s$'); ax3.set_ylabel('$\dot{X}_s$')
    ax3.margins(0); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(y3, y4)
    ax4.set_xlabel('$X_d$'); ax4.set_ylabel('$\dot{X}_d$')
    ax4.margins(0); ax4.grid(True);
    
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(time, dxdata[3])
    ax5.plot(time, dxdata[2])
    ax5.plot(time, dxdata[1])
    ax5.plot(time, dxdata[0])
    ax5.margins(0); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Linear Variation, |X(t)-z|')
    ax5.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax5.grid(True);
    
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(time, dxdata[4])
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax6.margins(0); ax6.legend(['$\dot{X}_s$']); ax6.grid(True);
    plt.show()  
    return figure

# Plot for training data
def plot_TMDdata(dxdata, y1, y2, y3, y4, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, np.mean(np.array(y1[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y2[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y3[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y4[0]), axis = 0))
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System States');
    ax1.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, np.mean(np.array(y1[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y2[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y3[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y4[1]), axis = 0))
    ax2.margins(0); 
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('System States');
    ax2.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, dxdata[0][0])
    ax3.plot(time, dxdata[0][1])
    ax3.plot(time, dxdata[0][2])
    ax3.plot(time, dxdata[0][3])
    ax3.margins(0); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Linear Variation, |X(t)-z|')
    ax3.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(time, dxdata[1])
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax4.margins(0); ax4.legend(['$\dot{X}_s$']); ax4.grid(True);
    plt.show()
    return figure

# Basis function identification results
def TMD_results(Zmean, theta, index):
    # index : 1 for drift identification plot
    #         2 for diffusion identification plot
    
    if index == 1:
        figure = plt.figure(figsize =([20, 20]))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace = 0.3, hspace = 0.45)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='brown', basefmt="k")
        ax1.stem(xr+0.1, Zmean[1], use_line_collection = True, linefmt='red', basefmt="k", markerfmt ='rD')
        ax1.stem(xr+0.15, Zmean[2], use_line_collection = True, linefmt='green', basefmt="k", markerfmt ='gD')
        ax1.stem(xr+0.2, Zmean[3], use_line_collection = True, linefmt='k', basefmt="k", markerfmt ='kD')
        ax1.axhline(y= 0.5, color='r', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', '$X_1$', '$\dot{X}_2$', '$X_3$', '$\dot{X}_4$'])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2=sns.distplot(theta[0][2,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel(' 'r'$\theta (x_2)$', fontweight='bold'); 
        ax2.set_title('(b) First state', fontweight='bold');
        ax2.grid(True); plt.xlim([0.5, 1.5]) 
        # ax3.margins(0)
    
        ax3 = plt.subplot(gs[1, 1])
        xy = np.vstack([theta[1][1,:], theta[1][2,:]])
        z = gaussian_kde(xy)(xy)
        ax3.scatter(theta[1][1,:], theta[1][2,:], c=z, s=100)
        ax3.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax3.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        # ax3.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax3.set_title('(c) Second state', fontweight='bold')
        ax3.grid(True); 
        # ax3.margins(0)
    
        ax4 = plt.subplot(gs[1, 2])
        xy = np.vstack([theta[1][3,:], theta[1][4,:]])
        z = gaussian_kde(xy)(xy)
        ax4.scatter(theta[1][3,:], theta[1][4,:], c=z, s=100)
        ax4.set_xlabel(' 'r'$\theta (x_3)$', fontweight='bold'); 
        ax4.set_ylabel(' 'r'$\theta (x_4)$', fontweight='bold');
        # ax4.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax4.set_title('(d) Second state', fontweight='bold')
        ax4.grid(True); 
        # ax4.margins(0)
        
        ax5 = plt.subplot(gs[2, 0])
        ax5=sns.distplot(theta[2][4,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax5.set_xlabel(' 'r'$\theta (x_4)$', fontweight='bold'); 
        ax5.set_title('(e) Third state', fontweight='bold');
        ax5.grid(True); plt.xlim([0.5, 1.5]) 
        # ax5.margins(0)
        
        ax6 = plt.subplot(gs[2, 1])
        xy = np.vstack([theta[3][1,:], theta[3][2,:]])
        z = gaussian_kde(xy)(xy)
        ax6.scatter(theta[3][1,:], theta[3][2,:], c=z, s=100)
        ax6.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax6.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        ax6.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax6.set_title('(f) Fourth state \n', fontweight='bold')
        ax6.grid(True); 
        # ax6.margins(0)
        
        ax7 = plt.subplot(gs[2, 2])
        xy = np.vstack([theta[3][3,:], theta[3][4,:]])
        z = gaussian_kde(xy)(xy)
        ax7.scatter(theta[3][3,:], theta[3][4,:], c=z, s=100)
        ax7.set_xlabel(' 'r'$\theta (x_3)$', fontweight='bold'); 
        ax7.set_ylabel(' 'r'$\theta (x_4)$', fontweight='bold');
        ax7.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax7.set_title('(g) Fourth state \n', fontweight='bold')
        ax7.grid(True); 
        # ax7.margins(0)
        
        plt.show()
        
    elif index == 2:
        figure = plt.figure(figsize =([16, 12]))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace = 0.3, hspace = 0.35)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='blue', basefmt="k")
        ax1.axhline(y= 0.5, color='r', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', '$\dot{X}_1$' ])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, :])
        ax2=sns.distplot(theta[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
        ax2.set_title('(b) Second state', fontweight='bold');
        ax2.grid(True); #plt.xlim([0,0.1])
        plt.show()
    return figure

# Plor for verification of identified system:
plt.rcParams['font.size'] = 24
    
def TMD_verify(y1res, y2res, y3res, y4res, xtrain1, xtrain2, xtrain3, xtrain4, t):
    figure = plt.figure(figsize =([14, 20]))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1res, '--', color='grey');     ax1.plot(t, xtrain1, 'r:'); 
    ax1.margins(0)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('$X_1$(t)')
    ax1.legend(['Actual','Identified'], prop={"size":22})
    ax1.set_xlim([0, 30])
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, y2res, '--', color='grey');     ax2.plot(t, xtrain2, 'b:');
    ax2.margins(0)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('$X_2$(t)')
    ax2.legend(['Actual','Identified'], prop={"size":22})
    ax2.set_xlim([0, 30])
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(t, y3res, '--', color='grey');     ax3.plot(t, xtrain3, 'g:');
    ax3.margins(0)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('$X_3$(t)')
    ax3.legend(['Actual','Identified'], prop={"size":22})
    ax3.set_xlim([0, 30])
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(t, y4res, '--', color='grey');     ax4.plot(t, xtrain4, 'm:');
    ax4.margins(0)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('$X_4$(t)')
    ax4.legend(['Actual','Identified'], prop={"size":22})
    ax4.set_xlim([0, 30])
    
    ax5 = plt.subplot(gs[2, 0])
    ax5.plot(y1res[:3000], y2res[:3000])
    ax5.plot(xtrain1[:3000], xtrain2[:3000], '--')
    ax5.set_xlabel('$X_1$(t)'); ax5.set_ylabel('$X_2$(t)')
    ax5.legend(['True','(Predict)'], prop={"size":22})
    ax5.margins(0)
    
    ax6 = plt.subplot(gs[2, 1])
    ax6.plot(y3res[:3000], y4res[:3000])
    ax6.plot(xtrain3[:3000], xtrain4[:3000], '--')
    ax6.set_xlabel('$X_3$(t)'); ax6.set_ylabel('$X_4$(t)')
    ax6.legend(['True','(Predict)'], prop={"size":22})
    ax6.margins(0)
    
    plt.show()
    return figure

# Plot for prediction of identified system:
def TMD_predict(y1res, y2res, y3res, y4res, y1val, y2val, y3val, y4val, xpred1, xpred2, xpred3, xpred4, t, tval):
    figure = plt.figure(figsize =([12, 8]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, :])
    # Training phase
    ax1.plot(t, y1res, 'grey'); 
    ax1.plot(t, y2res, 'grey');
    ax1.plot(t, y3res, 'grey');
    ax1.plot(t, y4res, 'grey');
    # Actual prediction
    ax1.plot(tval, y1val); 
    ax1.plot(tval, y2val);
    ax1.plot(tval, y3val);
    ax1.plot(tval, y4val);
    # Prediction
    ax1.plot(tval, xpred1, 'r--');
    ax1.plot(tval, xpred2, 'b--');
    ax1.plot(tval, xpred3, 'g--');
    ax1.plot(tval, xpred4, 'm--');
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System states')
    ax1.legend(['X(train)','Y(train)','Z(train)','X(true)','Y(true)','Z(true)', \
                'X(predt)','Y(predt)','Z(predt)'], prop={"size":16})
    # ax1.axvline(x= 100, color='k', linestyle=':')
    ax1.margins(0)
    
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(y1res, y2res, 'grey')
    ax2.plot(y1val, y2val, 'orange')
    ax2.plot(xpred1, xpred2, 'b--')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    # ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(y3res, y4res, 'grey')
    ax3.plot(y3val, y4val, 'orange')
    ax3.plot(xpred3, xpred4, 'b--')
    ax3.set_xlabel('X'); ax3.set_ylabel('Z')
    # ax3.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax3.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax3.margins(0)
    
    plt.show()
    return figure


"""
Plot for 76DOF ATMD system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# System response
def plot_TMD76long(dxdata, y1, y2, y3, y4, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace = 0.5, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1)
    ax1.plot(t, y2)
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('SDOF system');
    ax1.legend(['$X_s$','$\dot{X}_s$']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, y3)
    ax2.plot(t, y4)
    ax2.margins(0); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('TMD');
    ax2.legend(['$X_d$','$\dot{X}_d$']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(y1, y2)
    ax3.set_xlabel('$X_s$'); ax3.set_ylabel('$\dot{X}_s$')
    ax3.margins(0); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(y3, y4)
    ax4.set_xlabel('$X_d$'); ax4.set_ylabel('$\dot{X}_d$')
    ax4.margins(0); ax4.grid(True);
    
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(time, dxdata[0])
    ax5.plot(time, dxdata[1])
    ax5.plot(time, dxdata[2])
    ax5.plot(time, dxdata[3])
    ax5.margins(0); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Linear Variation, |X(t)-z|')
    ax5.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax5.grid(True);
    
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(time, dxdata[4])
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax6.margins(0); ax6.legend(['$\dot{X}_s$']); ax6.grid(True);
    plt.show()  
    return figure

# Plot for training data
def plot_TMD76data(dxdata, y1, y2, y3, y4, t):
    time = t[0:-1]
    
    figure = plt.figure(figsize =([16, 10]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, np.mean(np.array(y1[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y2[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y3[0]), axis = 0))
    ax1.plot(t, np.mean(np.array(y4[0]), axis = 0))
    ax1.margins(0); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System States');
    ax1.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax1.grid(True);
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, np.mean(np.array(y1[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y2[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y3[1]), axis = 0))
    ax2.plot(t, np.mean(np.array(y4[1]), axis = 0))
    ax2.margins(0); 
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('System States');
    ax2.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax2.grid(True);
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, dxdata[0][0])
    ax3.plot(time, dxdata[0][1])
    ax3.plot(time, dxdata[0][2])
    ax3.plot(time, dxdata[0][3])
    ax3.margins(0); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Linear Variation, |X(t)-z|')
    ax3.legend(['$X_s$','$\dot{X}_s$','$X_d$','$\dot{X}_d$']); ax3.grid(True);
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(time, dxdata[1])
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Quad Variation,\n|(X(t)-z)^2|')
    ax4.margins(0); ax4.legend(['$\dot{X}_s$']); ax4.grid(True);
    plt.show()
    return figure

def TMD76_verify(y1res, y2res, y3res, y4res, xtrain1, xtrain2, xtrain3, xtrain4, t, tval):
    figure = plt.figure(figsize =([14, 20]))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(t, y1res, '--', color='grey');     
    ax1.plot(tval, xtrain1, 'r:'); 
    ax1.margins(0)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('$X_1$(t)')
    ax1.legend(['Actual','Identified'], prop={"size":22})
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(t, y2res, '--', color='grey');     
    ax2.plot(tval, xtrain2, 'b:');
    ax2.margins(0)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('$X_2$(t)')
    ax2.legend(['Actual','Identified'], prop={"size":22})
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(t, y3res, '--', color='grey');     
    ax3.plot(tval, xtrain3, 'g:');
    ax3.margins(0)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('$X_3$(t)')
    ax3.legend(['Actual','Identified'], prop={"size":22})
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(t, y4res, '--', color='grey');     
    ax4.plot(tval, xtrain4, 'm:');
    ax4.margins(0)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('$X_4$(t)')
    ax4.legend(['Actual','Identified'], prop={"size":22})
    
    ax5 = plt.subplot(gs[2, 0])
    ax5.plot(y1res, y2res)
    ax5.plot(xtrain1, xtrain2, '--')
    ax5.set_xlabel('$X_1$(t)'); ax5.set_ylabel('$X_2$(t)')
    ax5.legend(['True','(Predict)'], prop={"size":22})
    ax5.margins(0)
    
    ax6 = plt.subplot(gs[2, 1])
    ax6.plot(y3res, y4res)
    ax6.plot(xtrain3, xtrain4, '--')
    ax6.set_xlabel('$X_3$(t)'); ax6.set_ylabel('$X_4$(t)')
    ax6.legend(['True','(Predict)'], prop={"size":22})
    ax6.margins(0)
    
    plt.show()
    return figure

# Basis function identification results
def TMD76_results(Zmean, theta, index):
    # index : 1 for drift identification plot
    #         2 for diffusion identification plot
    
    if index == 1:
        figure = plt.figure(figsize =([20, 20]))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace = 0.3, hspace = 0.45)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='green', basefmt="k", markerfmt ='gD')
        ax1.stem(xr+0.1, Zmean[1], use_line_collection = True, linefmt='red', basefmt="k", markerfmt ='rs')
        ax1.stem(xr+0.15, Zmean[2], use_line_collection = True, linefmt='blue', basefmt="k", markerfmt ='bD')
        ax1.stem(xr+0.2, Zmean[3], use_line_collection = True, linefmt='k', basefmt="k", markerfmt ='ks')
        ax1.axhline(y= 0.5, color='m', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', '$X_1$', '$\dot{X}_2$', '$X_3$', '$\dot{X}_4$'])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2=sns.distplot(theta[0][2,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel(' 'r'$\theta (x_2)$', fontweight='bold'); 
        ax2.set_title('(b) First state', fontweight='bold');
        ax2.grid(True); plt.xlim([0.5, 1.5]) 
        # ax3.margins(0)
    
        ax3 = plt.subplot(gs[1, 1])
        xy = np.vstack([theta[1][1,:], theta[1][3,:]])
        z = gaussian_kde(xy)(xy)
        ax3.scatter(theta[1][1,:], theta[1][3,:], c=z, s=100)
        ax3.set_xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
        ax3.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        # ax3.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax3.set_title('(c) Second state', fontweight='bold')
        ax3.grid(True); 
        # ax3.margins(0)
    
        # ax4 = plt.subplot(gs[1, 2])
        # xy = np.vstack([theta[1][3,:], theta[1][4,:]])
        # z = gaussian_kde(xy)(xy)
        # ax4.scatter(theta[1][3,:], theta[1][4,:], c=z, s=100)
        # ax4.set_xlabel(' 'r'$\theta (x_3)$', fontweight='bold'); 
        # ax4.set_ylabel(' 'r'$\theta (x_4)$', fontweight='bold');
        # # ax4.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        # ax4.set_title('(d) Second state', fontweight='bold')
        # ax4.grid(True); 
        # ax4.margins(0)
        
        ax5 = plt.subplot(gs[2, 0])
        ax5=sns.distplot(theta[2][4,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax5.set_xlabel(' 'r'$\theta (x_4)$', fontweight='bold'); 
        ax5.set_title('(e) Third state', fontweight='bold');
        ax5.grid(True); plt.xlim([0.5, 1.5]) 
        # ax5.margins(0)
        
        ax6 = plt.subplot(gs[2, 1])
        xy = np.vstack([theta[3][1,:], theta[3][2,:]])
        z = gaussian_kde(xy)(xy)
        ax6.scatter(theta[3][1,:], theta[3][2,:], c=z, s=100)
        ax6.set_xlabel(' \n 'r'$\theta (x_1)$', fontweight='bold'); 
        ax6.set_ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
        ax6.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax6.set_title('(f) Fourth state \n', fontweight='bold')
        ax6.grid(True); 
        # ax6.margins(0)
        
        ax7 = plt.subplot(gs[2, 2])
        xy = np.vstack([theta[3][3,:], theta[3][4,:]])
        z = gaussian_kde(xy)(xy)
        ax7.scatter(theta[3][3,:], theta[3][4,:], c=z, s=100)
        ax7.set_xlabel(' \n 'r'$\theta (x_3)$', fontweight='bold'); 
        ax7.set_ylabel(' 'r'$\theta (x_4)$', fontweight='bold');
        ax7.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        ax7.set_title('(g) Fourth state \n', fontweight='bold')
        ax7.grid(True); 
        # ax7.margins(0)
        
        plt.show()
        
    elif index == 2:
        figure = plt.figure(figsize =([16, 12]))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace = 0.3, hspace = 0.35)
        ax1 = plt.subplot(gs[0, :])
        xr = np.array(range(len(Zmean[0])))
        ax1.stem(xr, Zmean[0], use_line_collection = True, linefmt='blue', basefmt="k")
        ax1.axhline(y= 0.5, color='m', linestyle='-.')
        ax1.set_ylabel('PIP ', fontweight='bold');
        ax1.set_xlabel('Library Functions', fontweight='bold');
        ax1.set_title('(a)', fontweight='bold')
        ax1.legend(['PIP = 0.5', '$\dot{X}_1$' ])
        ax1.grid(True); plt.ylim(0,1.05)
        # ax1.margins(0)
        
        ax2 = plt.subplot(gs[1, :])
        ax2=sns.distplot(theta[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
        ax2.set_xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
        ax2.set_title('(b) Second state', fontweight='bold');
        ax2.grid(True); #plt.xlim([0,0.1])
        plt.show()
    return figure

# Plot for prediction of identified system:
def TMD76_predict(y1val, y2val, y3val, y4val, xpred1, xpred2, xpred3, xpred4, t, tval):
    figure = plt.figure(figsize =([12, 8]))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace = 0.25, hspace = 0.3)
    ax1 = plt.subplot(gs[0, :])
    # Actual prediction
    ax1.plot(tval, y1val); 
    ax1.plot(tval, y2val);
    ax1.plot(tval, y3val);
    ax1.plot(tval, y4val);
    # Prediction
    ax1.plot(tval, xpred1, 'r--');
    ax1.plot(tval, xpred2, 'b--');
    ax1.plot(tval, xpred3, 'g--');
    ax1.plot(tval, xpred4, 'm--');
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('System states')
    ax1.legend(['X(true)','Y(true)','Z(true)', \
                'X(predt)','Y(predt)','Z(predt)'], prop={"size":16})
    # ax1.axvline(x= 100, color='k', linestyle=':')
    ax1.margins(0)
    
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(y1val, y2val, 'orange')
    ax2.plot(xpred1, xpred2, 'b--')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    # ax2.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax2.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax2.margins(0)
    
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(y3val, y4val, 'orange')
    ax3.plot(xpred3, xpred4, 'b--')
    ax3.set_xlabel('X'); ax3.set_ylabel('Z')
    # ax3.text(100, 20, "Stable\nPoint\n(100,20)", fontsize=12, bbox=dict(facecolor='g', alpha=0.5),ha='right')
    ax3.legend(['Training', 'Actual', 'Prediction'], prop={"size":16})
    ax3.margins(0)
    
    plt.show()
    return figure
