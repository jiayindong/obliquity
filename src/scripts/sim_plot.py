import paths
import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import subprocess
import sys

import pymc as pm
import arviz as az
import pytensor.tensor as at

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.integrate as integrate
from scipy import stats

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{physics}')

plt.rcParams['xtick.top'] =  True
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.major.width'] =  1.0
plt.rcParams['xtick.minor.width'] =  1.0
plt.rcParams['ytick.right'] =  True
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] =  1.0
plt.rcParams['lines.markeredgewidth'] =  1.0

from scipy.stats import beta
x = np.linspace(1e-5,1-1e-5,1000)
σ = 0.2        

# load the posteriors from the MCMC and calculate psi dists
def psi_dist_draws(model_name):

    idata = az.from_netcdf(paths.data / "simulation" / (model_name + ".nc"))
    noistar_idata = az.from_netcdf(paths.data / "simulation" / (model_name + "_noistar.nc"))
    nolam_idata = az.from_netcdf(paths.data / "simulation" / (model_name + "_nolam.nc"))

    post = idata.posterior
    draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    post = noistar_idata.posterior
    noistar_draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            noistar_draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    post = nolam_idata.posterior
    nolam_draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            nolam_draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    return draws, noistar_draws, nolam_draws

uni_draws, uni_noistar_draws, uni_nolam_draws = psi_dist_draws("uni")
norm1_draws, norm1_noistar_draws, norm1_nolam_draws = psi_dist_draws("norm1")
norm2_draws, norm2_noistar_draws, norm2_nolam_draws = psi_dist_draws("norm2")
norm3_draws, norm3_noistar_draws, norm3_nolam_draws = psi_dist_draws("norm3")


### Make the plot ###
fig, big_axes = plt.subplots(figsize=(6.5,7),dpi=110,nrows=4,ncols=1,sharey=True) 

for row, big_ax in enumerate(big_axes, start=1):
    
    if row == 4:
        big_ax.set_title(r"$\cos{\psi} \sim \mathcal{N}(0.4,0.2)$" "\n")
    if row == 3:
        big_ax.set_title(r"$\cos{\psi} \sim \mathcal{N}(-0.4,0.2)$" "\n")
    if row == 2:
        big_ax.set_title(r"$\cos{\psi} \sim \mathcal{N}(0,0.2)$" "\n")
    elif row == 1:
        big_ax.set_title(r"$\cos{\psi} \sim \mathcal{U}(-1,1)$" "\n")

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

# cosψ ~ U(-1,1)
ax = fig.add_subplot(4,3,1)
        
q025, q16, q50, q84, q975 = np.percentile(uni_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

# cosψ ~ U(-1,1), noistar
ax = fig.add_subplot(4,3,2)

q025, q16, q50, q84, q975 = np.percentile(uni_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

# cosψ ~ U(-1,1), nolam
ax = fig.add_subplot(4,3,3)

q025, q16, q50, q84, q975 = np.percentile(uni_nolam_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{i_\star}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=10)


# cosψ ~ N(0,0.2)
ax = fig.add_subplot(4,3,4)
        
q025, q16, q50, q84, q975 = np.percentile(norm1_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0,0.2), noistar
ax = fig.add_subplot(4,3,5)

q025, q16, q50, q84, q975 = np.percentile(norm1_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0,0.2), nolam
ax = fig.add_subplot(4,3,6)

q025, q16, q50, q84, q975 = np.percentile(norm1_nolam_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{i_\star}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=10)


# cosψ ~ N(-0.4,0.2)
ax = fig.add_subplot(4,3,7)
        
q025, q16, q50, q84, q975 = np.percentile(norm2_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(-0.4,0.2), noistar
ax = fig.add_subplot(4,3,8)

q025, q16, q50, q84, q975 = np.percentile(norm2_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(-0.4,0.2), nolam
ax = fig.add_subplot(4,3,9)

q025, q16, q50, q84, q975 = np.percentile(norm2_nolam_draws[:,:1000], [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

q025, q16, q50, q84, q975 = np.percentile(norm2_nolam_draws[:,1000:2000], [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{i_\star}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=10)


# cosψ ~ N(0.4,0.2)
ax = fig.add_subplot(4,3,10)
        
q025, q16, q50, q84, q975 = np.percentile(norm3_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0.4,0.2), noistar
ax = fig.add_subplot(4,3,11)

q025, q16, q50, q84, q975 = np.percentile(norm3_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0.4,0.2), nolam
ax = fig.add_subplot(4,3,12)

q025, q16, q50, q84, q975 = np.percentile(norm3_nolam_draws[:,:1000], [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

q025, q16, q50, q84, q975 = np.percentile(norm3_nolam_draws[:,2000:3000], [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{i_\star}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=10)

fig.set_facecolor('w')

plt.tight_layout(pad=0,w_pad=0.5, h_pad=-2)

plt.savefig(paths.figures / "simulation.pdf", bbox_inches="tight", dpi=600)
plt.close()