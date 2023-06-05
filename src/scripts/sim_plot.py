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
    #nolam_idata = az.from_netcdf(paths.data / "simulation" / (model_name + "_nolam.nc"))

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

    # post = nolam_idata.posterior
    # nolam_draws = np.zeros(shape=(len(x),16000))
    # for a in range(16):
    #     for b in range(1000):
    #         nolam_draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    return draws, noistar_draws

uni_draws, uni_noistar_draws = psi_dist_draws("uni")
norm1_draws, norm1_noistar_draws = psi_dist_draws("norm1")
norm2_draws, norm2_noistar_draws = psi_dist_draws("norm2")
norm3_draws, norm3_noistar_draws = psi_dist_draws("norm3")
beta_draws, beta_noistar_draws = psi_dist_draws("beta")

### Make the plot ###
fig, big_axes = plt.subplots(figsize=(3.5,8),dpi=150,nrows=5,ncols=1,sharey=True) 

for row, big_ax in enumerate(big_axes, start=1):
    if row == 5:
        big_ax.set_title(r"$\cos{\psi} \sim \mathrm{Beta}(3,6)$" "\n")
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

# cosψ ~ U(-1,1), noistar
ax = fig.add_subplot(5,2,1)
        
q025, q16, q50, q84, q975 = np.percentile(uni_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

# cosψ ~ U(-1,1), with istar
ax = fig.add_subplot(5,2,2)

q025, q16, q50, q84, q975 = np.percentile(uni_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

# cosψ ~ Beta(3,6), no istar
ax = fig.add_subplot(5,2,3)
        
q025, q16, q50, q84, q975 = np.percentile(beta_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(0,1,200)
pcosψ = stats.beta.pdf(cosψ, 3, 6)/2
plt.plot(2*cosψ-1, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


# cosψ ~ Beta(3,6), with istar
ax = fig.add_subplot(5,2,4)

q025, q16, q50, q84, q975 = np.percentile(beta_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,1.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(0,1,200)
pcosψ = stats.beta.pdf(cosψ, 3, 6)/2
plt.plot(2*cosψ-1, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


# cosψ ~ N(0,0.2), noistar
ax = fig.add_subplot(5,2,5)
        
q025, q16, q50, q84, q975 = np.percentile(norm1_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


# cosψ ~ N(0,0.2), with istar
ax = fig.add_subplot(5,2,6)

q025, q16, q50, q84, q975 = np.percentile(norm1_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


# cosψ ~ N(-0.4,0.2), noistar
ax = fig.add_subplot(5,2,7)

q025, q16, q50, q84, q975 = np.percentile(norm2_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(-0.4,0.2), with istar
ax = fig.add_subplot(5,2,8)

q025, q16, q50, q84, q975 = np.percentile(norm2_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)


cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0.4,0.2), no istar
ax = fig.add_subplot(5,2,9)
        
q025, q16, q50, q84, q975 = np.percentile(norm3_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ only', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

# cosψ ~ N(0.4,0.2), with istar
ax = fig.add_subplot(5,2,10)

q025, q16, q50, q84, q975 = np.percentile(norm3_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.5])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

fig.set_facecolor('w')

plt.tight_layout(pad=0,w_pad=0.5, h_pad=-2)

plt.savefig(paths.figures / "simulation.pdf", bbox_inches="tight", dpi=600)
plt.close()