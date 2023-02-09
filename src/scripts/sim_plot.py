import paths
import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import subprocess
import sys

import pymc as pm
import arviz as az
import aesara.tensor as at

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

x = np.linspace(1e-5,1-1e-5,1000)
σ = 0.2        

# load the distribution draws from MCMC
uni_istar_draws = np.load(paths.data / "uni_istar_draws.npy")
uni_noistar_draws = np.load(paths.data / "uni_noistar_draws.npy")

norm1_istar_draws = np.load(paths.data / "norm1_istar_draws.npy")
norm1_noistar_draws = np.load(paths.data / "norm1_noistar_draws.npy")

norm2_istar_draws = np.load(paths.data / "norm2_istar_draws.npy")
norm2_noistar_draws = np.load(paths.data / "norm2_noistar_draws.npy")

norm3_istar_draws = np.load(paths.data / "norm3_istar_draws.npy")
norm3_noistar_draws = np.load(paths.data / "norm3_noistar_draws.npy")

### Make the plot ###
fig, big_axes = plt.subplots(figsize=(3.5,6.5),dpi=110,nrows=4,ncols=1,sharey=True) 

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

### Uniform cosψ ###
ax = fig.add_subplot(4,2,1)
        
q025, q16, q50, q84, q975 = np.percentile(uni_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2

plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')


plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

ax = fig.add_subplot(4,2,2)

q025, q16, q50, q84, q975 = np.percentile(uni_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{no} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = np.ones_like(cosψ)*0.5
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)


### cosψ ~ Normal(0., 0.2) ###
ax = fig.add_subplot(4,2,3)
        
q025, q16, q50, q84, q975 = np.percentile(norm1_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

ax = fig.add_subplot(4,2,4)

q025, q16, q50, q84, q975 = np.percentile(norm1_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{no} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


### cosψ ~ Normal(-0.4, 0.2) ###
ax = fig.add_subplot(4,2,5)
        
q025, q16, q50, q84, q975 = np.percentile(norm2_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

ax = fig.add_subplot(4,2,6)

q025, q16, q50, q84, q975 = np.percentile(norm2_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{no} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)


### cosψ ~ Normal(0.4, 0.2) ###
ax = fig.add_subplot(4,2,7)
        
q025, q16, q50, q84, q975 = np.percentile(norm3_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

ax = fig.add_subplot(4,2,8)

q025, q16, q50, q84, q975 = np.percentile(norm3_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.2)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')


plt.ylim([0,2.1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.title(r'\textbf{no} $i_{\star}$ likelihood', fontsize=10)

cosψ = np.linspace(-1,1,200)
pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

fig.set_facecolor('w')

plt.tight_layout(pad=0,w_pad=0.5, h_pad=-2)

plt.savefig(paths.figures / "simulation.pdf", bbox_inches="tight", dpi=600)
plt.close()