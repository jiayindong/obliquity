import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import sys
import subprocess

import paths
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as at
import numpy as np
import arviz as az
from scipy.stats import beta

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=True)

plt.rcParams['xtick.top'] =  True
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.major.width'] =  1.0
plt.rcParams['xtick.minor.width'] =  1.0
plt.rcParams['ytick.right'] =  True
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] =  1.0
plt.rcParams['lines.markeredgewidth'] =  1.0

# plot the cospsi distribution
all_noistar = az.from_netcdf(paths.data / "all_noistar.nc")

x = np.linspace(1e-5,1-1e-5,1000)

post = all_noistar.posterior
comp0 = np.zeros(shape=(len(x),4000))
comp1 = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        comp0[:, a*1000+b] = post.w[a,b,0].values*beta.pdf(x, post.μ[a,b,0].values*post.κ[a,b,0].values, (1-post.μ[a,b,0]).values*post.κ[a,b,0].values)
        comp1[:, a*1000+b] = post.w[a,b,1].values*beta.pdf(x, post.μ[a,b,1].values*post.κ[a,b,1].values, (1-post.μ[a,b,1]).values*post.κ[a,b,1].values)

# make the figure
fig = plt.figure(figsize=(3.5,2.7),dpi=150)

q025, q16, q50, q84, q975 = np.percentile(comp0, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '-.', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(comp1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '--', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(comp0+comp1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.5)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1.])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.ylabel('Probablity density function')

fig.set_facecolor('w')

plt.tight_layout()
plt.savefig(paths.figures / "psi_dist.pdf", bbox_inches="tight", dpi=600)
plt.close()