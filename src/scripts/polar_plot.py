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

# load the posteriors from the MCMC and calculate psi dists
def polar_dist_draws(model_name):

	istar_idata = az.from_netcdf(paths.data / "polar" / (model_name + "_istar.nc"))
	noistar_idata = az.from_netcdf(paths.data / "polar" / (model_name + "_noistar.nc"))

	post = istar_idata.posterior
	istar_draws0 = np.zeros(shape=(len(x),4000))
	istar_draws1 = np.zeros(shape=(len(x),4000))
	for a in range(4):
		for b in range(1000):
			istar_draws0[:, a*1000+b] = post.w[a,b,0].values*beta.pdf(x, post.a[a,b,0], post.b[a,b,0])
			istar_draws1[:, a*1000+b] = post.w[a,b,1].values*beta.pdf(x, post.a[a,b,1], post.b[a,b,1])
	
	post = noistar_idata.posterior
	noistar_draws0 = np.zeros(shape=(len(x),4000))
	noistar_draws1 = np.zeros(shape=(len(x),4000))
	for a in range(4):
		for b in range(1000):
			noistar_draws0[:, a*1000+b] = post.w[a,b,0].values*beta.pdf(x, post.a[a,b,0], post.b[a,b,0])
			noistar_draws1[:, a*1000+b] = post.w[a,b,1].values*beta.pdf(x, post.a[a,b,1], post.b[a,b,1])

	return istar_draws0, istar_draws1, noistar_draws0, noistar_draws1

istar_draws0, istar_draws1, noistar_draws0, noistar_draws1= polar_dist_draws("polar")

fig = plt.figure(figsize=(5,2.4),dpi=110)

plt.subplot(1,2,1)

q025, q16, q50, q84, q975 = np.percentile(noistar_draws0, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '-.', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(noistar_draws1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '--', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(noistar_draws0+noistar_draws1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='C0', lw=1.5)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.ylabel('Probablity density function')

plt.title(r'measured $\vb*{\lambda}$ only', fontsize=11)

plt.subplot(1,2,2)

q025, q16, q50, q84, q975 = np.percentile(istar_draws0, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '-.', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(istar_draws1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, '--', c='k', lw=1., zorder=10)

q025, q16, q50, q84, q975 = np.percentile(istar_draws0+istar_draws1, [2.5, 16, 50, 84, 97.5], axis=1)/2
plt.plot(2*x-1, q50, color='#f56e4a', lw=1.5)
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,1])
plt.xlim([-1,1])

plt.title(r'measured $\vb*{\lambda}$ and $\vb*{i_\star}$', fontsize=11)

plt.tight_layout()

fig.set_facecolor('w')

plt.savefig(paths.figures / "polar.pdf", bbox_inches="tight", dpi=600)
plt.close()