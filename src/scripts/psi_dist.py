import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'arviz'])

import paths
import matplotlib.pyplot as plt
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

# load MCMC draws
all_randinc = az.from_netcdf('src/data/all_randinc.nc')
istar_randinc = az.from_netcdf('src/data/istar_randinc.nc')
istar = az.from_netcdf('src/data/istar.nc')

x = np.linspace(1e-5,1-1e-5,1000)

post = istar.posterior
istar_draws = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        istar_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                    +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

post = istar_randinc.posterior
istar_randinc_draws = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        istar_randinc_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                            +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

post = all_randinc.posterior
all_randinc_draws = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        all_randinc_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                          +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

# make the figure
plt.figure(figsize=(7,2.5),dpi=110)

plt.subplot(1,3,1)
q025, q16, q50, q84, q975 = np.percentile(istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)
plt.plot(2*x-1, q50, color='#f56e4a')
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2])
plt.xlim([-1,1])

plt.xlabel(r'$\cos{\psi}$',fontsize=11)
plt.ylabel('Probablity density function')

plt.title(r'65 systems -- informative $i_{\star}$')


plt.subplot(1,3,2)
q025, q16, q50, q84, q975 = np.percentile(istar_randinc_draws, [2.5, 16, 50, 84, 97.5], axis=1)
plt.plot(2*x-1, q50, color='#69a9a3')
plt.fill_between(2*x-1, q16, q84, alpha=0.5, label="posterior", color='#cfe7ea')
plt.fill_between(2*x-1, q025, q975, alpha=0.5, color='#cfe7ea')

plt.ylim([0,2])
plt.xlim([-1,1])

plt.title(r'65 systems -- random $i_{\star}$')


plt.subplot(1,3,3)
q025, q16, q50, q84, q975 = np.percentile(all_randinc_draws, [2.5, 16, 50, 84, 97.5], axis=1)
plt.plot(2*x-1, q50, color='C0')
plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2])
plt.xlim([-1,1])

plt.title(r'161 systems -- random $i_{\star}$')

plt.tight_layout()
plt.savefig(paths.figures / "psi_dist.pdf", bbox_inches="tight", dpi=600)
