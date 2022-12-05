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

numistar = 62
numall = 161

# load MCMC draws
lam_all = az.from_netcdf(paths.data / 'lam_all.nc')
lam_istar = az.from_netcdf(paths.data / 'lam_istar.nc')

x = np.linspace(1e-5,1-1e-5,1000)

post = lam_all.posterior
lam_all_draws = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        lam_all_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                    +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

post = lam_istar.posterior
lam_istar_draws = np.zeros(shape=(len(x),4000))
for a in range(4):
    for b in range(1000):
        lam_istar_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                    +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

# make the figure
plt.figure(figsize=(5,2.5),dpi=110)

plt.subplot(1,2,1)
q025, q16, q50, q84, q975 = np.percentile(lam_istar_draws, [2.5, 16, 50, 84, 97.5], axis=1)
plt.plot(x*180, q50, color='#f56e4a')
plt.fill_between(x*180, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
plt.fill_between(x*180, q025, q975, alpha=0.3, color='#fbc1ad')

plt.ylim([0,2])
plt.xlim([0,180])

plt.xticks(np.arange(0,185,30))


plt.xlabel(r'$\lambda$ [$^\circ$]',fontsize=11)
plt.ylabel('Probablity density function')

plt.title(r'%i systems with observed $i_{\star}$'%numistar,fontsize=11)

plt.subplot(1,2,2)
q025, q16, q50, q84, q975 = np.percentile(lam_all_draws, [2.5, 16, 50, 84, 97.5], axis=1)
plt.plot(x*180, q50, color='C0')
plt.fill_between(x*180, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
plt.fill_between(x*180, q025, q975, alpha=0.3, color='#7dabd0')

plt.ylim([0,2])
plt.xlim([0,180])

plt.xticks(np.arange(0,185,30))

plt.title(r'all %i systems'%numall, fontsize=11)

plt.tight_layout()
plt.savefig(paths.figures / "lam_dist.pdf", bbox_inches="tight", dpi=600)
plt.close()
