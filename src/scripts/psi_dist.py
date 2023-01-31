import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'arviz'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pymc==4.1.7'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'aesara==2.8.2'])

import paths
import matplotlib.pyplot as plt
import pymc as pm
import aesara.tensor as at
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

# Sky-projected stellar obliquity and uncertainty extracted from Albrecht+22
Lam = np.array([8.20304715e-02, 5.23598776e-02, 1.74532925e-02, 6.56243772e-01,
       1.74532928e-03, 1.74532925e-01, 5.11381482e-02, 2.58308733e-02,
       1.25663706e+00, 6.45771832e-02, 3.49065856e-03, 3.70009815e-01,
       8.55211350e-02, 2.87979327e+00, 2.47836754e+00, 2.96705973e-01,
       2.79252680e-01, 1.79768913e+00, 9.42477796e-01, 3.31612554e-02,
       2.98276759e+00, 1.74532925e-01, 3.31612558e-01, 2.30383461e+00,
       1.39626340e-01, 3.66519126e-02, 2.61799388e-01, 3.49065850e-01,
       1.28281700e+00, 1.48352986e+00, 0.00000000e+00, 2.44346095e-01,
       3.85717771e-01, 1.39626340e-01, 5.28834750e-01, 1.97396736e+00,
       1.39626340e-01, 5.23598776e-02, 1.55334297e-01, 1.71042267e+00,
       1.74532925e-01, 1.39626340e-01, 7.33038286e-01, 1.90240882e-01,
       2.09439510e-01, 6.98131711e-03, 1.04719759e-02, 8.90117902e-02,
       5.23598776e-02, 2.61799388e-02, 1.74532925e-02, 8.72664626e-03,
       2.67035376e+00, 3.49065850e-02, 6.28318531e-01, 4.71238906e-02,
       1.48370443e+00, 2.02283663e+00, 3.13635661e+00, 9.77384364e-02,
       4.53785589e-02, 8.72664626e-02, 2.26892803e-01, 1.03323493e+00,
       0.00000000e+00, 8.72664626e-03, 6.98131701e-02, 1.91986218e+00,
       1.04719755e-01, 1.29154365e+00, 1.23918375e-01, 1.21300383e+00,
       5.93411962e-02, 1.96349541e+00, 1.97222209e-01, 1.46607651e-01,
       0.00000000e+00, 1.74532925e-02, 2.61799388e-01, 5.23598776e-01,
       1.57079633e-01, 1.09955746e-01, 8.72664626e-02, 1.74532925e-02,
       2.11184846e-01, 1.25663703e-01, 1.50098316e+00, 2.49582083e+00,
       1.22173048e-01, 1.02974426e+00, 1.39626340e-01, 5.77703956e-01,
       2.43647974e+00, 7.33038253e-02, 2.59181394e+00, 2.26892803e-01,
       1.74532925e-02, 2.21656812e-01, 1.39626340e-01, 3.83972435e-01,
       8.20304715e-02, 2.54818077e-01, 5.93411946e-01, 1.39626340e-01,
       1.22173048e-01, 4.88692182e-02, 1.83259571e-01, 1.97100033e+00,
       1.30899694e-01, 0.00000000e+00, 1.04719755e-01, 6.10865238e-02,
       0.00000000e+00, 1.91986222e-02, 1.74532925e-02, 2.25147474e+00,
       6.98131701e-02, 3.38593868e-01, 6.98131701e-02, 6.98131711e-03,
       3.31612554e-02, 1.22173048e-01, 1.34390349e-02, 1.06988682e+00,
       1.11701074e-01, 1.72962126e+00, 2.44346095e-01, 5.23598796e-03,
       0.00000000e+00, 1.39626340e-01, 2.63544717e+00, 1.37881011e+00,
       5.23598776e-02, 1.96576429e+00, 1.72787596e+00, 8.72664626e-02,
       8.18559446e-01, 1.52192705e+00, 2.24100265e+00, 7.62708896e-01,
       5.23598776e-02, 2.87979327e+00, 5.41052068e-01, 2.82743339e+00,
       1.55857908e+00, 3.66519143e-01, 1.22173048e-01, 6.51007798e-01,
       8.15068774e-01, 3.61283168e-01, 4.18879020e-01])

err_Lam = np.array([0.11170107, 0.08726646, 0.13439035, 0.17453293, 0.04537856,
       0.34906585, 0.0153589 , 0.0148353 , 0.57595865, 0.03665191,
       0.21293017, 0.15184364, 0.20769417, 0.10471976, 0.20943951,
       0.20071286, 0.13962634, 0.45378561, 0.2268928 , 0.15009832,
       0.08901179, 0.27925268, 0.2443461 , 0.26179939, 0.12042772,
       0.05235988, 0.38397244, 0.27925268, 0.15707963, 0.02617994,
       0.2443461 , 0.31415927, 0.10471976, 0.03490659, 0.10646508,
       0.08901179, 0.13962634, 0.43633231, 0.09773844, 0.40142573,
       0.08901179, 0.57595865, 0.13962634, 0.06632251, 0.12217305,
       0.00349066, 0.00698132, 0.06457718, 0.27925268, 0.15184364,
       0.15707963, 0.16929693, 0.13962634, 0.27925268, 0.19198622,
       0.01047198, 0.00401426, 0.0715585 , 0.06632251, 0.03316126,
       0.08901179, 0.12217305, 0.27925268, 0.00087266, 0.26179939,
       0.09948376, 0.17453293, 0.2443461 , 0.19198622, 0.55850536,
       0.04886922, 0.05235988, 0.03665191, 0.0296706 , 0.08377581,
       0.12391838, 0.13962634, 0.71558499, 0.45378561, 0.36651914,
       0.20943951, 0.08203047, 0.10471976, 0.20943951, 0.13962634,
       0.06457718, 0.10471976, 0.02617994, 0.08726646, 0.26179939,
       0.2268928 , 0.12915437, 0.09075712, 0.24260076, 0.09424778,
       0.12217305, 0.02094395, 0.07330383, 0.45378561, 0.27925268,
       0.06981317, 0.11693706, 0.45378561, 0.31415927, 0.33161256,
       0.05410521, 0.11170107, 0.00383972, 0.08203047, 0.19198622,
       0.19198622, 0.11868239, 0.41887902, 0.01919862, 0.20943951,
       0.29670597, 0.29845131, 0.08901179, 0.38397244, 0.03490659,
       0.13089969, 0.20943951, 0.01727876, 0.13264502, 0.10297443,
       0.06806784, 0.2443461 , 0.0296706 , 0.2443461 , 0.19198622,
       0.27925268, 0.33161256, 0.57595865, 0.43406343, 0.17453293,
       0.27925268, 0.08377581, 0.00715585, 0.09599311, 0.17278759,
       0.08726646, 0.08726646, 0.01745329, 0.08726646, 0.02443461,
       0.10471976, 0.19198622, 0.05235988, 0.10646508, 0.04014257,
       0.0715585 ])

if __name__ == '__main__':

    nplanet = len(Lam)
    with pm.Model() as randinc:
        
        ### hyper priors
        w = pm.Dirichlet('w', np.ones(2))

        a0 = pm.Uniform('a0', lower=0, upper=50)
        b0 = pm.Uniform('b0', lower=0, upper=1)
        
        a1 = pm.Uniform('a1', lower=0, upper=10)
        b1 = pm.Uniform('b1', lower=0, upper=10)
        
        components = [pm.Beta.dist(a0,b0), pm.Beta.dist(a1,b1)]
                            
        # flat prior on cosi
        cosi = pm.Uniform('cosi', lower=0., upper=1., shape=nplanet)
        sini = pm.Deterministic('sini', at.sqrt(1-cosi**2))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        # flat priors on λ
        λ = pm.Uniform('λ', lower=0, upper=np.pi, shape=nplanet)
        cosλ = pm.Deterministic('cosλ', at.cos(λ))

        cosψ = pm.Deterministic('cosψ', cosλ*sini)
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        
        cosθ = pm.Deterministic('cosθ', cosi/at.sin(ψ))
        θ = pm.Deterministic('θ', at.arccos(cosθ))
        
        # logl for λ
        logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_Lam, observed=Lam)

        ### logl for hyper priors
        mix = pm.Potential("mix", pm.logp(pm.Mixture.dist(w=w, comp_dists=components), (cosψ+1)/2))
        
        all_randinc = pm.sample()

    x = np.linspace(1e-5,1-1e-5,1000)

    post = all_randinc.posterior
    all_randinc_draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            all_randinc_draws[:, a*1000+b] = (post.w[a,b,0].values*beta.pdf(x, post.a0[a,b], post.b0[a,b])
                                              +post.w[a,b,1].values*beta.pdf(x, post.a1[a,b], post.b1[a,b]))

    # make the figure
    plt.figure(figsize=(3.5,2.7),dpi=110)

    q025, q16, q50, q84, q975 = np.percentile(all_randinc_draws, [2.5, 16, 50, 84, 97.5], axis=1)
    plt.plot(2*x-1, q50, color='C0', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

    plt.ylim([0,2])
    plt.xlim([-1,1])

    plt.xlabel(r'$\cos{\psi}$',fontsize=11)
    plt.ylabel('Probablity density function')

    plt.tight_layout()
    plt.savefig(paths.figures / "psi_dist.pdf", bbox_inches="tight", dpi=600)
    plt.close()
