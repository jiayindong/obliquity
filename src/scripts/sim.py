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

from scipy.stats import beta
x = np.linspace(1e-5,1-1e-5,1000)
σ = 0.2

# return posteriors of HBM models with or without using measured istar
def posteriors(this_model):
    
    nsample = 200
    err_istar = 10*np.pi/180
    err_lam = 8*np.pi/180

    with this_model:    
        idata = pm.sample(chains=4, draws=50)

    true_istar = idata.posterior.i.values.ravel()
    obs_istar = true_istar + err_istar*np.random.normal(size=nsample)

    true_lam = idata.posterior.λ.values.ravel()
    obs_lam = true_lam + err_lam*np.random.normal(size=nsample)

    with pm.Model() as model_istar:

        # hyperprior
        a = pm.Uniform('a', lower=0, upper=10)
        b = pm.Uniform('b', lower=0, upper=10)

        # flat prior on cosi
        cosi = pm.Uniform('cosi', lower=0., upper=1., shape=nsample)
        sini = pm.Deterministic('sini', at.sqrt(1-cosi**2))

        i = pm.Deterministic('i', at.arccos(cosi))

        # flat priors on ψ
        λ = pm.Uniform('λ', lower=0, upper=np.pi, shape=nsample)
        cosλ = pm.Deterministic('cosλ', np.cos(λ))

        cosψ = pm.Deterministic('cosψ', cosλ*sini)
        ψ = pm.Deterministic('ψ', np.arccos(cosψ))

        # logl for λ
        logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_lam, observed=obs_lam)

        logl_i = pm.Normal('logl_i', mu=i, sigma=err_istar, observed=obs_istar)

        hyper = pm.Potential("hyper", pm.logp(pm.Beta.dist(a,b), (cosψ+1)/2))

        istar_idata = pm.sample(target_accept=0.9,chains=4)

    with pm.Model() as model_noistar:

        # hyperprior
        a = pm.Uniform('a', lower=0, upper=10)
        b = pm.Uniform('b', lower=0, upper=10)

        # flat prior on cosi
        cosi = pm.Uniform('cosi', lower=0., upper=1., shape=nsample)
        sini = pm.Deterministic('sini', at.sqrt(1-cosi**2))

        i = pm.Deterministic('i', at.arccos(cosi))

        # flat priors on ψ
        λ = pm.Uniform('λ', lower=0, upper=np.pi, shape=nsample)
        cosλ = pm.Deterministic('cosλ', np.cos(λ))

        cosψ = pm.Deterministic('cosψ', cosλ*sini)
        ψ = pm.Deterministic('ψ', np.arccos(cosψ))

        # logl for λ
        logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_lam, observed=obs_lam)

        hyper = pm.Potential("hyper", pm.logp(pm.Beta.dist(a,b), (cosψ+1)/2))

        noistar_idata = pm.sample(target_accept=0.9,chains=4)
        
    post = istar_idata.posterior
    istar_draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            istar_draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    post = noistar_idata.posterior
    noistar_draws = np.zeros(shape=(len(x),4000))
    for a in range(4):
        for b in range(1000):
            noistar_draws[:, a*1000+b] = beta.pdf(x, post.a[a,b], post.b[a,b])

    return istar_draws, noistar_draws


### PyMC models ###

if __name__ == '__main__':

    with pm.Model() as model_uni:

        cosψ = pm.Uniform('cosψ',lower=-1,upper=1)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
        
        θ = pm.Uniform('θ', lower=-np.pi/2, upper=np.pi/2)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        tanθ = pm.Deterministic('tanθ', at.tan(θ))

        cosλ = pm.Deterministic('cosλ', cosψ/at.sqrt(1-sinψ**2*cosθ**2))
        λ = pm.Deterministic('λ', at.arccos(cosλ))
        
        sini = pm.Deterministic('sini', cosψ/cosλ)
        i = pm.Deterministic('i', at.arcsin(sini))
        cosi = pm.Deterministic('cosi', at.sqrt(1-sini**2))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0, upper=1)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1-iso_cosi**2)*cosλ)
        

    with pm.Model() as model_norm1:

        cosψ = pm.Normal('cosψ', mu=0., sigma=0.2)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
        
        θ = pm.Uniform('θ', lower=-np.pi/2, upper=np.pi/2)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        tanθ = pm.Deterministic('tanθ', at.tan(θ))

        cosλ = pm.Deterministic('cosλ', cosψ/at.sqrt(1-sinψ**2*cosθ**2))
        λ = pm.Deterministic('λ', at.arccos(cosλ))
        
        sini = pm.Deterministic('sini', cosψ/cosλ)
        i = pm.Deterministic('i', at.arcsin(sini))
        cosi = pm.Deterministic('cosi', at.sqrt(1-sini**2))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0, upper=1)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1-iso_cosi**2)*cosλ)
        

    with pm.Model() as model_norm2:

        cosψ = pm.Normal('cosψ', mu=-0.4, sigma=0.2)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
        
        θ = pm.Uniform('θ', lower=-np.pi/2, upper=np.pi/2)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        tanθ = pm.Deterministic('tanθ', at.tan(θ))

        cosλ = pm.Deterministic('cosλ', cosψ/at.sqrt(1-sinψ**2*cosθ**2))
        λ = pm.Deterministic('λ', at.arccos(cosλ))
        
        sini = pm.Deterministic('sini', cosψ/cosλ)
        i = pm.Deterministic('i', at.arcsin(sini))
        cosi = pm.Deterministic('cosi', at.sqrt(1-sini**2))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0, upper=1)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1-iso_cosi**2)*cosλ)    
        

    with pm.Model() as model_norm3:

        cosψ = pm.Normal('cosψ', mu=0.4, sigma=0.2)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
        
        θ = pm.Uniform('θ', lower=-np.pi/2, upper=np.pi/2)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        tanθ = pm.Deterministic('tanθ', at.tan(θ))

        cosλ = pm.Deterministic('cosλ', cosψ/at.sqrt(1-sinψ**2*cosθ**2))
        λ = pm.Deterministic('λ', at.arccos(cosλ))
        
        sini = pm.Deterministic('sini', cosψ/cosλ)
        i = pm.Deterministic('i', at.arcsin(sini))
        cosi = pm.Deterministic('cosi', at.sqrt(1-sini**2))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0, upper=1)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1-iso_cosi**2)*cosλ)
        
        
    uni_istar_draws, uni_noistar_draws = posteriors(model_uni)
    np.save(paths.data / "uni_istar_draws.npy", uni_istar_draws)
    np.save(paths.data / "uni_noistar_draws.npy", uni_noistar_draws)

    norm1_istar_draws, norm1_noistar_draws = posteriors(model_norm1)
    np.save(paths.data / "norm1_istar_draws.npy", norm1_istar_draws)
    np.save(paths.data / "norm1_noistar_draws.npy", norm1_noistar_draws)

    norm2_istar_draws, norm2_noistar_draws = posteriors(model_norm2)
    np.save(paths.data / "norm2_istar_draws.npy", norm2_istar_draws)
    np.save(paths.data / "norm2_noistar_draws.npy", norm2_noistar_draws)

    norm3_istar_draws, norm3_noistar_draws = posteriors(model_norm3)
    np.save(paths.data / "norm3_istar_draws.npy", norm3_istar_draws)
    np.save(paths.data / "norm3_noistar_draws.npy", norm3_noistar_draws)    
