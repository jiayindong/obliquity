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

from scipy.stats import beta

x = np.linspace(1e-5,1-1e-5,1000)
σ = 0.2

# return posteriors of HBM models with or without using measured istar
def posteriors(this_model):
    
    nsample = 200
    err_istar = 10.*np.pi/180
    err_lam = 8.*np.pi/180

    with this_model:    
        idata = pm.sample(chains=4, draws=50, tune=0, random_seed=47)

    true_istar = idata.posterior.i.values.ravel()
    obs_istar = true_istar + err_istar*np.random.normal(size=nsample)

    true_lam = idata.posterior.λ.values.ravel()
    obs_lam = true_lam + err_lam*np.random.normal(size=nsample)

    # Limit obs_lam to [0, pi]
    obs_lam[obs_lam<0] = -obs_lam[obs_lam<0]

    with pm.Model() as model:

        ncomps = 1
        
        # hyperprior
        w = pm.Dirichlet('w', np.ones(ncomps))
        
        if ncomps > 1:
            μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps, 
                           transform=pm.distributions.transforms.Ordered(), 
                           initval=np.sort(np.random.rand(ncomps)))
        else:
            μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps)
            
        logκ = pm.Normal('logκ', 3., shape=ncomps)
        κ = pm.Deterministic('κ', pm.math.exp(logκ))
      
        a = pm.Deterministic('a', μ*κ)
        b = pm.Deterministic('b', (1.-μ)*κ)
        
        # mixture cosψ distribution
        u = pm.Mixture('u', w=w, comp_dists=pm.Beta.dist(a, b, shape=(ncomps,)), shape=nsample)
        
        cosψ = pm.Deterministic('cosψ', 2.*u-1.)
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        # uniform θ prior
        θ = pm.Uniform('θ', lower=0., upper=np.pi, shape=nsample)
        sinθ = pm.Deterministic('sinθ', at.sin(θ))
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        
        # iorb
        iorb = np.pi/2*np.ones(nsample)
        
        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))

        # logl for λ
        logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_lam, observed=obs_lam)

        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        istar = pm.Deterministic('i', at.arccos(cosi))

        # logl for i
        logl_i = pm.Normal('logl_i', mu=istar, sigma=err_istar, observed=obs_istar)

        idata = pm.sample(nuts={'target_accept':0.99, 'max_treedepth':13}, 
                          chains=4, random_seed=123)

    with pm.Model() as model_noistar:

        ncomps = 1
        
        # hyperprior
        w = pm.Dirichlet('w', np.ones(ncomps))
        
        if ncomps > 1:
            μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps, 
                           transform=pm.distributions.transforms.Ordered(), 
                           initval=np.sort(np.random.rand(ncomps)))
        else:
            μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps)
            
        logκ = pm.Normal('logκ', 3., shape=ncomps)
        κ = pm.Deterministic('κ', pm.math.exp(logκ))
      
        a = pm.Deterministic('a', μ*κ)
        b = pm.Deterministic('b', (1.-μ)*κ)
        
        # mixture cosψ distribution
        u = pm.Mixture('u', w=w, comp_dists=pm.Beta.dist(a, b, shape=(ncomps,)), shape=nsample)
        
        cosψ = pm.Deterministic('cosψ', 2.*u-1.)
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        # uniform θ prior
        θ = pm.Uniform('θ', lower=0, upper=np.pi, shape=nsample)
        sinθ = pm.Deterministic('sinθ', at.sin(θ))
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        
        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))

        # logl for λ
        logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_lam, observed=obs_lam)
        
        noistar_idata = pm.sample(nuts={'target_accept':0.99, 'max_treedepth':13}, 
                                  chains=4, random_seed=123)


    # with pm.Model() as model_nolam:

    #     ncomps = 1
        
    #     # hyperprior
    #     w = pm.Dirichlet('w', np.ones(ncomps))
        
    #     if ncomps > 1:
    #         μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps, 
    #                        transform=pm.distributions.transforms.Ordered(), 
    #                        initval=np.sort(np.random.rand(ncomps)))
    #     else:
    #         μ = pm.Uniform('μ', lower=0., upper=1., shape=ncomps)
            
    #     logκ = pm.Normal('logκ', 3., shape=ncomps)
    #     κ = pm.Deterministic('κ', pm.math.exp(logκ))
      
    #     a = pm.Deterministic('a', μ*κ)
    #     b = pm.Deterministic('b', (1.-μ)*κ)
        
    #     # mixture cosψ distribution
    #     u = pm.Mixture('u', w=w, comp_dists=pm.Beta.dist(a, b, shape=(ncomps,)), shape=nsample)
        
    #     cosψ = pm.Deterministic('cosψ', 2.*u-1.)
    #     sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
    #     # uniform θ prior
    #     θ = pm.Uniform('θ', lower=0, upper=np.pi, shape=nsample)
    #     sinθ = pm.Deterministic('sinθ', at.sin(θ))
    #     cosθ = pm.Deterministic('cosθ', at.cos(θ))
        
    #     # iorb
    #     iorb = np.pi/2

    #     # find i in terms of ψ, θ, and iorb
    #     cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
    #     istar = pm.Deterministic('i', at.arccos(cosi))

    #     # logl for i
    #     logl_i = pm.Normal('logl_i', mu=istar, sigma=err_istar, observed=obs_istar)
        
    #     nolam_idata = pm.sample(nuts={'target_accept':0.9}, 
    #                             chains=16, random_seed=int(datetime.now().strftime("%Y%m%d")))


    return idata, noistar_idata #, nolam_idata


### PyMC models ###

if __name__ == '__main__':

    # cosψ ~ U(-1,1)
    with pm.Model() as model_uni:

        cosψ = pm.Uniform('cosψ',lower=-1.,upper=1.)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        θ = pm.Uniform('θ', lower=0., upper=np.pi)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        sinθ = pm.Deterministic('sinθ', at.sin(θ))

        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
        cosλ = pm.Deterministic('cosλ', at.cos(λ))

        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0., upper=1.)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1.-iso_cosi**2)*cosλ)   

    # cosψ ~ N(0,0.2)
    with pm.Model() as model_norm1:

        cosψ = pm.TruncatedNormal('cosψ', mu=0., sigma=0.2, lower=-1., upper=1.)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        θ = pm.Uniform('θ', lower=0, upper=np.pi)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        sinθ = pm.Deterministic('sinθ', at.sin(θ))

        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
        cosλ = pm.Deterministic('cosλ', at.cos(λ))

        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0., upper=1.)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1.-iso_cosi**2)*cosλ)
        
    # cosψ ~ N(-0.4,0.2)
    with pm.Model() as model_norm2:

        cosψ = pm.TruncatedNormal('cosψ', mu=-0.4, sigma=0.2, lower=-1., upper=1.)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        θ = pm.Uniform('θ', lower=0., upper=np.pi)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        sinθ = pm.Deterministic('sinθ', at.sin(θ))

        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
        cosλ = pm.Deterministic('cosλ', at.cos(λ))
      
        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0., upper=1.)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1.-iso_cosi**2)*cosλ)   
        
    # cosψ ~ N(0.4,0.2)
    with pm.Model() as model_norm3:

        cosψ = pm.TruncatedNormal('cosψ', mu=0.4, sigma=0.2, lower=-1., upper=1.)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        θ = pm.Uniform('θ', lower=0., upper=np.pi)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        sinθ = pm.Deterministic('sinθ', at.sin(θ))

        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
        cosλ = pm.Deterministic('cosλ', at.cos(λ))

        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0., upper=1.)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1.-iso_cosi**2)*cosλ)

    # cosψ ~ 2*Beta(3,6)-1
    with pm.Model() as model_beta:

        cosψ_ = pm.Beta('cosψ_', alpha=3, beta=6)
        cosψ = pm.Deterministic('cosψ', 2*cosψ_-1)
        
        ψ = pm.Deterministic('ψ', at.arccos(cosψ))
        sinψ = pm.Deterministic('sinψ', at.sqrt(1.-cosψ**2))
        
        θ = pm.Uniform('θ', lower=0., upper=np.pi)
        cosθ = pm.Deterministic('cosθ', at.cos(θ))
        sinθ = pm.Deterministic('sinθ', at.sin(θ))

        # iorb
        iorb = np.pi/2

        # find λ in terms of ψ, θ, and iorb
        λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
        cosλ = pm.Deterministic('cosλ', at.cos(λ))

        # find i in terms of ψ, θ, and iorb
        cosi = pm.Deterministic('cosi', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
        i = pm.Deterministic('i', at.arccos(cosi))
        
        iso_cosi = pm.Uniform('iso_cosi', lower=0., upper=1.)
        iso_cosψ = pm.Deterministic('iso_cosψ', at.sqrt(1.-iso_cosi**2)*cosλ)
        
        
    sim_dir = paths.data / "simulation"
    sim_dir.mkdir(exist_ok=True, parents=True)

    # cosψ ~ U(-1,1)
    uni, uni_noistar = posteriors(model_uni)
    az.to_netcdf(uni.posterior, sim_dir / "uni.nc")
    az.to_netcdf(uni_noistar.posterior, sim_dir / "uni_noistar.nc")
    #az.to_netcdf(uni_nolam.posterior, sim_dir / "uni_nolam.nc")

    # cosψ ~ N(0,0.2)
    norm1, norm1_noistar = posteriors(model_norm1)
    az.to_netcdf(norm1.posterior, sim_dir / "norm1.nc")
    az.to_netcdf(norm1_noistar.posterior, sim_dir / "norm1_noistar.nc")
    #az.to_netcdf(norm1_nolam.posterior, sim_dir / "norm1_nolam.nc")

    # cosψ ~ N(-0.4,0.2)
    norm2, norm2_noistar = posteriors(model_norm2)
    az.to_netcdf(norm2.posterior, sim_dir / "norm2.nc")
    az.to_netcdf(norm2_noistar.posterior, sim_dir / "norm2_noistar.nc")
    #az.to_netcdf(norm2_nolam.posterior, sim_dir / "norm2_nolam.nc")

    # cosψ ~ N(0.4,0.2)
    norm3, norm3_noistar = posteriors(model_norm3)
    az.to_netcdf(norm3.posterior, sim_dir / "norm3.nc")
    az.to_netcdf(norm3_noistar.posterior, sim_dir / "norm3_noistar.nc")
    #az.to_netcdf(norm3_nolam.posterior, sim_dir / "norm3_nolam.nc")

    # cosψ ~ Beta(3,6)
    beta, beta_noistar = posteriors(model_beta)
    az.to_netcdf(beta.posterior, sim_dir / "beta.nc")
    az.to_netcdf(beta_noistar.posterior, sim_dir / "beta_noistar.nc")
