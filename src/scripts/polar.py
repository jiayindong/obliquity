import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import sys
import subprocess
from datetime import datetime

import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as at
import pytensor
import scipy.stats as stats
from numpy import nan

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=False)

plt.rcParams['xtick.top'] =  True
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.major.width'] =  1.0
plt.rcParams['xtick.minor.width'] =  1.0
plt.rcParams['ytick.right'] =  True
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] =  1.0
plt.rcParams['lines.markeredgewidth'] =  1.0

# Data from Table 2 in Albrecht+21

# Radii, err_Radii in Solar unit
Radii = np.array([0.75, 0.9 , 0.88, 0.87, 1.62,  nan, 0.46, 2.02, 0.68, 0.68, 1.06,
        1.04, 0.9 , 0.91, 0.75, 1.16,  nan, 0.29, 0.86, 1.51, 2.42, 1.65,
        1.5 , 0.96, 1.71, 0.98, 1.32, 0.9 , 1.63,  nan, 1.92, 0.8 , 0.7 ,
        0.86,  nan, 0.12, 0.91, 1.09, 0.86, 1.48, 0.98, 1.66, 1.02, 1.11,
        1.51, 0.89, 0.67, 0.79, 1.28, 0.81, 1.76, 0.77, 0.94, 1.62, 0.67,
        1.44, 1.22, 1.79, 2.36, 1.  , 1.93, 1.17])
err_Radii = np.array([0.03 , 0.02 , 0.03 , 0.03 , 0.04 ,   nan, 0.02 , 0.01 , 0.01 ,
        0.01 , 0.05 , 0.02 , 0.02 , 0.03 , 0.03 , 0.01 ,   nan, 0.01 ,
        0.01 , 0.08 , 0.06 , 0.06 , 0.04 , 0.02 , 0.04 , 0.035, 0.015,
        0.025, 0.15 ,   nan, 0.11 , 0.02 , 0.01 , 0.02 ,   nan, 0.   ,
        0.02 , 0.04 , 0.03 , 0.09 , 0.02 , 0.045, 0.01 , 0.05 , 0.025,
        0.01 , 0.01 , 0.02 , 0.05 , 0.03 , 0.07 , 0.02 , 0.02 , 0.045,
        0.02 , 0.03 , 0.06 , 0.05 , 0.03 , 0.03 , 0.18 , 0.02 ])

# Prot, err_Prot in days
Prot = np.array([ 4.85,  4.52,  5.53,  2.85,  1.14,   nan, 44.09,   nan, 29.32,
        14.48, 28.7 , 15.3 , 24.98,  6.61, 11.95, 10.65,   nan,  1.88,
        10.76,  6.63,   nan,   nan,  7.13, 16.49,   nan, 12.09, 23.15,
         5.4 ,  1.29,   nan,   nan, 23.7 , 18.5 , 10.84,   nan,  3.28,
        22.2 , 16.2 , 23.8 ,  3.68, 15.31,  6.77, 12.13, 11.6 ,  0.52,
        18.41, 15.6 , 17.26,  6.65, 23.07,  9.29, 14.36, 13.08, 10.48,
        17.1 ,  3.38, 12.3 ,  1.02,   nan, 41.6 ,  1.79, 18.3 ])

err_Prot = np.array([0.75, 0.02, 0.33, 0.06, 0.06,  nan, 0.08,  nan, 1.  , 0.02, 0.4 ,
        0.4 , 0.04, 0.71, 0.02, 0.75,  nan, 0.04, 0.22, 0.66,  nan,  nan,
        0.14, 0.33,  nan, 0.24, 0.04, 0.01, 0.03,  nan,  nan, 0.12, 1.9 ,
        0.07,  nan, 0.22, 3.3 , 0.4 , 0.15, 1.23, 0.8 , 1.58, 2.1 , 1.  ,
        0.05, 0.05, 0.4 , 0.45, 0.13, 0.16, 1.27, 0.35, 0.26, 1.6 , 1.  ,
        0.4 , 1.9 , 0.1 ,  nan, 1.1 , 0.06, 1.  ])

# Vsini, err_Vsini in km/s
Vsini = np.array([  9.23,  11.25,   8.  ,  20.58,  74.92,    nan,   0.33,   2.7 ,
          1.  ,   1.85,   1.65,   3.12,   1.5 ,   7.3 ,   3.25,   4.8 ,
           nan,   8.9 ,   3.7 ,   6.9 , 116.9 ,  44.2 ,   8.9 ,   2.74,
         62.7 ,   4.7 ,   8.2 ,   5.6 ,  66.43,    nan,  46.5 ,   1.7 ,
          2.8 ,   4.2 ,    nan,   2.04,   2.14,   3.2 ,   1.6 ,  14.  ,
          1.9 ,   1.6 ,   4.4 ,   3.9 ,  86.63,   1.6 ,   2.26,   2.62,
          9.3 ,   2.2 ,   1.48,   2.56,   3.41,   4.2 ,   2.5 ,  13.56,
          5.1 ,  49.94, 100.  ,   1.07,  48.  ,   3.16])

err_Vsini = np.array([0.55 , 0.45 , 1.   , 0.275, 0.61 ,   nan, 0.08 , 0.5  , 0.755,
        0.27 , 0.26 , 0.75 , 0.5  , 0.3  , 0.02 , 0.2  ,   nan, 0.6  ,
        0.5  , 0.55 , 1.8  , 1.4  , 1.   , 0.4  , 0.2  , 1.   , 0.2  ,
        0.8  , 0.975,   nan, 1.   , 0.3  , 0.5  , 0.5  ,   nan, 0.18 ,
        0.365, 0.3  , 0.22 , 2.   , 0.05 , 0.6  , 0.9  , 0.45 , 0.345,
        1.1  , 0.54 , 0.07 , 0.2  , 0.4  , 0.28 , 0.08 , 0.89 , 0.5  ,
        0.8  , 0.685, 0.3  , 0.04 , 5.   , 0.09 , 3.   , 0.27 ])

# Lam, err_Lam in degrees
Lam = np.array([  4.7,   1. ,  10. ,   2.9,   1.5, 101. ,  72. , 142. , 103. ,
          8. ,   2.1,  14. ,   8. ,   8. ,   0.4,   0.6,   5.8,   3. ,
          1.5, 153. ,  85. , 115.9,   5. ,  13. ,  59.2,   0. ,   0.5,
        110. ,   7.1,  69.5, 112.5,   8.4,   0. ,  25. ,   1. ,  15. ,
          1. ,  12.1,   7.2,  86. , 143. ,  59. ,   1. ,  10.5, 112.9,
          6. ,   3.5,   1.1,  19.4,   0.4,  61.3,   0.3,   0. , 151. ,
        112.6,  87.2,   3. , 165. ,  89.3,   7. ,  20.7,  24. ])

err_Lam = np.array([ 6.6 ,  6.85, 20.  ,  0.9 ,  0.9 , 21.5 , 28.5 , 14.  , 18.  ,
         6.9 ,  3.  , 18.  ,  8.  , 39.  ,  0.2 ,  0.4 ,  4.25, 16.  ,
         8.7 ,  8.  ,  0.2 ,  4.1 ,  7.  , 16.  ,  0.1 , 15.  ,  5.7 ,
        18.  ,  3.5 ,  3.  ,  1.6 ,  7.1 ,  8.  , 13.  , 37.  , 28.  ,
        13.  ,  9.  ,  3.7 ,  6.  ,  1.55, 17.5 ,  1.2 ,  6.45,  0.2 ,
        11.  ,  6.8 ,  1.1 ,  5.  ,  1.95,  6.35,  1.7 , 14.  , 19.5 ,
        22.75,  0.4 ,  5.  ,  5.  ,  1.4 , 11.  ,  2.3 ,  4.1 ])

# Istar, err_Istar in degrees
Istar = np.array([ nan,  nan,  nan,  nan,  nan, 51. ,  nan, 33. ,  nan,  nan,  nan,
         nan,  nan,  nan,  nan,  nan, 90. ,  nan,  nan,  nan, 55.9, 94. ,
         nan,  nan, 81. ,  nan,  nan,  nan,  nan, 55.5, 63. ,  nan,  nan,
         nan, 76. ,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
         nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
         nan,  nan,  nan, 75.5,  nan,  nan,  nan])

err_Istar = np.array([  nan,   nan,   nan,   nan,   nan, 23.  ,   nan, 27.  ,   nan,
          nan,   nan,   nan,   nan,   nan,   nan,   nan,  2.5 ,   nan,
          nan,   nan,  9.4 ,  9.5 ,   nan,   nan, 16.  ,   nan,   nan,
          nan,   nan,  2.6 ,  8.5 ,   nan,   nan,   nan, 10.  ,   nan,
          nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan,
          nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan,
          nan,   nan,   nan,   nan,  2.65,   nan,   nan,   nan])

Lam = Lam*np.pi/180
err_Lam = err_Lam*np.pi/180

Istar = Istar*np.pi/180
err_Istar = err_Istar*np.pi/180

# Replace missing data
err_Radii[err_Radii<0.01] = 0.01
err_Vsini[np.isnan(err_Vsini)] = 0.5

mask1 = np.isnan(Prot)
mask2 = ~np.isnan(Prot)

if __name__ == '__main__':

	# Full sample
	nsample = len(Lam)

	# Subsample with direct istar measurements
	nsample1 = int(np.sum(mask1))

	# Subsample with Prot measurements
	nsample2 = int(np.sum(mask2))

	sim_dir = paths.data / "polar"
	sim_dir.mkdir(exist_ok=True, parents=True)

	# Model using istar info
	with pm.Model() as model_istar:
	    
	    ncomps = 2

	    # hyperprior
	    w = pm.Dirichlet('w', np.ones(ncomps))

	    if ncomps > 1:
	        μ = pm.Uniform('mu', lower=0., upper=1., shape=ncomps, 
	                       transform=pm.distributions.transforms.Ordered(), 
	                       initval=np.array([0.5,0.9]))
	    else:
	        μ = pm.Uniform('mu', lower=0., upper=1., shape=ncomps)
	        
	    logκ = pm.Normal('logkappa', 3.0, shape=ncomps)
	    κ = pm.Deterministic('kappa', pm.math.exp(logκ))
	  
	    a = pm.Deterministic('a', μ*κ)
	    b = pm.Deterministic('b', (1-μ)*κ)
	    
	    # Subsample 1: with direct istar measurmenets
	    
	    # mixture cosψ distribution
	    u = pm.Mixture('u', w=w, comp_dists=pm.Beta.dist(a,b, shape=(ncomps,)), shape=nsample1)
	    
	    cosψ = pm.Deterministic('cosψ', 2*u-1)
	    sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
	    
	    # uniform θ prior
	    θ = pm.Uniform('θ', lower=0, upper=np.pi, shape=nsample1)
	    sinθ = pm.Deterministic('sinθ', at.sin(θ))
	    cosθ = pm.Deterministic('cosθ', at.cos(θ))
	    
	    iorb = np.pi/2*np.ones(nsample1)
	    
	    # find λ in terms of ψ, θ, and iorb
	    λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
	    
	    # find istar in terms of ψ, θ, and iorb
	    cosistar = pm.Deterministic('cosistar', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
	    istar = pm.Deterministic('istar', at.arccos(cosistar))
	    
	    # logl for λ
	    logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_Lam[mask1], observed=Lam[mask1])    

	    # logl for istar
	    logl_istar = pm.Normal('logl_istar', mu=istar, sigma=err_Istar[mask1], observed=Istar[mask1])
	    
	    # Subsample 2: with Prot measurmenets
	    
	    # mixture cosψ distribution
	    u_ = pm.Mixture('u_', w=w, comp_dists=pm.Beta.dist(a,b, shape=(ncomps,)), shape=nsample2)
	    
	    cosψ_ = pm.Deterministic('cosψ_', 2*u_-1)
	    sinψ_ = pm.Deterministic('sinψ_', at.sqrt(1-cosψ_**2))
	    
	    # uniform θ prior
	    θ_ = pm.Uniform('θ_', lower=0, upper=np.pi, shape=nsample2)
	    sinθ_ = pm.Deterministic('sinθ_', at.sin(θ_))
	    cosθ_ = pm.Deterministic('cosθ_', at.cos(θ_))
	    
	    iorb_ = np.pi/2*np.ones(nsample2)

	    # find λ in terms of ψ, θ, and iorb
	    λ_ = pm.Deterministic('λ_', at.arctan2(sinψ_*sinθ_, cosψ_*at.sin(iorb_)-sinψ_*cosθ_*at.cos(iorb_)))
	    
	    # find istar in terms of ψ, θ, and iorb
	    cosistar_ = pm.Deterministic('cosistar_', sinψ_*cosθ_*at.sin(iorb_)+cosψ_*at.cos(iorb_))
	    istar_ = pm.Deterministic('istar_', at.arccos(cosistar_))
	    
	    # find vsini from Rstar, Prot, and istar
	    Rstar = pm.Normal('Rstar', mu=Radii[mask2], sigma=err_Radii[mask2])  # solar radii
	    Prot = pm.Normal('Prot', mu=Prot[mask2], sigma=err_Prot[mask2])  # days
	    
	    vrot = pm.Deterministic('vrot', 2*np.pi*Rstar/Prot*8.052) # normalization (1 solar radii)/(1 day) in km/s
	    vsini = pm.Deterministic('vsini', vrot*at.sin(istar_))
	    
	    # logl for λ
	    logl_λ_ = pm.Normal('logl_λ_', mu=λ_, sigma=err_Lam[mask2], observed=Lam[mask2])   
	    
	    # logl for vsini
	    logl_vsini = pm.Normal('logl_vsini', mu=vsini, sigma=err_Vsini[mask2], observed=Vsini[mask2])

	# Sampling
	with model_istar:
		idata_istar = pm.sample(nuts={'target_accept':0.99, 'max_treedepth':13}, # 'step_scale':0.01
        						chains=4, random_seed=int(datetime.now().strftime("%Y%m%d")))

	# Save the traces
	az.to_netcdf(idata_istar.posterior, sim_dir / "polar_istar.nc")
	
	# Model not using istar info
	with pm.Model() as model_noistar:
    
	    ncomps = 2

	    # hyperprior
	    w = pm.Dirichlet('w', np.ones(ncomps))

	    if ncomps > 1:
	        μ = pm.Uniform('mu', lower=0., upper=1., shape=ncomps, 
	                       transform=pm.distributions.transforms.Ordered(), 
	                       initval=np.array([0.5,0.9]))
	    else:
	        μ = pm.Uniform('mu', lower=0., upper=1., shape=ncomps)
	        
	    logκ = pm.Normal('logkappa', 3.0, shape=ncomps)
	    κ = pm.Deterministic('kappa', pm.math.exp(logκ))
	  
	    a = pm.Deterministic('a', μ*κ)
	    b = pm.Deterministic('b', (1-μ)*κ)
	        
	    # mixture cosψ distribution
	    u = pm.Mixture('u', w=w, comp_dists=pm.Beta.dist(a,b, shape=(ncomps,)), shape=nsample)
	    
	    cosψ = pm.Deterministic('cosψ', 2*u-1)
	    sinψ = pm.Deterministic('sinψ', at.sqrt(1-cosψ**2))
	    
	    # uniform θ prior
	    θ = pm.Uniform('θ', lower=0, upper=np.pi, shape=nsample)
	    sinθ = pm.Deterministic('sinθ', at.sin(θ))
	    cosθ = pm.Deterministic('cosθ', at.cos(θ))
	    
	    iorb = np.pi/2*np.ones(nsample)
	    
	    # find λ in terms of ψ, θ, and iorb
	    λ = pm.Deterministic('λ', at.arctan2(sinψ*sinθ, cosψ*at.sin(iorb)-sinψ*cosθ*at.cos(iorb)))
	    
	    # find istar in terms of ψ, θ, and iorb
	    cosistar = pm.Deterministic('cosistar', sinψ*cosθ*at.sin(iorb)+cosψ*at.cos(iorb))
	    istar = pm.Deterministic('istar', at.arccos(cosistar))
	    
	    # logl for λ
	    logl_λ = pm.Normal('logl_λ', mu=λ, sigma=err_Lam, observed=Lam)

	# Sampling
	with model_noistar:
		idata_noistar = pm.sample(nuts={'target_accept':0.99, 'max_treedepth':13}, # 'step_scale':0.01
									chains=4, random_seed=int(datetime.now().strftime("%Y%m%d")))
	# Save the traces
	az.to_netcdf(idata_noistar.posterior, sim_dir / "polar_noistar.nc")
