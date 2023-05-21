# dummy
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

def read_table2(df, var_name):
    size = len(df)
    
    var = []
    err_var = []
    
    for i in range(size):
    
        this_var = df['%s'%var_name][i]
        
        if this_var == "cdots":
            var.append(np.nan)
            err_var.append(np.nan)
        elif '+or-' in this_var:
            for j in range(len(this_var)):
                if this_var[j] == '+':
                    var.append(float(this_var[:j-1]))
                if this_var[j] == '-':
                    err_var.append(float(this_var[j+2:]))
        else:
            for j in range(len(this_var)-1):
                if this_var[j:j+2] == '${':
                    for k in range(6):
                        if this_var[j+2+k] == '}':
                            var.append(float(this_var[j+2:j+2+k]))
                elif this_var[j] == '_':
                    for k in range(10):
                        if this_var[j+k] == '^':
                            low = float(this_var[j+3:j+k-1])
                            high = float(this_var[j+k+3:len(this_var)-2])
                    err_var.append((low+high)/2)
                    
    return np.array(var), np.array(err_var)

# Load Table 2 of Albrecht+21
df = pd.read_csv(paths.data / "Albrecht21_Table2.csv")

Radii, err_Radii = read_table2(df, 'R')
Prot, err_Prot = read_table2(df, 'Prot')
Vsini, err_Vsini = read_table2(df, 'vsini')
Lam, err_Lam = read_table2(df, 'lambda')
Psi, err_Psi = read_table2(df, 'psi')
Istar, err_Istar = read_table2(df, 'i')

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
