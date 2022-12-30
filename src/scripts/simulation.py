import paths
import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'arviz'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pymc==4.1.7'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'aesara'])

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
        idata = pm.sample(draws=50)

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

        istar_idata = pm.sample()

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

        noistar_idata = pm.sample()
        
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
    norm1_istar_draws, norm1_noistar_draws = posteriors(model_norm1)
    norm2_istar_draws, norm2_noistar_draws = posteriors(model_norm2)
    norm3_istar_draws, norm3_noistar_draws = posteriors(model_norm3)

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
    plt.plot(2*x-1, q50, color='C0', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

    plt.ylim([0,1])
    plt.xlim([-1,1])

    plt.xlabel(r'$\cos{\psi}$',fontsize=11)
    plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

    cosψ = np.linspace(-1,1,200)
    pcosψ = np.ones_like(cosψ)*0.5
    plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1)

    ax = fig.add_subplot(4,2,2)

    q025, q16, q50, q84, q975 = np.percentile(uni_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
    plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

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
    plt.plot(2*x-1, q50, color='C0', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

    plt.ylim([0,2.1])
    plt.xlim([-1,1])

    plt.xlabel(r'$\cos{\psi}$',fontsize=11)
    plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ)**2/2/σ**2)
    plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

    ax = fig.add_subplot(4,2,4)

    q025, q16, q50, q84, q975 = np.percentile(norm1_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
    plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

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
    plt.plot(2*x-1, q50, color='C0', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

    plt.ylim([0,2.1])
    plt.xlim([-1,1])

    plt.xlabel(r'$\cos{\psi}$',fontsize=11)
    plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)
    plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

    ax = fig.add_subplot(4,2,6)

    q025, q16, q50, q84, q975 = np.percentile(norm2_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
    plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

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
    plt.plot(2*x-1, q50, color='C0', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#7dabd0')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#7dabd0')

    plt.ylim([0,2.1])
    plt.xlim([-1,1])

    plt.xlabel(r'$\cos{\psi}$',fontsize=11)
    plt.title(r'\textbf{with} $i_{\star}$ likelihood', fontsize=10)

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)
    plt.plot(cosψ, pcosψ, c='slategrey', ls='--', lw=1, zorder=0)

    ax = fig.add_subplot(4,2,8)

    q025, q16, q50, q84, q975 = np.percentile(norm3_noistar_draws, [2.5, 16, 50, 84, 97.5], axis=1)/2
    plt.plot(2*x-1, q50, color='#f56e4a', lw=1.2)
    plt.fill_between(2*x-1, q16, q84, alpha=0.3, label="posterior", color='#fbc1ad')
    plt.fill_between(2*x-1, q025, q975, alpha=0.3, color='#fbc1ad')

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