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

# from matplotlib import rc
# rc('font', **{'family':'sans-serif'})
# rc('text', usetex=False)
# rc('text.latex', preamble=r'\usepackage{physics}')

plt.rcParams['xtick.top'] =  True
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.major.width'] =  1.0
plt.rcParams['xtick.minor.width'] =  1.0
plt.rcParams['ytick.right'] =  True
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] =  1.0
plt.rcParams['lines.markeredgewidth'] =  1.0

### PyMC models ###

if __name__ == '__main__':

    with pm.Model() as uni:

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
        
        uni = pm.sample()


    with pm.Model() as norm1:

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
        
        norm1 = pm.sample()

    with pm.Model() as norm2:

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
        
        norm2 = pm.sample()

    with pm.Model() as norm3:

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
        
        norm3 = pm.sample()



    ### Make the plot ###

    plt.figure(figsize=(7,5),dpi=110)


    ### cosψ ~ Uniform(-1,1) ###

    cosψ = uni.posterior.cosψ.values.ravel()
    λ = uni.posterior.λ.values.ravel()
    i = uni.posterior.i.values.ravel()
    iso_cosψ = uni.posterior.iso_cosψ.values.ravel()

    plt.subplot(4,4,1)
    plt.hist(cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\cos{\psi} \sim \mathcal{U}(-1,1)$')
    plt.ylim([0,0.6])
    plt.xlim([-1,1])

    plt.subplot(4,4,2)
    plt.hist(λ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\lambda$')
    plt.ylim([0,0.5])
    plt.xlim([0,np.pi])
    plt.xticks([0,np.pi/2,np.pi], ['0', r'$\pi/2$', r'$\pi$'])

    plt.subplot(4,4,3)
    plt.hist(i, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$i_\star$')
    plt.xlim([0,np.pi/2])
    plt.xticks([0,np.pi/4,np.pi/2], ['0', r'$\pi/4$', r'$\pi/2$'])
    plt.ylim([0,1.1])

    plt.subplot(4,4,4)
    plt.hist(iso_cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'Inferred $\cos{\psi}$')
    plt.ylim([0,0.6])
    plt.xlim([-1,1])


    cosψ = np.linspace(-1,1,200)
    pcosψ = np.ones_like(cosψ)/2

    λ = np.linspace(0,np.pi,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5*1/2,
                             0, 1)[0]*2/np.pi

    istar = np.linspace(0,np.pi/2,800)
    cosi = np.cos(istar)
    pcosi = np.zeros_like(cosi)
    for i,this_cosi in enumerate(cosi):
        pcosi[i] = integrate.quad(lambda x: this_cosi/x**2/np.sqrt(1-this_cosi**2/x**2)/np.sqrt(1-x**2)*1/2,
                             this_cosi, 1)[0]*4/np.pi
        
    plt.subplot(4,4,1)
    plt.plot(cosψ,pcosψ,lw=1.2)

    plt.subplot(4,4,2)
    plt.plot(np.arccos(cosλ),pcosλ*np.sqrt(1-cosλ**2),lw=1.2)

    plt.subplot(4,4,3)
    plt.plot(np.arccos(cosi), pcosi*np.sqrt(1-cosi**2),lw=1.2)

    plt.subplot(4,4,4)
    plt.plot(cosψ,pcosψ,lw=1.2,linestyle='--',label='True')
    plt.legend(framealpha=0,handlelength=1,handletextpad=0.3,borderpad=0.1)


    ### cosψ ~ Normal(0,0.2) ###

    σ = 0.2

    cosψ = norm1.posterior.cosψ.values.ravel()
    λ = norm1.posterior.λ.values.ravel()
    i = norm1.posterior.i.values.ravel()
    iso_cosψ = norm1.posterior.iso_cosψ.values.ravel()

    plt.subplot(4,4,5)
    plt.hist(cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\cos{\psi} \sim \mathcal{N}(0,0.2)$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    plt.subplot(4,4,6)
    plt.hist(λ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\lambda$')
    plt.ylim([0,1.5])
    plt.xlim([0,np.pi])
    plt.xticks([0,np.pi/2,np.pi], ['0', r'$\pi/2$', r'$\pi$'])

    plt.subplot(4,4,7)
    plt.hist(i, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$i_\star$')
    plt.xlim([0,np.pi/2])
    plt.xticks([0,np.pi/4,np.pi/2], ['0', r'$\pi/4$', r'$\pi/2$'])
    plt.ylim([0,1.1])

    plt.subplot(4,4,8)
    plt.hist(iso_cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'Inferred $\cos{\psi}$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-cosψ**2/2/σ**2)

    λ = np.linspace(0,np.pi,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(this_cosλ**2*x**2-this_cosλ**2)/(this_cosλ**2*x**2-1)/2/σ**2),
                             0, 1)[0]*2/np.pi

    istar = np.linspace(0,np.pi/2,800)
    cosi = np.cos(istar)
    pcosi = np.zeros_like(cosi)
    for i,this_cosi in enumerate(cosi):
        pcosi[i] = integrate.quad(lambda x: this_cosi/x**2/np.sqrt(1-this_cosi**2/x**2)/np.sqrt(1-x**2)
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(1-this_cosi**2/x**2)/2/σ**2),
                             this_cosi, 1)[0]*4/np.pi

    plt.subplot(4,4,5)
    plt.plot(cosψ,pcosψ,lw=1.2)

    plt.subplot(4,4,6)
    plt.plot(λ,pcosλ*np.sqrt(1-cosλ**2),lw=1.2)

    plt.subplot(4,4,7)
    plt.plot(istar, pcosi*np.sqrt(1-cosi**2),lw=1.2)

    plt.subplot(4,4,8)
    plt.plot(cosψ,pcosψ,lw=1.2,linestyle='--',label='True')
    plt.legend(framealpha=0,handlelength=1,handletextpad=0.3,borderpad=0.1)



    ### cosψ ~ Normal(-0.4,0.2) ###

    cosψ = norm2.posterior.cosψ.values.ravel()
    λ = norm2.posterior.λ.values.ravel()
    i = norm2.posterior.i.values.ravel()
    iso_cosψ = norm2.posterior.iso_cosψ.values.ravel()


    plt.subplot(4,4,9)
    plt.hist(cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\cos{\psi} \sim \mathcal{N}(-0.4,0.2)$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    plt.subplot(4,4,10)
    plt.hist(λ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\lambda$')
    plt.ylim([0,1.5])
    plt.xlim([0,np.pi])
    plt.xticks([0,np.pi/2,np.pi], ['0', r'$\pi/2$', r'$\pi$'])

    plt.subplot(4,4,11)
    plt.hist(i, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$i_\star$')
    plt.xlim([0,np.pi/2])
    plt.xticks([0,np.pi/4,np.pi/2], ['0', r'$\pi/4$', r'$\pi/2$'])
    plt.ylim([0,1.1])

    plt.subplot(4,4,12)
    plt.hist(iso_cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'Inferred $\cos{\psi}$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ+0.4)**2/2/σ**2)

    λ = np.linspace(0,np.pi/2,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt((this_cosλ**2*x**2-this_cosλ**2)/(this_cosλ**2*x**2-1))+0.4)**2/2/σ**2),
                             0, 1)[0]*2/np.pi
        
    plt.subplot(4,4,10)
    plt.plot(λ,pcosλ*np.sqrt(1-cosλ**2),lw=1.2)

    λ = np.linspace(np.pi/2,3*np.pi/2,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt((this_cosλ**2*x**2-this_cosλ**2)/(this_cosλ**2*x**2-1))-0.4)**2/2/σ**2),
                             0, 1)[0]*2/np.pi

    plt.plot(λ,pcosλ*np.sqrt(1-cosλ**2), c='C0',lw=1.2)
        
    istar = np.linspace(0,np.pi/2,800)
    cosi = np.cos(istar)
    pcosi = np.zeros_like(cosi)
    for i,this_cosi in enumerate(cosi):
        pcosi[i] = integrate.quad(lambda x: this_cosi/x**2/np.sqrt(1-this_cosi**2/x**2)/np.sqrt(1-x**2)
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt(1-this_cosi**2/x**2)-0.4)**2/2/σ**2),
                             this_cosi, 1)[0]*4/np.pi

    plt.subplot(4,4,9)
    plt.plot(cosψ,pcosψ,lw=1.2)

    plt.subplot(4,4,11)
    plt.plot(istar, pcosi*np.sqrt(1-cosi**2)/2,lw=1.2)

    plt.subplot(4,4,12)
    plt.plot(cosψ,pcosψ,lw=1.2,linestyle='--',label='True')
    plt.legend(framealpha=0,handlelength=1,handletextpad=0.3,borderpad=0.1)



    ### cosψ ~ Normal(0.4,0.2) ###

    cosψ = norm3.posterior.cosψ.values.ravel()
    λ = norm3.posterior.λ.values.ravel()
    i = norm3.posterior.i.values.ravel()
    iso_cosψ = norm3.posterior.iso_cosψ.values.ravel()

    plt.subplot(4,4,13)
    plt.hist(cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\cos{\psi} \sim \mathcal{N}(0.4,0.2)$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    plt.subplot(4,4,14)
    plt.hist(λ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$\lambda$')
    plt.ylim([0,1.5])
    plt.xlim([0,np.pi])
    plt.xticks([0,np.pi/2,np.pi], ['0', r'$\pi/2$', r'$\pi$'])

    plt.subplot(4,4,15)
    plt.hist(i, color='#e5e1e0', density=True, bins=40)
    plt.title(r'$i_\star$')
    plt.xlim([0,np.pi/2])
    plt.xticks([0,np.pi/4,np.pi/2], ['0', r'$\pi/4$', r'$\pi/2$'])
    plt.ylim([0,1.1])

    plt.subplot(4,4,16)
    plt.hist(iso_cosψ, color='#e5e1e0', density=True, bins=40)
    plt.title(r'Inferred $\cos{\psi}$')
    plt.ylim([0,2.5])
    plt.xlim([-1,1])

    cosψ = np.linspace(-1,1,200)
    pcosψ = 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(cosψ-0.4)**2/2/σ**2)

    λ = np.linspace(0,np.pi/2,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt((this_cosλ**2*x**2-this_cosλ**2)/(this_cosλ**2*x**2-1))-0.4)**2/2/σ**2),
                             0, 1)[0]*2/np.pi
        
    plt.subplot(4,4,14)
    plt.plot(λ,pcosλ*np.sqrt(1-cosλ**2),lw=1.2)

    λ = np.linspace(np.pi/2,3*np.pi/2,200)
    cosλ = np.cos(λ)
    pcosλ = np.zeros_like(cosλ)
    for i,this_cosλ in enumerate(cosλ):
        pcosλ[i] = integrate.quad(lambda x: (1-this_cosλ**2*x**2)**-1.5
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt((this_cosλ**2*x**2-this_cosλ**2)/(this_cosλ**2*x**2-1))+0.4)**2/2/σ**2),
                             0, 1)[0]*2/np.pi

    plt.plot(λ,pcosλ*np.sqrt(1-cosλ**2),c='C0',lw=1.2)
        
    istar = np.linspace(0,np.pi/2,800)
    cosi = np.cos(istar)
    pcosi = np.zeros_like(cosi)
    for i,this_cosi in enumerate(cosi):
        pcosi[i] = integrate.quad(lambda x: this_cosi/x**2/np.sqrt(1-this_cosi**2/x**2)/np.sqrt(1-x**2)
                                  *1/np.sqrt(2*np.pi*σ**2)*np.exp(-(np.sqrt(1-this_cosi**2/x**2)-0.4)**2/2/σ**2),
                             this_cosi, 1)[0]*4/np.pi

    plt.subplot(4,4,13)
    plt.plot(cosψ,pcosψ,lw=1.2)

    plt.subplot(4,4,15)
    plt.plot(istar, pcosi*np.sqrt(1-cosi**2)/2,lw=1.2)

    plt.subplot(4,4,16)
    plt.plot(cosψ,pcosψ,lw=1.2,linestyle='--',label='True')
    plt.legend(framealpha=0,handlelength=1,handletextpad=0.3,borderpad=0.1)

    plt.tight_layout()

    plt.savefig(paths.figures / "transform.pdf", bbox_inches="tight", dpi=600)
    plt.close()