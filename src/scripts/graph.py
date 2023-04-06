import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import paths

import sys
import subprocess

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{physics}')

import daft

import daft

pgm = daft.PGM(observed_style="shaded",node_unit=1, node_ec='k',dpi=100,line_width=0.8)
               
# Hierarchical parameters.
pgm.add_node("beta", r"$\vb*{\beta}$", 2, 4.)

# Latent variable.
pgm.add_node("psi", r"$\psi_n$", 2, 3.)

pgm.add_node("theta", r"$\theta_n$", 3, 3.)

pgm.add_node("iorb", r"$i_{{\rm orb},n}$", 4, 3.)

pgm.add_node("lambda", r"$\lambda_n$", 2., 1.5)

pgm.add_node("istar", r"$i_{\star,n}$", 3, 1.5)

pgm.add_node("pstar", r"$\gamma_{\star,n}$", 4, 1.5)


# Observed variable.
pgm.add_node("obs_iorb", r"$\hat{i}_{{\rm orb}, n}$", 4, 2.2, observed=True)

pgm.add_node("obs_lambda", r"$\hat{\lambda}_n$", 2., 0.7, observed=True)

pgm.add_node("Obs", r"Obs$_{\star, n}$", 3, 0.7, observed=True, aspect=1., fontsize=8)


# Add edges
pgm.add_edge("beta", "psi")

pgm.add_edge("psi", "lambda")
pgm.add_edge("theta", "lambda")
pgm.add_edge("iorb", "lambda")

pgm.add_edge("psi", "istar")
pgm.add_edge("theta", "istar")
pgm.add_edge("iorb", "istar")

pgm.add_edge("iorb", "obs_iorb")

pgm.add_edge("lambda", "obs_lambda")


pgm.add_edge("pstar", "Obs", plot_params=dict(ls=(0, (2, 2)),head_width=0,head_length=0))
pgm.add_edge("pstar", "Obs", plot_params={'ls':''})

pgm.add_edge("istar", "Obs", plot_params=dict(ls=(0, (2, 2)),head_width=0,head_length=0))
pgm.add_edge("istar", "Obs", plot_params={'ls':''})

# And a plate.
pgm.add_plate([1.5, 0.2, 3, 3.4], label=r"$n = 1, \ldots, N$", shift=-0.1, fontsize=8)

# Render and save.
pgm.render();

pgm.savefig(paths.figures / "graph.pdf", bbox_inches="tight", dpi=600)
