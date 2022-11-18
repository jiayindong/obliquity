import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

import paths

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'daft'])
import daft

from matplotlib import rc
rc('font', **{'family':'sans-serif'})
rc('text', usetex=True)

pgm = daft.PGM(observed_style="outer",node_unit=1, node_ec='k',dpi=150,line_width=0.8)
               
# Hierarchical parameters.
pgm.add_node("beta", r"$\beta$", 2, 2.3)

# Latent variable.
pgm.add_node("psi", r"$\psi_n$", 2, 1.5)

pgm.add_node("lambda", r"$\lambda_n$", 1., 1.5)

pgm.add_node("istar", r"$i_{\star,n}$", 3., 1.5)

pgm.add_node("pstar", r"$\theta_{\star,n}$", 3.6, 1.5)


pgm.add_edge("beta", "psi")

pgm.add_edge("lambda", "psi")

pgm.add_edge("istar", "psi")


pgm.add_node("obs_lambda", r"$\hat{\lambda}_n$", 1., 0.7, observed=True)

pgm.add_node("obs_vsini", r"$\hat{v}\sin{i}_{\star, n}$", 3., 0.7, observed=True, aspect=1.3, fontsize=8)

pgm.add_edge("lambda", "obs_lambda")

pgm.add_edge("pstar", "obs_vsini", plot_params=dict(ls=(0, (1, 2)),head_width=0,head_length=0))
pgm.add_edge("pstar", "obs_vsini", plot_params={'ls':''})

pgm.add_edge("istar", "obs_vsini", plot_params=dict(ls=(0, (1, 2)),head_width=0,head_length=0))
pgm.add_edge("istar", "obs_vsini", plot_params={'ls':''})

pgm.add_edge("lambda", "obs_vsini", plot_params=dict(ls=(0, (1, 2)),head_width=0,head_length=0))
pgm.add_edge("lambda", "obs_vsini", plot_params={'ls':''})

# And a plate.
pgm.add_plate([0.5, 0.2, 3.6, 1.75], label=r"$n = 1, \ldots, N$", shift=-0.1, fontsize=8)

# Render and save.
pgm.render()

pgm.savefig(paths.figures / "hbm.pdf", bbox_inches="tight", dpi=600)
