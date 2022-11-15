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

pgm = daft.PGM(observed_style="outer", dpi=150, node_unit=1.1, node_ec='k')

# Hierarchical parameters.
pgm.add_node("w", r"$w$", 1.5, 2.7)
pgm.add_node("alpha", r"$\alpha$", 1.5, 2.15)
pgm.add_node("beta", r"$\beta$", 1.5, 1.6)

# Latent variable.
pgm.add_node("lam", r"$\lambda$", 3, 4)
pgm.add_node("obslam", r"$\lambda$", 3, 3.2, observed=True)

pgm.add_node("coslam", r"$\cos{\lambda}$", 4, 3.2)
pgm.add_node("errlam", r"$\sigma_\lambda$", 2.2, 3.2, fixed=True)


pgm.add_node("cosi", r"$\cos{i_{\star}}$", 5, 4)
pgm.add_node("sini", r"$\sin{i_{\star}}$", 5, 3.2)

pgm.add_node("Rstar", r"$R_{\star}$", 5.8, 4)
pgm.add_node("Prot", r"$P_{\rm rot}$", 6.6, 4)

pgm.add_node("v", r"$v$", 6.2, 3.2)

pgm.add_node("cospsi", r"$\cos{\psi}$", 4.5, 1.9)

pgm.add_node("vsini", r"$v\sin{i_{\star}}$", 6.2, 2.4)
pgm.add_node("obsvsini", r"$v\sin{i_{\star}}$", 6.2, 1.4, observed=True)
pgm.add_node("errvsini", r"$\sigma_{v\sin{i_{\star}}}$", 7, 1.4, fixed=True)

# Add in the edges.
pgm.add_edge("lam", "coslam")
pgm.add_edge("lam", "obslam")
pgm.add_edge("errlam", "obslam")

pgm.add_edge("cosi", "sini")

pgm.add_edge("Rstar", "v")
pgm.add_edge("Prot", "v")


pgm.add_edge("sini", "cospsi")
pgm.add_edge("coslam", "cospsi")

pgm.add_edge("sini", "vsini")
pgm.add_edge("v", "vsini")
pgm.add_edge("vsini", "obsvsini")
pgm.add_edge("errvsini", "obsvsini")


pgm.add_edge("w", "cospsi")
pgm.add_edge("alpha", "cospsi")
pgm.add_edge("beta", "cospsi")

# And a plate.
pgm.add_plate([2, 1, 5.3, 3.4], label=r"$n = 1, \ldots, N$")

pgm.add_plate([1.1, 1., 0.8, 2.], label=r"$j = 1,2$")

# Render and save.
pgm.render()

pgm.savefig(paths.figures / "hbm_istar.pdf", bbox_inches="tight", dpi=600)