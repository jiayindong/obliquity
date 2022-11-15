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

pgm = daft.PGM(observed_style="inner")

# Hierarchical parameters.
pgm.add_node("w", r"$w$", 1.5, 2.7)
pgm.add_node("alpha", r"$\alpha$", 1.5, 2.15)
pgm.add_node("beta", r"$\beta$", 1.5, 1.6)

# Latent variable.
pgm.add_node("lam", r"$\lambda$", 3, 4)
pgm.add_node("obslam", r"$\lambda$", 3, 3.2, observed=True)

pgm.add_node("coslam", r"$\cos{\lambda}$", 4, 3.2)
pgm.add_node("errlam", r"$\sigma_\lambda$", 2.2, 3.2, fixed=True)


pgm.add_node("cosi", r"$\cos{i}$", 5, 4)
pgm.add_node("sini", r"$\sin{i}$", 5, 3.2)

pgm.add_node("Rstar", r"$R_{\star}$", 5.8, 4)
pgm.add_node("Prot", r"$P_{\rm rot}$", 6.6, 4)

pgm.add_node("v", r"$v$", 6.2, 3.2)

pgm.add_node("cospsi", r"$\cos{\psi}$", 4.5, 1.9)

pgm.add_node("vsini", r"$v\sin{i}$", 6.2, 2.4)
pgm.add_node("obsvsini", r"$v\sin{i}$", 6.2, 1.6, observed=True)
pgm.add_node("errvsini", r"$\sigma_{v\sin{i}}$", 7, 1.6, fixed=True)

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



####### Reflected ###########

pgm.add_node("cospsi_", r"$\cos{\psi}$", -1.5, 1.9)

pgm.add_edge("w", "cospsi_")
pgm.add_edge("alpha", "cospsi_")
pgm.add_edge("beta", "cospsi_")


pgm.add_node("lam_", r"$\lambda$", 0, 4)
pgm.add_node("obslam_", r"$\lambda$", 0, 3.2, observed=True)

pgm.add_node("coslam_", r"$\cos{\lambda}$", -1, 3.2)
pgm.add_node("errlam_", r"$\sigma_\lambda$", 0.8, 3.2, fixed=True)

pgm.add_edge("lam_", "coslam_")
pgm.add_edge("lam_", "obslam_")
pgm.add_edge("errlam_", "obslam_")


pgm.add_node("sini_", r"$\sin{i}$", -2, 3.2)

pgm.add_edge("sini_", "cospsi_")
pgm.add_edge("coslam_", "cospsi_")

pgm.add_node("i_", r"$i$", -3, 4)
pgm.add_node("obsi_", r"$i$", -3, 3.2, observed=True)

pgm.add_edge("i_", "obsi_")
pgm.add_edge("i_", "sini_")

pgm.add_node("erri_", r"$\sigma_i$", -3.8, 3.2, fixed=True)

pgm.add_edge("erri_", "obsi_")


# And a plate.
pgm.add_plate([2, 1, 5.4, 3.4], label=r"$n = 1, \cdots, N$")

pgm.add_plate([5.4, 1.1, 1.85, 3.2], label=r"$l = 1, \ldots, L$")

pgm.add_plate([1.1, 1., 0.8, 2.], label=r"$j = 1,2$")

pgm.add_plate([-4, 1, 5, 3.4], label=r"$m = 1, \ldots, M$")

# Render and save.
pgm.render()

pgm.savefig(paths.figures / "hbm_all.pdf", bbox_inches="tight", dpi=600)
