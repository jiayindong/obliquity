rule psi:
    output:
        "src/data/all_randinc.nc"
    cache:
        True
    script:
        "src/scripts/psi.py"

rule sim:
    output:
        "src/data/uni_istar_draws.npy"
	"src/data/uni_noistar_draws.npy"
        "src/data/norm1_istar_draws.npy"
	"src/data/norm1_noistar_draws.npy"
        "src/data/norm2_istar_draws.npy"
	"src/data/norm2_noistar_draws.npy"
        "src/data/norm3_istar_draws.npy"
	"src/data/norm3_noistar_draws.npy"
    cache:
        True
    script:
        "src/scripts/sim.py"