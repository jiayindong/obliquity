rule psi:
    output:
        "src/data/all_randinc.nc"
    cache:
        True
    script:
        "src/scripts/psi.py"

rule sim:
    output:
	"src/data/simulation"
    cache:
        True
    script:
        "src/scripts/sim.py"

rule sim_plot:
    input:
        "src/data/simulation"
    output:
        "src/figures/simulation.pdf"
    script:
        "src/scripts/sim_plot.py"