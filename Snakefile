rule psi:
    output:
        "src/data/all_noistar.nc"
    cache:
        True
    script:
        "src/scripts/psi.py"

rule psi_plot:
    input:
        rules.psi.output
    output:
        "src/figures/psi_dist.pdf"
    script:
        "src/scripts/psi_plot.py"

rule sim:
    output:
        directory("src/data/simulation")
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