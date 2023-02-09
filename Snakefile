rule psi:
    output:
        "src/data/all_randinc.nc"
    cache:
        True
    script:
        "src/scripts/psi.py"

rule sim:
    output:
	directory("src/data/simulation")
    cache:
        True
    script:
        "src/scripts/sim.py"