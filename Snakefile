rule psi:
    output:
        "src/data/all_randinc.nc"
    cache:
        True
    script:
        "src/scripts/psi.py"