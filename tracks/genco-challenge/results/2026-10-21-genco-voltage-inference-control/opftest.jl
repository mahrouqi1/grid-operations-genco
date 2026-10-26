using PowerModels, Ipopt
PowerModels.silence()
for f in ARGS
    d = PowerModels.parse_file(f)
    r = PowerModels.solve_ac_opf(d, Ipopt.Optimizer)
    println(basename(f), " ", r["termination_status"], " ", r["objective"])
end
