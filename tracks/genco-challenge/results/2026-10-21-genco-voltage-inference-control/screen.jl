using PowerModels, Ipopt
PowerModels.silence()
for f in ARGS
    d = PowerModels.parse_file(f)
    r = PowerModels.solve_ac_opf(d, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 500))
    println("SCREEN ", splitext(basename(f))[1], " ", r["termination_status"])
end
