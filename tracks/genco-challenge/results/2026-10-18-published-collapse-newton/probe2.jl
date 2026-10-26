# Rebuilt datakit PF input (flat start, as datakit's fast path); compute_ac_pf with the default trust-region method and
# with :newton. Compare each converged solution to the PUBLISHED label (label csv) and to our physical root (phys csv).
using PowerModels, DelimitedFiles
PowerModels.silence()
const JSON = PowerModels.JSON
case, labelf, physf, outf, tag = ARGS
net0 = PowerModels.parse_file(case)
rd(f) = (m = readdlm(f, ',', Float64, '\n'; skipstart=1); Dict(Int(m[i,1]) => (m[i,2], m[i,3]) for i in 1:size(m,1)))
lab = rd(labelf); phys = isfile(physf) ? rd(physf) : nothing
for (name, kw) in [("trust_region_default", Dict()), ("newton", Dict(:method=>:newton))]
    err = nothing
    t = @elapsed r = try compute_ac_pf(deepcopy(net0); kw...) catch e; err = sprint(showerror, e); nothing end
    rec = Dict{String,Any}("tag"=>tag, "variant"=>name, "t_s"=>t)
    # audit R1-F3: an exception is recorded as such, not as non-convergence
    if err !== nothing
        rec["status"] = "julia_exception"; rec["err"] = first(err, 300)
    elseif r["termination_status"] != true
        rec["status"] = "not_converged"
    else
        s = r["solution"]["bus"]
        vm = Dict(parse(Int,k)=>v["vm"] for (k,v) in s); va = Dict(parse(Int,k)=>v["va"] for (k,v) in s)
        dl = maximum(abs(vm[i]*cis(va[i]) - lab[i][1]*cis(lab[i][2])) for i in keys(vm))
        rec["status"] = "converged"; rec["min_vm"] = minimum(values(vm)); rec["max_dV_vs_label"] = dl
        if phys !== nothing
            rec["max_dV_vs_phys"] = maximum(abs(vm[i]*cis(va[i]) - phys[i][1]*cis(phys[i][2])) for i in keys(vm))
        end
    end
    open(outf, "a") do io; println(io, JSON.json(rec)); end
end
