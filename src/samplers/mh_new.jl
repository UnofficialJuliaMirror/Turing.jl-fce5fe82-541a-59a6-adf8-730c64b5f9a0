"""
    MH

# Fields
- `proposals::Dict{Symbol, Any}`: proposal distributions for variables in space
- `space::Set`: set of variables to be sampled
"""
struct MH <: MarkovTransitionOperator
    proposals::Dict{Symbol, Any}
    space::Set
    info::Dict{Symbol, Any}
    gid::Int
    function MH(space...)
        new_space, proposals = Set(), Dict{Symbol,Any}()

        # parse random variables with their hypothetical proposal
        for element in space
            if element isa Symbol
                push!(new_space, element)
            else
                @assert element[1] isa Symbol "[MH] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
                push!(new_space, element[1])
                proposals[element[1]] = element[2]
            end
        end

        return new(proposals, new_space, info, 0)
    end
end

alg_str(::MH) = "MH"

# Convenience constructor for backward compatibility. To go eventually.
MH(S::Int, space...) = MCMCSampler(MH(space...), S)

# MH proposal step.
function propose(model::Function, vi::VarInfo, mh::MH)
    mh.info[:proposal_ratio] = 0.0
    mh.info[:prior_prob] = 0.0
    mh.info[:violating_support] = false
    return runmodel(model, vi, mh)
end

function initialize(mode::Function, mh::MH, vi::VarInfo)
    push!(mh.info[:accept_his], true)
    return vi
end

function step(model::Function, mh::MH, vi::VarInfo)

    # Recompute the logpdf.
    mh.gid == 0 || runmodel(model, vi, nothing)
    old_θ, old_logp = copy(vi[mh]), getlogp(vi)

    dprintln(2, "Propose new parameters from proposals...")
    propose(model, vi, mh)

    dprintln(2, "computing accept rate α...")
    α = getlogp(vi) - old_logp + mh.info[:proposal_ratio]

    dprintln(2, "decide whether to accept...")
    if log(rand()) < α && !mh.info[:violating_support]  # accepted
        push!(mh.info[:accept_his], true)
    else                      # rejected
        push!(mh.info[:accept_his], false)
        vi[mh] = old_θ         # reset Θ
        setlogp!(vi, old_logp)  # reset logp
    end

    return vi
end

"""
    assume(spl::MH, dist::Distribution, vn::VarName, vi::VarInfo)

Do whatever it does that this function does. IMPROVE THIS DOCUMENTATION.

# Arguments
- `trans::MH`: MCMC transition operator
- `dist::Distribution`: assumed distribution over the RV
- `vn::VarName`: random variable name
- `vi::VarInfo`: usual `VarInfo` stuff
"""
function assume(trans::MH, dist::Distribution, vn::VarName, vi::VarInfo)
    if isempty(space(trans)) || vn.sym in space(trans)
        haskey(vi, vn) || error("[MH] does not handle stochastic existence yet")
        old_val = vi[vn]

        if vn.sym in keys(trans.proposals) # Custom proposal for this parameter
            proposal = trans.proposals[vn.sym](old_val)

            if proposal isa Distributions.Normal{<:Real} # If Gaussian proposal
                μ, σ = mean(proposal), std(proposal)
                lb, ub = support(dist).lb, support(dist).ub
                stdG = Normal()
                r = rand(TruncatedNormal(μ, σ, lb, ub))
                # cf http://fsaad.scripts.mit.edu/randomseed/metropolis-hastings-sampling-with-gaussian-drift-proposal-on-bounded-support/
                spl.info[:proposal_ratio] += log(cdf(stdG, (ub - old_val) / σ) - cdf(stdG, (lb - old_val) / σ))
                spl.info[:proposal_ratio] -= log(cdf(stdG, (ub - r) / σ) - cdf(stdG, (lb - r) / σ))

            else
                r = rand(proposal)
                if (r < support(dist).lb) | (r > support(dist).ub) # check if value lies in support
                    spl.info[:violating_support] = true
                    r = old_val
                end
                spl.info[:proposal_ratio] -= logpdf(proposal, r) # accumulate pdf of proposal
                reverse_proposal = spl.alg.proposals[vn.sym](r)
                spl.info[:proposal_ratio] += logpdf(reverse_proposal, old_val)
            end

        else # Prior as proposal
            r = rand(dist)
            spl.info[:proposal_ratio] += (logpdf(dist, old_val) - logpdf(dist, r))
        end

        spl.info[:prior_prob] += logpdf(dist, r) # accumulate prior for PMMH
        vi[vn] = vectorize(dist, r)
        setgid!(vi, trans.gid, vn)
    else
        r = vi[vn]
    end

    return r, logpdf(dist, r)
end
