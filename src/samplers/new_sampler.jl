"""
    MCMCSampler{T}

# Fields
- `trans::T`: the MCMC transition operator
- `S::Int`: number of samples to take
- `gid::Int`: a parameter IMPROVE THIS DOCUMENTATION.
"""
struct MCMCSampler{T<:TransitionOperator} <: InferenceAlgorithm
    trans::T
    S::Int
    gid::Int
    MCMCSampler(trans::T, S::Int) = new{T}(trans, S, 0)
    MCMCSampler(sampler::MCMCSampler, gid::Int) = new{T}(sampler.trans, sampler.S, gid)
end

"""
    step(model::Function, trans::TransitionOperator, vi::VarInfo)::VarInfo

Perform a single sampling step on `model` using `trans`.

# Arguments
- `model::Function` - a `Turing` model
- `trans::TransitionOperator` - a Metropolis-Hastings transition operator
- `vi::VarInfo` - the usual `VarInfo` object
"""
function step(model::Function, trans::TransitionOperator, vi::VarInfo) end

"""
    initialize(model::Function, trans::TransitionOperator, vi::VarInfo)::VarInfo

Perform the initialization of `model` specified by `trans`.

# Arguments
- `model::Function` - a `Turing` model
- `trans::TransitionOperator` - a Metropolis-Hastings transition operator
- `vi::VarInfo` - the usual `VarInfo` object
"""
function initialize(model::Function, trans::TransitionOperator, vi::VarInfo) end

# Retained for interop with old code.
function Sampler(sampler::MCMCSampler)

    info = Dict{Symbol, Any}()
    info[:accept_his] = []
    info[:total_eval_num] = 0
    info[:proposal_ratio] = 0.0
    info[:prior_prob] = 0.0
    info[:violating_support] = false

    return Sampler(alg, info)
end

function __sanity_check(sampler::MCMCSampler)
    if gid(sampler) === 0 && !isempty(space(sampler))
        @assert issubset(Turing._compiler_[:pvars], space(sampler)) "[$alg_str] symbols " *
            "specified to samplers ($alg.space) doesn't cover the model parameters " *
            "($(Turing._compiler_[:pvars]))"
        if Turing._compiler_[:pvars] != space(sampler)
            warn("[$alg_str] extra parameters specified by samplers don't exist in " *
                "model: $(setdiff(alg.space, Turing._compiler_[:pvars]))")
        end
    end
end

"""
    sample(
        model::Function,
        sampler::MCMCSampler;
        save_state::Bool=false,
        resume_from=nothing,
        reuse_spl_n::Bool=false,
    )

Iterate `sampler` over `model`.

# Arguments:
- `model::Function`:
- `sampler::MCMCSampler`:

# Keywords
- `save_state::Bool=false`:
- `resume_from::Nothing=nothing`:
- `reuse_spl_n::Bool=false`:
- `show_progress::Bool=true`:
"""
function sample(
    model::Function,
    sampler::MCMCSampler;
    save_state::Bool=false,
    resume_from=nothing,
    reuse_spl_n::Int=0,
    show_progress::Bool=true,
)
    spl = reuse_spl_n > 0 ? resume_from.info[:spl] : Sampler(alg)

    # Initialization
    N = reuse_spl_n > 0 ? reuse_spl_n : sampler.S
    samples = [Sample(1 / n, Dict{Symbol, Any}())]
    vi = resume_from === nothing ?
            Base.invokelatest(model, VarInfo(), nothing) :
            resume_from.info[:vi]

    spl.alg.gid == 0 && runmodel(model, vi, spl)

    # Iterate `TransitionOperator`.
    if show_progress
        spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0)
    end

    time_total, vi = 0.0, initialize(model, sampler.trans, vi)
    for n in 1:N
        dprintln(2, "$alg_str stepping...")

        time_elapsed = @elapsed vi = step(model, spl, vi, s == 1)
        time_total += time_elapsed

        samples[n] = spl.info[:accept_his][end] ? Sample(vi, spl).value : samples[i - 1]
        samples[i].value[:elapsed] = time_elapsed

        if PROGRESS ProgressMeter.next!(spl.info[:progress]) end
    end

    println("[$alg_str] Finished with")
    println("  Running time        = $time_total;")
    println("  Accept rate         = $(mean(spl.info[:accept_his]));")

    if resume_from != nothing   # concat samples
        unshift!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain
    if save_state               # save state
        save!(c, spl, model, vi)
    end

    return c
end

################## VarInfo-related crap. Should be able to be removed. #####################

# NOTE: vi[vview] will just return what insdie vi (no transformations applied)
Base.getindex(vi::VarInfo, trans::TransitionOperator) = getval(vi, getranges(vi, trans))
function Base.setindex!(vi::VarInfo, val::Any, trans::TransitionOperator)
    return setval!(vi, val, getranges(vi, trans))
end
function getranges(vi::VarInfo, trans::TransitionOperator)
    if :cache_updated ∉ keys(trans.info)
        trans.info[:cache_updated] = CACHERESET
    end
    if haskey(trans.info, :ranges) && (trans.info[:cache_updated] & CACHERANGES) > 0
        trans.info[:ranges]
    else
        trans.info[:cache_updated] = trans.info[:cache_updated] | CACHERANGES
        trans.info[:ranges] = union(map(i -> vi.ranges[i], getidcs(vi, trans))...)
    end
end
