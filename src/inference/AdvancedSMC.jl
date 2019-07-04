###
### Particle Filtering and Particle MCMC Samplers.
###

const Particle = Trace

"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T<:Particle, F, Tvals <: Array{T}, TlogW <: Array{Float64}}
    model :: F
    num_particles :: Int
    vals  :: Tvals
    logWs :: TlogW       # Log weights (Trace) or incremental likelihoods (ParticleContainer)
    logE  :: Float64     # Log model evidence
    # conditional :: Union{Nothing,Conditional} # storing parameters, helpful for implementing rejuvenation steps
    conditional :: Nothing # storing parameters, helpful for implementing rejuvenation steps
    n_consume :: Int # helpful for rejuvenation steps, e.g. in SMC2
end
ParticleContainer{T}(m) where T = ParticleContainer{T}(m, 0)
function ParticleContainer{T}(m, n::Int) where {T}
    ParticleContainer(m, n, Vector{T}(), Vector{Float64}(), 0.0, nothing, 0)
end

Base.collect(pc :: ParticleContainer) = pc.vals # prev: Dict, now: Array
Base.length(pc :: ParticleContainer)  = length(pc.vals)
Base.similar(pc :: ParticleContainer{T}) where T = ParticleContainer{T}(pc.model, 0)
# pc[i] returns the i'th particle
Base.getindex(pc :: ParticleContainer, i :: Real) = pc.vals[i]


# registers a new x-particle in the container
function Base.push!(pc :: ParticleContainer, p :: Particle)
    pc.num_particles += 1
    push!(pc.vals, p)
    push!(pc.logWs, 0)
    pc
end
Base.push!(pc :: ParticleContainer) = Base.push!(pc, eltype(pc.vals)(pc.model))

function Base.push!(pc :: ParticleContainer, n :: Int, spl :: Sampler, varInfo :: VarInfo)
    vals  = Vector{eltype(pc.vals)}(undef,n)
    logWs = zeros(eltype(pc.logWs), n)
    for i=1:n
        vals[i]  = Trace(pc.model, spl, varInfo)
    end
    append!(pc.vals, vals)
    append!(pc.logWs, logWs)
    pc.num_particles += n
    pc
end

# clears the container but keep params, logweight etc.
function Base.empty!(pc :: ParticleContainer)
    pc.num_particles = 0
    pc.vals  = Vector{Particle}()
    pc.logWs = Vector{Float64}()
    pc
end

# clones a theta-particle
function Base.copy(pc :: ParticleContainer)
    particles = collect(pc)
    newpc     = similar(pc)
    for p in particles
        newp = fork(p)
        push!(newpc, newp)
    end
    newpc.logE        = pc.logE
    newpc.logWs       = deepcopy(pc.logWs)
    newpc.conditional = deepcopy(pc.conditional)
    newpc.n_consume   = pc.n_consume
    newpc
end

# run particle filter for one step, return incremental likelihood
function Libtask.consume(pc :: ParticleContainer)
    @assert pc.num_particles == length(pc)
    # normalisation factor: 1/N
    _, z1      = weights(pc)
    n = length(pc.vals)

    particles = collect(pc)
    num_done = 0
    for i=1:n
        p = pc.vals[i]
        score = Libtask.consume(p)
        if score isa Real
            score += getlogp(p.vi)
            resetlogp!(p.vi)
            increase_logweight(pc, i, Float64(score))
        elseif score == Val{:done}
            num_done += 1
        else
            error("[consume]: error in running particle filter.")
        end
    end

    if num_done == length(pc)
        res = Val{:done}
    elseif num_done != 0
        error("[consume]: mis-aligned execution traces, num_particles= $(n), num_done=$(num_done).")
    else
        # update incremental likelihoods
        _, z2      = weights(pc)
        res = increase_logevidence(pc, z2 - z1)
        pc.n_consume += 1
        # res = increase_loglikelihood(pc, z2 - z1)
    end

    res
end

function weights(pc :: ParticleContainer)
    @assert pc.num_particles == length(pc)
    logWs = pc.logWs
    Ws = exp.(logWs .- maximum(logWs))
    logZ = log(sum(Ws)) + maximum(logWs)
    Ws = Ws ./ sum(Ws)
    return Ws, logZ
end

function effectiveSampleSize(pc :: ParticleContainer)
    Ws, _ = weights(pc)
    ess = 1.0 / sum(Ws .^ 2) # sum(Ws) ^ 2 = 1.0, because weights are normalised
end

increase_logweight(pc :: ParticleContainer, t :: Int, logw :: Float64) =
    (pc.logWs[t]  += logw)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
    (pc.logE += logw)


function resample!(
    pc :: ParticleContainer,
    randcat :: Function = Turing.Inference.resample_systematic,
    ref :: Union{Particle, Nothing} = nothing
)
    n1, particles = pc.num_particles, collect(pc)
    @assert n1 == length(particles)

    # resample
    Ws, _ = weights(pc)

    # check that weights are not NaN
    @assert !any(isnan.(Ws))

    n2    = isa(ref, Nothing) ? n1 : n1-1
    indx  = randcat(Ws, n2)

    # fork particles
    empty!(pc)
    num_children = zeros(Int,n1)
    map(i->num_children[i]+=1, indx)
    for i = 1:n1
        is_ref = particles[i] == ref
        p = is_ref ? fork(particles[i], is_ref) : particles[i]
        num_children[i] > 0 && push!(pc, p)
        for k=1:num_children[i]-1
            newp = fork(p, is_ref)
            push!(pc, newp)
        end
    end

    if isa(ref, Particle)
        # Insert the retained particle. This is based on the replaying trick for efficiency
        #  reasons. If we implement PG using task copying, we need to store Nx * T particles!
        push!(pc, ref)
    end

    pc
end

####################
# Transition Types #
####################

# used by PG, SMC, PMMH
struct ParticleTransition{T} <: AbstractTransition
    θ::Vector{T}
    lp::Float64
    le::Float64
    weight::Float64
end

abstract type ParticleInference <: InferenceAlgorithm end

transition_type(::Sampler{<:ParticleInference}) = ParticleTransition

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le,:weight]
end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC(1000)
```
"""
struct SMC{T, F} <: ParticleInference
    n_particles           ::  Int
    resampler             ::  F
    resampler_threshold   ::  Float64
    space                 ::  Set{T}
end

alg_str(spl::Sampler{SMC}) = "SMC"

SMC(n::Int) = SMC(n, resample_systematic, 0.5, Set())
function SMC(n_particles::Int, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SMC(n_particles, resample_systematic, 0.5, _space)
end

mutable struct ParticleState <: SamplerState
    logevidence        ::   Vector{Float64}
    vi                 ::   TypedVarInfo
    final_logevidence  ::   Float64
end

ParticleState(model::Model) = ParticleState(Float64[], VarInfo(model), 0.0)

function Sampler(alg::SMC, model::Model, s::Selector)
    dict = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{SMC,ParticleState}(alg, dict, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument.
    model::Turing.Model,
    spl::Sampler{SMC, ParticleState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl),
        typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    push!(particles, spl.alg.n_particles, spl, empty!(spl.state.vi))

    while consume(particles) != Val{:done}
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler)
      end
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.state.logevidence, particles.logE)

    params = particles[indx].vi[spl]
    spl.state.vi = particles[indx].vi
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return transition(spl.state.vi[spl], lp, Ws[indx], particles.logE)
end

####
#### Particle Gibbs sampler.
####

"""
    PG(n_particles::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
```
"""
struct PG{T, F} <: ParticleInference
    n_particles           ::    Int         # number of particles used
    resampler             ::    F           # function to resample
    space                 ::    Set{T}      # sampling space, emtpy means all
end

PG(n1::Int) = PG(n1, resample_systematic, Set())
function PG(n1::Int, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    alg = PG(n1, resample_systematic, _space)
    return alg
end

alg_str(spl::Sampler{PG}) = "PG"

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::PG, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{PG,ParticleState}(alg, info, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument for now.
    model::Turing.Model,
    spl::Sampler{PG, ParticleState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl), typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    ref_particle = isempty(spl.state.vi) ?
              nothing :
              forkr(Trace(model, spl, spl.state.vi))

    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    if ref_particle == nothing
        push!(particles, spl.alg.n_particles, spl, spl.state.vi)
    else
        push!(particles, spl.alg.n_particles-1, spl, spl.state.vi)
        push!(particles, ref_particle)
    end

    while consume(particles) != Val{:done}
        resample!(particles, spl.alg.resampler, ref_particle)
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.state.logevidence, particles.logE)

    # Extract the VarInfo from the retained particle.
    params = particles[indx].vi[spl]
    spl.state.vi = particles[indx].vi
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return transition(spl.state.vi[spl], lp, Ws[indx], particles.logE)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ParticleInference},
    ::Integer,
    ::Vector{ParticleTransition};
    kwargs...
)
    # Set the default for resuming the sampler.
    resume_from = get(kwargs, :resume_from, nothing)

    # Exponentiate the average log evidence.
    loge = exp.(mean(spl.state.logevidence))

    # If we already had a chain, grab it's logevidence.
    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = exp.(resume_from.logevidence)
        # Calculate new log-evidence
        pre_n = length(resume_from.info[:samples])
        loge = (log(pre_loge) * pre_n + log(loge) * n) / (pre_n + n)
    end

    # Store the logevidence.
    spl.state.final_logevidence = loge
end

function assume(  spl::Sampler{T},
                  dist::Distribution,
                  vn::VarName,
                  _::VarInfo
                ) where T<:Union{PG,SMC}

    vi = current_trace().vi
    if isempty(spl.alg.space) || vn.sym in spl.alg.space
        if ~haskey(vi, vn)
            r = rand(dist)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(dist)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, vi.num_produce)
        else
            updategid!(vi, vn, spl)
            r = vi[vn]
        end
    else # vn belongs to other sampler <=> conditionning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(dist)
            push!(vi, vn, r, dist, Selector(:invalid))
        end
        acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    end
    return r, zero(Real)
end

function assume(  spl::Sampler{A},
                  dists::Vector{D},
                  vn::VarName,
                  var::Any,
                  vi::VarInfo
                ) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing assume statement")
end

function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:Union{PG,SMC}
    produce(logpdf(dist, value))
    return zero(Real)
end

function observe( spl::Sampler{A},
                  ds::Vector{D},
                  value::Any,
                  vi::VarInfo
                ) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing observe statement")
end

####
#### Resampling schemes for particle filters
####

# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# Default resampling scheme
function resample(w::AbstractVector{<:Real}, num_particles::Integer=length(w))
    return resample_systematic(w, num_particles)
end

# More stable, faster version of rand(Categorical)
function randcat(p::AbstractVector{T}) where T<:Real
    r, s = rand(T), 1
    for j in eachindex(p)
        r -= p[j]
        if r <= zero(T)
            s = j
            break
        end
    end
    return s
end

function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer)

    M = length(w)

    # "Repetition counts" (plus the random part, later on):
    Ns = floor.(length(w) .* w)

    # The "remainder" or "residual" count:
    R = Int(sum(Ns))

    # The number of particles which will be drawn stocastically:
    M_rdn = num_particles - R

    # The modified weights:
    Ws = (M .* w - floor.(M .* w)) / M_rdn

    # Draw the deterministic part:
    indx1, i = Array{Int}(undef, R), 1
    for j in 1:M
        for k in 1:Ns[j]
            indx1[i] = j
            i += 1
        end
    end

    # And now draw the stocastic (Multinomial) part:
    return append!(indx1, rand(Distributions.sampler(Categorical(w)), M_rdn))
end

function resample_stratified(w::AbstractVector{<:Real}, num_particles::Integer)

    Q, N = cumsum(w), num_particles

    T = Array{Float64}(undef, N + 1)
    for i=1:N,
        T[i] = rand() / N + (i - 1) / N
    end
    T[N+1] = 1

    indx, i, j = Array{Int}(undef, N), 1, 1
    while i <= N
        if T[i] < Q[j]
            indx[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indx
end

function resample_systematic(w::AbstractVector{<:Real}, num_particles::Integer)

    Q, N = cumsum(w), num_particles

    T = collect(range(0, stop = maximum(Q)-1/N, length = N)) .+ rand()/N
    push!(T, 1)

    indx, i, j = Array{Int}(undef, N), 1, 1
    while i <= N
        if T[i] < Q[j]
            indx[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indx
end


#############################
# Common particle functions #
#############################

vnames(vi::VarInfo) = Symbol.(collect(keys(vi)))

"""
    transition(vi::AbstractVarInfo, spl::Sampler{<:Union{SMC, PG}}, weight::Float64)

Returns a basic TransitionType for the particle samplers.
"""
function transition(
        theta::Vector{T},
        lp::Float64,
        weight::Float64,
        le::Float64
) where {T<:Real}
    return ParticleTransition{T}(theta, lp, weight, le)
end
