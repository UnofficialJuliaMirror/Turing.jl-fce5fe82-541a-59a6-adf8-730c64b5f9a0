include("support/hmc_helper.jl")

doc"""
    HMC(n_samples::Int64, lf_size::Float64, lf_num::Int64)

Hamiltonian Monte Carlo sampler.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
@model example begin
  ...
end

sample(example, HMC(1000, 0.05, 10))
```
"""
immutable HMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
end

immutable StochHMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
  obs_frac  ::  Float64   # fraction of observations used for gradient
  cor_rate  ::  Float64   # rate of correlation
end

type HMCSampler{T} <: GradientSampler{T}
  alg         :: T                            # the HMC algorithm info
  model       :: Function                     # model function
  values      :: GradientInfo                 # container for variables
  dists       :: Dict{VarInfo, Distribution}  # variable to its distribution
  samples     :: Array{Sample}                # samples
  predicts    :: Dict{Symbol, Any}            # outputs
  info        :: Dict{Any, Any}               # store helpful infomation

  function init(alg::Union{HMC, StochHMC})
    values = GradientInfo()   # GradientInfo initialize logjoint as Dual(0)
    dists = Dict{VarInfo, Distribution}()
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    predicts = Dict{Symbol, Any}()
    info = Dict{Symbol, Any}(:curr_iter => 1)
    values, dists, samples, predicts, info
  end

  function HMCSampler(alg :: HMC, model :: Function)
    values, dists, samples, predicts, info = init(alg)
    new(alg, model, values, dists, samples, predicts, info)
  end

  function HMCSampler(alg :: StochHMC, model :: Function)
    values, dists, samples, predicts, info = init(alg)
    info[:total_obs] = 0
    info[:curr_obs] = 0
    info[:obs_set] = []
    info[:obs_for_iter] = 0
    new(alg, model, values, dists, samples, predicts, info)
  end
end

function Base.run(spl :: Union{Sampler{HMC}, Sampler{StochHMC}})
  # Record the start time of HMC
  t_start = time()

  # Run the model for the first time
  dprintln(2, "initialising...")
  find_logjoint(spl.model, spl.values)

  # Store the first predicts
  spl.samples[1].value = deepcopy(spl.predicts)

  # Set parameters
  n, ϵ, τ = spl.alg.n_samples, spl.alg.lf_size, spl.alg.lf_num
  accept_num = 1        # the first samples is always accepted

  # HMC steps
  for i = 2:n
    dprintln(2, "HMC stepping...")
    spl.info[:curr_iter] = i

    dprintln(2, "recording old θ...")
    old_values = deepcopy(spl.values)

    dprintln(2, "sampling momentum...")
    p = Dict(k => randn(length(spl.values[k])) for k in keys(spl.values))

    dprintln(2, "recording old H...")
    oldH = find_H(p, spl.model, spl.values)

    dprintln(3, "first gradient...")
    val∇E = get_gradient_dict(spl.values, spl.model)

    dprintln(2, "leapfrog stepping...")
    for t in 1:τ  # do 'leapfrog' for each var
      spl.values, val∇E, p = leapfrog(spl.values, val∇E, p, ϵ, spl.model)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, spl.model, spl.values)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)  # accepted => store the new predcits
      spl.samples[i].value, accept_num = deepcopy(spl.predicts), accept_num + 1
    else                            # rejected => store the previous predcits
      spl.values, spl.samples[i] = old_values, spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function assume(spl :: Union{HMCSampler{HMC}, Sampler{StochHMC}}, dist :: Distribution, var :: VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  if spl.info[:curr_iter] == 1  # first time -> generate
    # Build {var -> dist} dictionary
    spl.dists[var] = dist

    # Sample a new prior
    dprintln(2, "sampling prior...")
    r = rand(dist)

    # Transform
    v = link(dist, r)        # X -> R
    val = vectorize(dist, v) # vectorize

    # Store the generated var
    addVarInfo(spl.values, var, val)
  else         # not first time -> replay
    # Replay varibale
    dprintln(2, "fetching values...")
    val = spl.values[var]
  end

  # Step 2 - Reconstruct variable
  dprintln(2, "reconstructing values...")
  val = reconstruct(dist, val)  # reconstruct
  val = invlink(dist, val)      # R -> X

  # Computing logjoint
  dprintln(2, "computing logjoint...")
  spl.values.logjoint += logpdf(dist, val, true)
  dprintln(2, "compute logjoint done")
  dprintln(2, "assume done")
  return val
end

function observe(spl :: HMCSampler{HMC}, d :: Distribution, value)
  dprintln(2, "observing...")
  if length(value) == 1
    spl.values.logjoint += logpdf(d, Dual(value))
  else
    spl.values.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

function observe(spl :: HMCSampler{StochHMC}, d :: Distribution, value)
  function observe_current()
    if length(value) == 1
      spl.values.logjoint += logpdf(d, Dual(value))
    else
      spl.values.logjoint += logpdf(d, map(x -> Dual(x), value))
    end
  end

  dprintln(2, "observing...")
  if spl.info[:curr_iter] == 1  # first time -> count total obs
    spl.info[:total_obs] += 1
    observe_current()
  else
    # Initialize if it is the first obs for this iteration
    if spl.info[:obs_for_iter] != spl.info[:curr_iter]
      spl.info[:obs_for_iter] = spl.info[:curr_iter]
      # Choose k from n without replacement
      spl.info[:obs_set] = shuffle(1:spl.info[:total_obs])[1:round(Int, spl.alg.obs_frac * spl.info[:total_obs])]
      spl.info[:curr_obs] = 0
    end

    # Count current obs
    spl.info[:curr_obs] += 1
    if spl.info[:curr_obs] > spl.info[:total_obs]
      spl.info[:curr_obs] = 1
    end

    # Compute logpdf if this obs is chosen
    if spl.info[:curr_obs] in spl.info[:obs_set]
      observe_current()
    end
  end
  dprintln(2, "observe done")
end

function predict(spl :: Union{HMCSampler{HMC}, Sampler{StochHMC}}, name :: Symbol, value)
  dprintln(2, "predicting...")
  spl.predicts[name] = realpart(value)
  dprintln(2, "predict done")
end

function sample(model :: Function, alg :: HMC)
  global sampler = HMCSampler{HMC}(alg, model);
  run(sampler)
end

function sample(model :: Function, alg :: StochHMC)
  global sampler = HMCSampler{StochHMC}(alg, model);
  run(sampler)
end
