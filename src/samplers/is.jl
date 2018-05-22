doc"""
    IS(n_particles::Int)

Importance sampling algorithm object.

# Fields
- `n_particles` is the number of particles to use

Usage:

```julia
IS(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  x[1] ~ Normal(m, sqrt.(s))
  x[2] ~ Normal(m, sqrt.(s))
  return s, m
end

sample(gdemo([1.5, 2]), IS(1000))
```
"""
struct IS <: InferenceAlgorithm
    n_particles::Int
end

function sample(model::Function, alg::IS)
    spl = Sampler(alg, Dict{Symbol, Any}())
    samples = [Sample(model(VarInfo(), spl)) for _ in 1:alg.n_particles]
    le = logsum(map(x->x[:lp], samples)) - log(n)
    return Chain(exp.(le), samples)
end

function assume(spl::Sampler{IS}, dist::Distribution, vn::VarName, vi::VarInfo)
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
    return r, zero(Real)
end

observe(spl::Sampler{IS}, dist::Distribution, value, vi::VarInfo) = logpdf(dist, value)
