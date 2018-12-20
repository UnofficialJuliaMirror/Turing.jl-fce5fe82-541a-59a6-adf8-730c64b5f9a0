---
title: Library
permalink: /docs/library/
toc: true
---



<a id='Modelling-1'></a>

## Modelling

### <a id='Turing.@model' href='#Turing.@model'>#</a> **`Turing.@model`** &mdash; *Macro*.


```
@model(name, fbody)
```

Macro to specify a probabilistic model.

Example:

```julia
@model Gaussian(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt.(s))
    end
    return (s, m)
end
```

Compiler design: `sample(fname(x,y), sampler)`.

```julia
fname(x=nothing,y=nothing; compiler=compiler) = begin
    ex = quote
        # Pour in kwargs for those args where value != nothing.
        fname_model(vi::VarInfo, sampler::Sampler; x = x, y = y) = begin
            vi.logp = zero(Real)
          
            # Pour in model definition.
            x ~ Normal(0,1)
            y ~ Normal(x, 1)
            return x, y
        end
    end
    return Main.eval(ex)
end
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/core/compiler.jl#L211-L246' class='documenter-source'>source</a><br>

### <a id='Turing.@~' href='#Turing.@~'>#</a> **`Turing.@~`** &mdash; *Macro*.


```
macro: @~ var Distribution()
```

Tilde notation macro. This macro constructs Turing.observe or Turing.assume calls depending on the left-hand argument. Note that the macro is interconnected with the @model macro and assumes that a `compiler` struct is available.

Example:

```julia
@~ x Normal()
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/core/compiler.jl#L130-L142' class='documenter-source'>source</a><br>


<a id='Samplers-1'></a>

## Samplers

### <a id='Turing.Sampler' href='#Turing.Sampler'>#</a> **`Turing.Sampler`** &mdash; *Type*.


```
Sampler{T}
```

Generic interface for implementing inference algorithms. An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file, then include that file at the end of this one.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/Turing.jl#L99-L110' class='documenter-source'>source</a><br>

### <a id='Turing.Gibbs' href='#Turing.Gibbs'>#</a> **`Turing.Gibbs`** &mdash; *Type*.


```
Gibbs(n_iters, algs...)
```

Compositional MCMC interface. Gibbs sampling combines one or more sampling algorithms, each of which samples from a different set of variables in a model.

Example:

```julia
@model gibbs_example(x) = begin
    v1 ~ Normal(0,1)
    v2 ~ Categorical(5)
        ...
end

# Use PG for a 'v2' variable, and use HMC for the 'v1' variable.
# Note that v2 is discrete, so the PG sampler is more appropriate
# than is HMC.
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```

Tips:

  * `HMC` and `NUTS` are fast samplers, and can throw off particle-based

methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including more particles in the particle sampler.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/gibbs.jl#L1-L26' class='documenter-source'>source</a><br>

### <a id='Turing.HMC' href='#Turing.HMC'>#</a> **`Turing.HMC`** &mdash; *Type*.


```
HMC(n_iters::Int, epsilon::Float64, tau::Int)
```

Hamiltonian Monte Carlo sampler.

Arguments:

  * `n_iters::Int` : The number of samples to pull.
  * `epsilon::Float64` : The leapfrog step size to use.
  * `tau::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x[1] ~ Normal(m, sqrt(s))
    x[2] ~ Normal(m, sqrt(s))
    return s, m
end

sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))
```

Tips:

  * If you are receiving gradient errors when using `HMC`, try reducing the

`step_size` parameter.

```julia
# Original step_size
sample(gdemo([1.5, 2]), HMC(1000, 0.1, 10))

# Reduced step_size.
sample(gdemo([1.5, 2]), HMC(1000, 0.01, 10))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/hmc.jl#L1-L45' class='documenter-source'>source</a><br>

### <a id='Turing.HMCDA' href='#Turing.HMCDA'>#</a> **`Turing.HMCDA`** &mdash; *Type*.


```
HMCDA(n_iters::Int, n_adapts::Int, delta::Float64, lambda::Float64)
```

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `n_adapts::Int` : Numbers of samples to use for adaptation.
  * `delta::Float64` : Target acceptance rate. 65% is often recommended.
  * `lambda::Float64` : Target leapfrop length.

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))
```

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research 15, no. 1 (2014): 1593-1623.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/hmcda.jl#L1-L37' class='documenter-source'>source</a><br>

### <a id='Turing.IPMCMC' href='#Turing.IPMCMC'>#</a> **`Turing.IPMCMC`** &mdash; *Type*.


```
IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)
```

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

  * `n_particles::Int` : Number of particles to use.
  * `n_iters::Int` : Number of iterations to employ.
  * `n_nodes::Int` : The number of nodes running SMC and CSMC.
  * `n_csmc_nodes::Int` : The number of CSMC nodes.

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), IPMCMC(100, 100, 4, 2))
```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/ipmcmc.jl#L1-L38' class='documenter-source'>source</a><br>

### <a id='Turing.IS' href='#Turing.IS'>#</a> **`Turing.IS`** &mdash; *Type*.


```
IS(n_particles::Int)
```

Importance sampling algorithm.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray) object.

Arguments:

  * `n_particles` is the number of particles to use.

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


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/is.jl#L1-L33' class='documenter-source'>source</a><br>

### <a id='Turing.MH' href='#Turing.MH'>#</a> **`Turing.MH`** &mdash; *Type*.


```
MH(n_iters::Int)
```

Metropolis-Hastings sampler.

Usage:

```julia
MH(100, (:m, (x) -> Normal(x, 0.1)))
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

chn = sample(gdemo([1.5, 2]), MH(1000))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/mh.jl#L1-L26' class='documenter-source'>source</a><br>

### <a id='Turing.NUTS' href='#Turing.NUTS'>#</a> **`Turing.NUTS`** &mdash; *Type*.


```
NUTS(n_iters::Int, n_adapts::Int, delta::Float64)
```

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.6j_max)
```

Arguments:

  * `n_iters::Int` : The number of samples to pull.
  * `n_adapts::Int` : The number of samples to use with adapatation.
  * `delta::Float64` : Target acceptance rate.

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.j_max, 2]), NUTS(1000, 200, 0.6j_max))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/nuts.jl#L1-L32' class='documenter-source'>source</a><br>

### <a id='Turing.PG' href='#Turing.PG'>#</a> **`Turing.PG`** &mdash; *Type*.


```
PG(n_particles::Int, n_iters::Int)
```

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray) object.

Usage:

```julia
PG(100, 100)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), PG(100, 100))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/pgibbs.jl#L1-L29' class='documenter-source'>source</a><br>

### <a id='Turing.PMMH' href='#Turing.PMMH'>#</a> **`Turing.PMMH`** &mdash; *Type*.


```
PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})
```

Particle independant Metropolis–Hastings and Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

  * `n_iters::Int` : Number of iterations to run.
  * `smc_alg:::SMC` : An [`SMC`]({{site.baseurl}}/docs/library/#Turing.SMC) algorithm to use.
  * `parameters_algs::Tuple{MH}` : An [`MH`]({{site.baseurl}}/docs/library/#Turing.MH) algorithm, which includes a

sample space specification.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/pmmh.jl#L1-L23' class='documenter-source'>source</a><br>

### <a id='Turing.SGHMC' href='#Turing.SGHMC'>#</a> **`Turing.SGHMC`** &mdash; *Type*.


```
SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)
```

Stochastic Gradient Hamiltonian Monte Carlo sampler.

Usage:

```julia
SGHMC(1000, 0.01, 0.1)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `learning_rate::Float64` : The learning rate.
  * `momentum_decay::Float64` : Momentum decay variable.

Example:

```julia
@model example begin
  ...
end

sample(example, SGHMC(1000, 0.01, 0.1))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/sghmc.jl#L1-L27' class='documenter-source'>source</a><br>

### <a id='Turing.SGLD' href='#Turing.SGLD'>#</a> **`Turing.SGLD`** &mdash; *Type*.


```
SGLD(n_iters::Int, epsilon::Float64)
```

Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `epsilon::Float64` : The scaling factor for the learing rate.

Example:

```julia
@model example begin
  ...
end

sample(example, SGLD(1000, 0.5))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/sgld.jl#L1-L26' class='documenter-source'>source</a><br>

### <a id='Turing.SMC' href='#Turing.SMC'>#</a> **`Turing.SMC`** &mdash; *Type*.


```
SMC(n_particles::Int)
```

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray) object.

Usage:

```julia
SMC(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), SMC(1000))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e16d07cf84c3cb44c669b67e53b596f081b0024b/src/samplers/smc.jl#L1-L29' class='documenter-source'>source</a><br>


<a id='Data-Structures-1'></a>

## Data Structures

### <a id='Libtask.TArray' href='#Libtask.TArray'>#</a> **`Libtask.TArray`** &mdash; *Type*.


```
TArray{T}(dims, ...)
```

Implementation of data structures that automatically perform copy-on-write after task copying.

If current*task is an existing key in `s`, then return `s[current*task]`. Otherwise, return`s[current*task] = s[last*task]`.

Usage:

```julia
TArray(dim)
```

Example:

```julia
ta = TArray(4)              # init
for i in 1:4 ta[i] = i end  # assign
Array(ta)                   # convert to 4-element Array{Int64,1}: [1, 2, 3, 4]
```


<a id='Utilities-1'></a>

## Utilities

### <a id='Libtask.tzeros' href='#Libtask.tzeros'>#</a> **`Libtask.tzeros`** &mdash; *Function*.


```
 tzeros(dims, ...)
```

Construct a distributed array of zeros. Trailing arguments are the same as those accepted by `TArray`.

```julia
tzeros(dim)
```

Example:

```julia
tz = tzeros(4)              # construct
Array(tz)                   # convert to 4-element Array{Int64,1}: [0, 0, 0, 0]
```


<a id='Index-1'></a>

## Index

- [`Libtask.TArray`]({{site.baseurl}}/docs/library/#Libtask.TArray)
- [`Turing.Gibbs`]({{site.baseurl}}/docs/library/#Turing.Gibbs)
- [`Turing.HMC`]({{site.baseurl}}/docs/library/#Turing.HMC)
- [`Turing.HMCDA`]({{site.baseurl}}/docs/library/#Turing.HMCDA)
- [`Turing.IPMCMC`]({{site.baseurl}}/docs/library/#Turing.IPMCMC)
- [`Turing.IS`]({{site.baseurl}}/docs/library/#Turing.IS)
- [`Turing.MH`]({{site.baseurl}}/docs/library/#Turing.MH)
- [`Turing.NUTS`]({{site.baseurl}}/docs/library/#Turing.NUTS)
- [`Turing.PG`]({{site.baseurl}}/docs/library/#Turing.PG)
- [`Turing.PMMH`]({{site.baseurl}}/docs/library/#Turing.PMMH)
- [`Turing.SGHMC`]({{site.baseurl}}/docs/library/#Turing.SGHMC)
- [`Turing.SGLD`]({{site.baseurl}}/docs/library/#Turing.SGLD)
- [`Turing.SMC`]({{site.baseurl}}/docs/library/#Turing.SMC)
- [`Turing.Sampler`]({{site.baseurl}}/docs/library/#Turing.Sampler)
- [`Libtask.tzeros`]({{site.baseurl}}/docs/library/#Libtask.tzeros)
- [`Turing.@model`]({{site.baseurl}}/docs/library/#Turing.@model)
- [`Turing.@~`]({{site.baseurl}}/docs/library/#Turing.@~)

