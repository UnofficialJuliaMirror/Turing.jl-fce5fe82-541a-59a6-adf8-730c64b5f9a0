---
title: Guide
---

# Guide


## Basics


### Introduction

A probabilistic program is Julia code wrapped in a `@model` macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using task copying. To help avoid those we provide a Turing-safe datatype `TArray` that can be used to create mutable arrays in Turing programs.


To specify distributions of random variables, Turing programs should use the `~` notation:


`x ~ distr` where `x` is a symbol and `distr` is a distribution. If `x` is undefined in the model function, inside the probabilistic program, this puts a random variable named `x`, distributed according to `distr`, in the current scope. `distr` can be a value of any type that implements `rand(distr)`, which samples a value from the distribution `distr`. If `x` is defined, this is used for conditioning in a style similar to [Anglican](https://probprog.github.io/anglican/index.html) (another PPL). In this case, `x` is an observed value, assumed to have been drawn from the distribution `distr`. The likelihood is computed using `logpdf(distr,y)`. The observe statements should be arranged so that every possible run traverses all of them in exactly the same order. This is equivalent to demanding that they are not placed inside stochastic control flow.


Available inference methods include  Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC), Hamiltonian Monte Carlo with Dual Averaging (HMCDA) and The No-U-Turn Sampler (NUTS).


### Simple Gaussian Demo

Below is a simple Gaussian demo illustrate the basic usage of Turing.jl.


```julia
# Import packages.
using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end
```


Note: As a sanity check, the expectation of `s` is 49/24 (2.04166666...) and the expectation of `m` is 7/6 (1.16666666...).


We can perform inference by using the `sample` function, the first argument of which is our probabalistic program and the second of which is a sampler. More information on each sampler is located in the [API]({{site.baseurl}}/docs/library).


```julia
#  Run sampler, collect results.
c1 = sample(gdemo(1.5, 2), SMC(), 1000)
c2 = sample(gdemo(1.5, 2), PG(10), 1000)
c3 = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
c4 = sample(gdemo(1.5, 2), Gibbs(PG(10, :m), HMC(0.1, 5, :s)), 1000)
c5 = sample(gdemo(1.5, 2), HMCDA(0.15, 0.65), 1000)
c6 = sample(gdemo(1.5, 2), NUTS(0.65), 1000)
```


The `MCMCChains` module (which is re-exported by Turing) provides plotting tools for the `Chain` objects returned by a `sample` function. See the [MCMCChains](https://github.com/TuringLang/MCMCChains.jl) repository for more information on the suite of tools available for diagnosing MCMC chains.


```julia
# Summarise results
describe(c3)

# Plot results
plot(c3)
savefig("gdemo-plot.png")
```


The arguments for each sampler are:


  * SMC: number of particles.
  * PG: number of particles, number of iterations.
  * HMC: number of samples, leapfrog step size, leapfrog step numbers.
  * Gibbs: number of samples, component sampler 1, component sampler 2, ...
  * HMCDA: number of samples, total leapfrog length, target accept ratio.
  * NUTS: number of samples, target accept ratio.


For detailed information on the samplers, please review Turing.jl's [API]({{site.baseurl}}/docs/library) documentation.


### Modelling Syntax Explained


Using this syntax, a probabilistic model is defined in Turing. The model function generated by Turing can then be used to condition the model onto data. Subsequently, the sample function can be used to generate samples from the posterior distribution.


In the following example, the defined model is conditioned to the date (arg*1 = 1, arg*2 = 2) by passing (1, 2) to the model function.


```julia
@model model_name(arg_1, arg_2) = begin
  ...
end
```


The conditioned model can then be passed onto the sample function to run posterior inference.


```julia
model_func = model_name(1, 2)
chn = sample(model_func, HMC(..)) # Perform inference by sampling using HMC.
```


The returned chain contains samples of the variables in the model.


```julia
var_1 = mean(chn[:var_1]) # Taking the mean of a variable named var_1.
```


The key (`:var_1`) can be a `Symbol` or a `String`. For example, to fetch `x[1]`, one can use `chn[Symbol("x[1]")` or `chn["x[1]"]`.


The benefit of using a `Symbol` to index allows you to retrieve all the parameters associated with that symbol. As an example, if you have the parameters `"x[1]"`, `"x[2]"`, and `"x[3]"`, calling `chn[:x]` will return a new chain with only `"x[1]"`, `"x[2]"`, and `"x[3]"`.


Turing does not have a declarative form. More generally, the order in which you place the lines of a `@model` macro matters. For example, the following example works:


```julia
# Define a simple Normal model with unknown mean and variance.
@model model_function(y) = begin
  s ~ Poisson(1)
  y ~ Normal(s, 1)
  return y
end

sample(model_function(10), SMC(), 100)
```


But if we switch the `s ~ Poisson(1)` and `y ~ Normal(s, 1)` lines, the model will no longer sample correctly:


```julia
# Define a simple Normal model with unknown mean and variance.
@model model_function(y) = begin
  y ~ Normal(s, 1)
  s ~ Poisson(1)
  return y
end

sample(model_function(10), SMC(), 100)
```



### Sampling Multiple Chains


If you wish to run multiple chains, you can do so with the `mapreduce` function:


```julia
# Replace num_chains below with however many chains you wish to sample.
chains = mapreduce(c -> sample(model_fun, sampler), chainscat, 1:num_chains)
```


The `chains` variable now contains a `Chains` object which can be indexed by chain. To pull out the first chain from the `chains` object, use `chains[:,:,1]`.


Having multiple chains in the same object is valuable for evaluating convergence. Some diagnostic functions like `gelmandiag` require multiple chains.


Please note that Turing does not have native support for chains sampled in parallel.



### Sampling from an Unconditional Distribution (The Prior)


Turing allows you to sample from a declared model's prior by calling the model without specifying inputs or a sampler. In the below example, we specify a `gdemo` model which returns two variables, `x` and `y`. The model includes `x` and `y` as arguments, but calling the function without passing in `x` or `y` means that Turing's compiler will assume they are missing values to draw from the relevant distribution. The `return` statement is necessary to retrieve the sampled `x` and `y` values.


```julia
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return x, y
end
```


Assign the function without inputs to a variable, and Turing will produce a sample from the prior distribution.


```julia
# Samples from p(x,y)
g_prior_sample = gdemo()
g_prior_sample()
```


Output:


```
(0.685690547873451, -1.1972706455914328)
```


### Sampling from a Conditional Distribution (The Posterior)


#### Using `Missing`


Values that are `missing` are treated as parameters to be estimated. This can be useful if you want to simulate draws for that parameter, or if you are sampling from a conditional distribution. Turing v0.6.7 supports the following syntax:


```julia
@model gdemo(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# Treat x as a vector of missing values.
model = gdemo(fill(missing, 2))
c = sample(model, HMC(0.01, 5), 500)
```


The above case tells the model compiler the dimensions of the values it needs to generate. The generated values for `x` can be extracted from the `Chains` object using `c[:x]`.


Currently, Turing does not support vector-valued inputs containing mixed `missing` and non-missing values, i.e. vectors of type `Union{Missing, T}` where `T` is any type. The following **will not work**:


```julia
@model gdemo(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# Warning: This will provide an error!
model = gdemo([missing, 2.4])
c = sample(model, HMC(0.01, 5), 500)
```


If this is functionality you need, you may need to define each parameter as a separate variable, as below:


```julia
@model gdemo(x1, x2) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    # Note that x1 and x2 are no longer vector-valued.
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
end

# Equivalent to sampling p( x1 | x2 = 1.5).
model = gdemo(missing, 1.5)
c = sample(model, HMC(0.01, 5), 500)
```


#### Using Argument Defaults


Turing models can also be treated as generative by providing default values in the model declaration, and then calling that model without arguments.


Suppose we wish to generate data according to the model


$$ s \sim \text{InverseGamma}(2, 3) \\
m \sim \text{Normal}(0, \sqrt{s}) \\
x_i \sim \text{Normal}(m, \sqrt{s}), \space i = 1\dots10 $$


Each $$x_i$$ can be generated by Turing. In the model below, if `x` is not provided when the function is called, `x` will default to `Vector{Real}(undef, 10)`, a 10-element array of `Real` values. The sampler will then treat `x` as a parameter and generate those quantities.


```julia
using Turing

# Declare a model with a default value.
@model generative(x = Vector{Real}(undef, 10)) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
    return s, m
end
```


This model can be called in a traditional fashion, with an argument vector of any size:


```julia
# The values 1.5 and 2.0 will be observed by the sampler.
m = generative([1.5,2.0])
chain = sample(m, HMC(0.01, 5), 1000)
```


We can generate observations by providing no arguments in the `sample` call.


```julia
# This call will generate a vector of 10 values
# every sampler iteration.
generated = sample(generative(), HMC(0.01, 5), 1000)
```


The generated quantities can then be accessed by pulling them out of the chain. To access all the `x` values, we first subset the chain using `generated[:x]`


```julia
xs = generated[:x]
```


You can access the values inside a chain several ways:


1. Turn them into a `DataFrame` object
2. Use their raw `AxisArray` form
3. Create a three-dimensional `Array` object


```julia
# Convert to a DataFrame.
DataFrame(xs)

# Retrieve an AxisArray.
xs.value

# Retrieve a basic 3D Array.
xs.value.data
```


#### What to Use as a Default Value


Currently, the actual *value* of the default argument does not matter. Only the dimensions and type of a non-atomic value are relevant. Turing uses default values to pre-allocate vectors when they are treated as parameters, because if the value is not provided, the model will not know the size or type of a vector. Consider the following model:


```julia
@model generator(x) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  for i in 1:length(x)
      x[i] ~ Normal(m, sqrt(s))
  end
  return s, m
end
```


If we are trying to generate random random values from the `generator` model and we call `sample(generator(), HMC(1000, 0.01, 5))`, we will receive an error. This is because there is no way to determine `length(x)`, whether `x` is a vector, and the type of the values in `x`.


A sensible default value might be:


```julia
@model generator(x = zeros(10)) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  for i in 1:length(x)
      x[i] ~ Normal(m, sqrt(s))
  end
  return s, m
end
```


In this case, the model compiler can now determine that `x` is a `Vector{Float64,1}` of length 10, and the model will work as intended. It doesn't matter what the values in the vector are — at current, `x` will be treated as a parameter if it assumes its default value, i.e. no value was provided in the function call for that variable.


The element type of the vector (or matrix) should match the type of the random variable, `<: Integer` for discrete random variables and `<: AbstractFloat` for continuous random variables. Moreover, if the continuous random variable is to be sampled using a Hamiltonian sampler, the vector's element type needs to be `Real` to enable auto-differentiation through the model which uses special number types that are sub-types of `Real`. Finally, when using a particle sampler, a `TArray` should be used.


## Beyond the Basics


### Compositional Sampling Using Gibbs


Turing.jl provides a Gibbs interface to combine different samplers. For example, one can combine an `HMC` sampler with a `PG` sampler to run inference for different parameters in a single model as below.


```julia
@model simple_choice(xs) = begin
  p ~ Beta(2, 2)
  z ~ Bernoulli(p)
  for i in 1:length(xs)
    if z == 1
      xs[i] ~ Normal(0, 1)
    else
      xs[i] ~ Normal(2, 1)
    end
  end
end

simple_choice_f = simple_choice([1.5, 2.0, 0.3])

chn = sample(simple_choice_f, Gibbs(HMC(0.2, 3, :p), PG(20, :z)), 1000)
```


The `Gibbs` sampler can be used to specify unique automatic differentation backends for different variable spaces. Please see the [Automatic Differentiation]({{site.baseurl}}/docs/using-turing/autodiff) article for more.


For more details of compositional sampling in Turing.jl, please check the corresponding [paper](http://xuk.ai/assets/aistats2018-turing.pdf).


### Working with MCMCChains.jl


Turing.jl wraps its samples using `MCMCChains.Chain` so that all the functions working for `MCMCChains.Chain` can be re-used in Turing.jl. Two typical functions are `MCMCChains.describe` and `MCMCChains.plot`, which can be used as follows for an obtained chain `chn`. For more information on `MCMCChains`, please see the [GitHub repository](https://github.com/TuringLang/MCMCChains.jl).


```julia
describe(chn) # Lists statistics of the samples.
plot(chn) # Plots statistics of the samples.
```


There are numerous functions in addition to `describe` and `plot` in the `MCMCChains` package, such as those used in convergence diagnostics. For more information on the package, please see the [GitHub repository](https://github.com/TuringLang/MCMCChains.jl).


### Working with Libtask.jl


The [Libtask.jl](https://github.com/TuringLang/Libtask.jl) library provides write-on-copy data structures that are safe for use in Turing's particle-based samplers. One data structure in particular is often required for use – the [`TArray`](http://turing.ml/docs/library/#Libtask.TArray). The following sampler types require the use of a `TArray` to store distributions:


  * `IPMCMC`
  * `IS`
  * `PG`
  * `PMMH`
  * `SMC`


If you do not use a `TArray` to store arrays of distributions when using a particle-based sampler, you may experience errors.


Here is an example of how the `TArray` (using a `TArray` constructor function called `tzeros`) can be applied in this way:


```julia
# Turing model definition.
@model BayesHmm(y) = begin
    # Declare a TArray with a length of N.
    s = tzeros(Int, N)
    m = Vector{Real}(undef, K)
    T = Vector{Vector{Real}}(undef, K)
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 0.01)
    end

    # Draw from a distribution for each element in s.
    s[1] ~ Categorical(K)
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
    return (s, m)
end;
```


### Changing Default Settings


Some of Turing.jl's default settings can be changed for better usage.


#### AD Chunk Size


ForwardDiff (Turing's default AD backend) uses forward-mode chunk-wise AD. The chunk size can be manually set by `setchunksize(new_chunk_size)`; alternatively, use an auto-tuning helper function `auto_tune_chunk_size!(mf::Function, rep_num=10)`, which will profile various chunk sizes. Here `mf` is the model function, e.g. `gdemo(1.5, 2)`, and `rep_num` is the number of repetitions during profiling.


#### AD Backend


Since [#428](https://github.com/TuringLang/Turing.jl/pull/428), Turing.jl supports `Tracker` as backend for reverse mode autodiff. To switch between `ForwardDiff.jl` and `Tracker`, one can call function `setadbackend(backend_sym)`, where `backend_sym` can be `:forward_diff` or `:reverse_diff`.


For more information on Turing's automatic differentiation backend, please see the [Automatic Differentiation]({{site.baseurl}}/docs/using-turing/autodiff) article.


#### Progress Meter


Turing.jl uses ProgressMeter.jl to show the progress of sampling, which may lead to slow down of inference or even cause bugs in some IDEs due to I/O. This can be turned on or off by `turnprogress(true)` and `turnprogress(false)`, of which the former is set as default.