# Notes on semantic constraints

It is necessary to consider a different set of semantics in a PPL than a traditional programming language. This results from the inherent tension between encapsulation / scoping, and the need for the programmer to be able to assign particular values and / or proposal distributions to random variables in a programme.

Below we provide motivating examples of problems that can arise if semantics aren't constrained, and propose semantic constraints to address these issues.

## Motivating Problem 1

Consider the following example:
```julia
@model function bar(α::Real, β::Real)
    s ~ InverseGamma(α, β)
    m ~ Normal(0, s)
    return m
end

@model function foo(α::Real, β::Real)
    m ~ bar(α, β)
    x1 ~ Normal(m, 1)
    x2 ~ Normal(m, 1)
    return x1, x2
end
```
To evaluate the `logpdf` of `bar`, we need to be able to set values for both `s` and `m`, which is simple enough:
```julia
logpdf(bar(1.0, 1.0), (1.0, 1.0))
```
Although `bar` doesn't explicitly return `s`, we might reasonably expect the programmer to be aware that `s` is a variable in the programme and to know that it is necessary to provide it's value to compute the `logpdf`. On the other hand, the evaluation of the `logpdf` of `foo` is not so straightforward. Suppose that someone other than the person who wrote `foo` wrote `bar`, then the person who wrote `foo` could not know that it is necessary to provide a value for `s` without inspecting `bar`. Moreover, in a language like Julia where it is possible to generate code on the fly via metaprogramming, it is not impossible that the body for `bar` will simply be unavailable. This is problematic if we wish to be able to encapsulate our code.

Somewhat separately, we could not write
```julia
b, f = bar(1.0, 1.0), foo(1.0, 1.0)
logpdf(b, rand(b))
logpdf(f, rand(f))
```
since in both cases `logpdf` requires arguments which are different than the data returned by the `rand` procedure implied in the example (i.e. that `rand` returns whatever is `return`ed in an `@model`). Although it is arguably a matter of taste, if one samples from a generative model one ought to be able to query the `logpdf` of said sample. Thus this example provides further motivation for thinking carefully about what we do and don't allow a model to do.



## Motivating Problem 2

Consider the following modification to `bar`:
```julia
@model function bar(α::Real, β::Real)
    m ~ InverseGamma(α, β)
    m ~ Normal(0, m)
    return m
end
```
in which we have reused the variable name `m`. Clearly we know how to generate from this model; we could write something like
```julia
function rand(bar::Bar)
    m = rand(InverseGamma(bar.α, bar.β))
    m = rand(Normal(0, m), m)
    return m
end
```
Moreover, we might propose to use the order in which the random variables are declared to define our `logpdf` function:
```julia
function logpdf(bar::Bar, args::Tuple)
    l = logpdf(InverseGamma(bar.α, bar.β), args[1])
    l += logpdf(Normal(0, args[1]), args[2])
    return l
end
```
In the above, we have assumed that the first element of `args` corresponds to a value for the "first" `m`, and that the second a value for the "second" `m`. This random-variable ordering approach is certainly viable, but it is brittle.

The following model is a bit more troublesome:
```julia
@model function baz()
    x ~ Normal(0, 1)
    for n in 1:N
        x ~ Normal(x, 1)
    end
    return x
end
```
In this example you would have to know the value of `N` to know the number of arguments to pass to `logpdf`. Again, you _could_ use variable ordering to deduce uniquely which variable corresponds to which value passed to a `logpdf` call, but it is (in this author's opinion) undeniably brittle.


## Proposed Semantic Constraints

We propose to place the following constraints in addition to Julia's usual semantic constraints:
1. All random variables (anything on the lhs of a `~`) must be returned from a `@model`.
2. Within any `@model`, random variable names must be unique.

These two constraints directly address the above problems. Although we have certainly sacrificed some of the nice encapsulation properties that one expects from a traditional programming language, this merely reflects that in a PPL one simply needs more information regarding the state of a programme.

One way to think about the kinds of encapsulation that we are able to achieve in the above model is that when writing a model (function) we must expose whatever quantities we generate, but we do not have to expose the precise mechanism for generating it.

Note that neither of the above constraints preclude programmes containing random variables with variables dimensionality, they simply preclude programmes with a variable number of random variables. Thus, for example, a Dirichlet process must be viewed as a single multivariate random variable whose dimensionality changes from sample to sample for the purposes of the proposed semantics.
