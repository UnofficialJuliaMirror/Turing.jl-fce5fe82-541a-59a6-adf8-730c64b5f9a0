# Some notes on the problem of compiler design for Turing.

Below we discuss a proposal for a rethink of the internals of Turing. There is a mock implementation in `src/composite_distribution.jl` that is able to handle the basics. Crucially, this design approach is both compositional and highly performant. There are a couple of open questions regarding semantics that need to be addressed; these are examined at the end of the document.

## Design Fundamentals
The basic proposal is to make the core building block of Turing programme a subtype of
```julia
abstract type CompositeDistribution <: Distribution end
```
that is generated automatically. For example,
```julia
using Turing, Distributions, BenchmarkTools
using Turing: @model

@model function foo(α::Real, β::Real)
    s ~ InverseGamma(α, β)
    m ~ Normal(0, sqrt(s))
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
    return x1, x2
end
```
is transformed into something like the following:
```julia
struct Foo{T<:Real V<:Real} <: CompositeDistribution
    α::T
    β::V
end

function logpdf(foo::Foo, s::Real, m::Real, x1::Real, x2::Real)
    l = 0.0
    l += logpdf(InverseGamma(foo.α, foo.β), s)
    l += logpdf(Normal(0, sqrt(s)), m)
    l += logpdf(Normal(m, sqrt(s)), x1)
    l += logpdf(Normal(m, sqrt(s)), x2)
    return l
end

function rand(foo::Foo)
    s = rand(InverseGamma(foo.α, foo.β))
    m = rand(Normal(0, sqrt(s)))
    x1 = rand(Normal(m, sqrt(s)))
    x2 = rand(Normal(m, sqrt(s)))
    return x1, x2
end
```

`foo` is just a typical generative model to which we have passed in data `α` and `β`, and from which we return random variables `x1` and `x2`. An altered version of the `@model` macro has been used to programmatically transform the specified model into three distinct components:
- a `struct`, `Foo`, which is a `Distribution` that is constructed using the `data` (`α` and `β`) that are specified as arguments to `foo`, and
- `logpdf`, which accepts a `Foo` and values for _each_ random variable specified in the function `foo`, and
- `rand`, which accepts a `Foo` and generates from the prior.

The above list is almost certainly not a complete list of things that will be necessary to automatically 

## Compositionality
Consider the following minor re-write of the previous example:
```julia
@model function bar(α::Real, β::Real)
    s ~ InverseGamma(α, β)
    w ~ Normal(0, s)
    return s, w
end

@model function foo(α::Real, β::Real)
    s, m ~ bar(α, β)
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
    return s, m, x1, x2
end
```
We have simply encapsulated `m` inside another composite distribution, `bar`. Although this example is contrived, it demonstrates that compositionality of generative procedures comes (almost) for free with this kind of design, enabling code re-use.

## Performance

### `foo` vs hand-written `foo`

`foo` accrues no overhead relative to a hand-written version for random number generation:
```julia
function bar_rand_manual(α::Real, β::Real)
    s = rand(InverseGamma(α, β))
    w = rand(Normal(0, s))
    return s, w
end

julia> @benchmark bar_rand_manual(1.0, 1.0)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     93.691 ns (0.00% GC)
  median time:      106.888 ns (0.00% GC)
  mean time:        107.232 ns (0.00% GC)
  maximum time:     156.455 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     972

julia> @benchmark rand(bar(1.0, 1.0))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     95.912 ns (0.00% GC)
  median time:      106.510 ns (0.00% GC)
  mean time:        107.141 ns (0.00% GC)
  maximum time:     291.328 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     975
```
The same applies for `logpdf` computation:
```julia
function bar_logpdf_manual(α::Real, β::Real, s, w)
    l = 0.0
    l += logpdf(InverseGamma(α, β), s)
    l += logpdf(Normal(0, s), w)
    return l
end

julia> @benchmark bar_logpdf_manual(1.0, 1.0, 1.0, 1.0)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     48.531 ns (0.00% GC)
  median time:      48.550 ns (0.00% GC)
  mean time:        48.946 ns (0.00% GC)
  maximum time:     113.179 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     988

julia> @benchmark logpdf(bar(1.0, 1.0), (1.0, 1.0))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     48.940 ns (0.00% GC)
  median time:      49.127 ns (0.00% GC)
  mean time:        50.932 ns (0.00% GC)
  maximum time:     167.871 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     988
```

### `bar`
Benchmarks for `bar`. They are clearly as fast as the hand-written versions.
```julia
julia> @benchmark rand(bar(1.0, 1.0))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     95.658 ns (0.00% GC)
  median time:      106.645 ns (0.00% GC)
  mean time:        107.172 ns (0.00% GC)
  maximum time:     237.714 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     973

julia> @benchmark logpdf(bar(1.0, 1.0), (0.5, 4.0, -1.0, 1.0))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     51.329 ns (0.00% GC)
  median time:      51.509 ns (0.00% GC)
  mean time:        52.387 ns (0.00% GC)
  maximum time:     112.912 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     987
```

## Open Problems
There are several technical limitations in the above examples which are a consequence of the state of the implementation (it's not finished) as opposed to issues with the proposed approach in general. These include:
- It is currently necessary to always return a `Tuple` from one's `@model`s. Only a relatively small amount of work will be required to remove this constraint.
- We don't have a performant way to generate arrays of observations - all random-number generation is done by `rand` rather than `rand!`. Relatively straightforward to fix, but probably does require a bit of thought to get right. There are a few options here:
  + Leave it as it is -- a win for simplicity, but bad for performance.
  + Provide an in-place version of `~`, which allows the user to explicitly provide destination data for their random variable to avoid allocating when `rand` is called -- possibly a bit ugly, probably good for performance, not entirely clear how this should interact with, for example, `logpdf`.
  + Force all `rand` calls when `rand` produces arrays to be `rand!` calls, and automatically construct appropriately managed destination arrays -- probably a win for performance, but likely tricky.
  On the whole, I think I favour the third option, but it would be worth working through the consequences of both the second and third options before committing to anything.
- We don't currently propagate type constraints on `~`'d random variables. Again, straightforward to fix.

The major block (as I see it) is understanding how to do parameter / random variable handling properly. Clearly, we need a naming scheme similar to that proposed in [1] \(relating to the `c_sym` variables in the current Turing design\), but this aspect clearly requires some more thought. The current `logpdf` semantics are also sub-optimal.

Furthermore, we haven't really discussed conditioning yet. I feel quite strongly that it is desirable to avoid having to hard-code which data will be observed at the time when the models are written, as this has strong negative implications for code-reuseability. It is not, however, clear exactly what the correct semantics are; this requires some work.

## Bibliography

[1] - Wingate, David, Andreas Stuhlmüller, and Noah Goodman. "Lightweight implementations of probabilistic programming languages via transformational compilation." Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.
