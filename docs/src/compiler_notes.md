# Some notes on the problem of compiler design for Turing.

Below we discuss a proposal for a rethink of the internals of Turing. There is a mock implementation in `src/composite_distribution.jl` that is able to handle the basics. Crucially, this design approach is both compositional and highly performant. There are a couple of open questions regarding semantics that need to be addressed; these are examined at the end of the document.

## Design Fundamentals
The basic proposal is to make the core building block of Turing programme a subtype of
```julia
abstract type CompositeDistribution <: Distribution end
```
that is generated automatically. For example,
```julia
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
@model function bar(s::Real)
    w ~ Normal(0, s)
    return w
end

@model function foo(α::Real, β::Real)
    s ~ InverseGamma(α, β)
    m ~ bar(s)
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
    return x1, x2
end
```
We have simply encapsulated `m` inside another composite distribution, `bar`. Although this example is contrived, it demonstrates that compositionality of generative procedures comes for free with this kind of design, enabling code re-use.

## Performance

### `Normal` vs `foo`

`foo` accrues virtually no overhead relative to `Normal` for random number generation:
```julia
julia> @benchmark rand(Normal(0, 1))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     14.829 ns (0.00% GC)
  median time:      15.671 ns (0.00% GC)
  mean time:        15.757 ns (0.00% GC)
  maximum time:     37.405 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     998

julia> @benchmark rand(foo())
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     14.863 ns (0.00% GC)
  median time:      15.714 ns (0.00% GC)
  mean time:        16.223 ns (0.00% GC)
  maximum time:     48.624 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     998
```
Similarly, `foo` accrues virtually no overhead relative to `Normal` for `logpdf` evaluation:
```julia
julia> @benchmark logpdf(Normal(0, 1), 1.0)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     12.768 ns (0.00% GC)
  median time:      12.787 ns (0.00% GC)
  mean time:        12.897 ns (0.00% GC)
  maximum time:     32.556 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999

julia> @benchmark logpdf(foo(), 1.0)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     12.771 ns (0.00% GC)
  median time:      12.788 ns (0.00% GC)
  mean time:        12.921 ns (0.00% GC)
  maximum time:     56.056 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     999
```

### `bar`
Benchmarks for `bar`. They are clearly as fast as the hand-written versions.
```julia
julia> @benchmark rand(bar(1.0, 1.0))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     123.070 ns (0.00% GC)
  median time:      136.067 ns (0.00% GC)
  mean time:        136.763 ns (0.00% GC)
  maximum time:     269.462 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     949

julia> @benchmark logpdf(bar(1.0, 1.0), 0.5, 4.0, -1.0, 1.0)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     92.676 ns (0.00% GC)
  median time:      92.876 ns (0.00% GC)
  mean time:        94.507 ns (0.00% GC)
  maximum time:     267.597 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     954
```

## Open Problems
There are several technical limitations in the above examples which are a consequence of the state of the implementation (it's not finished) as opposed to issues with the proposed approach in general. These include:
- `~` with anything other than a plain symbol on the lhs doesn't work. For example the following won't currently work:
    + `x[i] ~ ...`,
    + `x, y, z ~ ...`,
- We don't have a performant way to generate arrays of observations - all random-number generation is done by `rand` rather than `rand!`.

The major block (as I see it) is understanding how to do parameter / random variable handling properly. Clearly, we need a naming scheme similar to that proposed in [1] \(relating to the `c_sym` variables in the current Turing design\), but this aspect clearly requires some more thought. The current `logpdf` semantics are also sub-optimal.

Furthermore, we haven't really discussed conditioning yet. I feel quite strongly that it is desirable to avoid having to hard-code which data will be observed at the time when the models are written, as this has strong negative implications for code-reuseability. It is not, however, clear exactly what the correct semantics are; this requires some work.

## Bibliography

[1] - Wingate, David, Andreas Stuhlmüller, and Noah Goodman. "Lightweight implementations of probabilistic programming languages via transformational compilation." Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.
