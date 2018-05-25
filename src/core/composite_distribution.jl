"""
    _model(def::Expr)

Internals for @model
"""
function _model(def::Expr)

    # Split up the definition using MacroTools.
    def_dict = splitdef(def)

    # Explicitly disallow some things that we don't currently handle.
    length(def_dict[:kwargs]) != 0 && throw(error("Don't support kwargs at the minute."))

    # THIS IMPLEMENTATION IS SHIT. NEEDS TO BE IMPROVED. Currently allows fields of a
    # particular type to be abstract, thus screwing over performance. Easy to fix though.
    # Generate new type for the function based on its args.
    type_signature = :($(def_dict[:name]){$(def_dict[:whereparams]...)} <: Distribution)
    type_fields = Expr(:block, def_dict[:args]...)
    model_type = Expr(:type, false, type_signature, type_fields)

    # Generate logpdf by traversing body and playing around with the ~ - terms.


    return model_type
end

"""
    model(def::Expr)

A model.
"""
macro model(def::Expr)
    return esc(_model(def))
end

# A possible Bayesian linear regressor.
@model function foo(X::AbstractMatrix{T}, σw::T, σn::Real) where T<:Real
    w ~ Normal(0, σw)
    f = X * w
    return f, w
end


@model function model_f(X, σw)
    w ~ Normal(0, σw)
    f = X * w
    return f
end

@model function noise_model(f, σn)
    y ~ Normal(f, σn)
    return y
end

@model function lr(X, σw, σn)
    f ~ model_f(X, σw)
    y ~ noise_model(f, σn)
    return y
end

# @model function blr(X)
#     σw ~ Gamma...
#     σn ~ Gamma...
#     return lr(X, σw, σn)
# end

logpdf(blr(X, σw, σn), y)

dag(blr, X, σw, σn)

noise_model ∘ foo

bar = :(function foo(X::AbstractMatrix{T}, σw::T, σn::Real) where T<:Real
    w ~ Normal(0, σw)
    f = X * w
    y ~ Normal(f, σn)
end)

# Output should look something like this:
struct MangledStruct{Tθ}
    θ::Tθ
end

# Slow and ugly allocating implementation of rand. A good implementation would use rand!.
function rand(rng::AbstractRNG, f::MangledStruct)
    w = rand(rng, Normal(0, σw))
    f = X * w
    y = rand(rng, Normal(f, σn))
    return vcat(w, y)
end

# Probably sensible logpdf evaluation function.
function logpdf(foo::MangledStruct, x)
    w = x[:w]
    l = logpdf(Normal(0, foo.σw), w)
    f = foo.X * w
    y = x[:y]
    l += logpdf(Normal(f, foo.σ), y)
    return l
end






