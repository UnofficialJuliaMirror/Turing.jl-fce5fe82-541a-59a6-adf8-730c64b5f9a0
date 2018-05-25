using MacroTools: splitdef, postwalk

# Remove type information from function arguments obtained using `MacroTools.splitdef`
detype_args(args::Vector) = map(x->x isa Expr ? x.args[1] : x, args)

"""
    _model(def::Expr)

Internals for @model
"""
function _model(def::Expr)

    # Split up the definition using MacroTools.
    def_dict = splitdef(def)
    name, typed_args, body = def_dict[:name], def_dict[:args], def_dict[:body]

    # Get arguments with types removed.
    args = detype_args(typed_args)

    # Get the names of all of the things treated as random variables.
    rv_names = []
    postwalk(body) do x
        @capture(x, lhs_ ~ rhs_) && push!(rv_names, lhs)
        return x
    end

    # Explicitly disallow some things that we don't currently handle.
    length(def_dict[:kwargs]) != 0 && throw(error("Don't support kwargs at the minute."))

    # Generate some symbols.
    foo_name, l, type_name = gensym(), gensym(), gensym()

    # THIS IMPLEMENTATION IS SHIT. NEEDS TO BE IMPROVED. Currently allows fields of a
    # particular type to be abstract, thus screwing over performance. Easy to fix though.
    # Generate new type for the function based on its args.
    type_signature = :($type_name{$(def_dict[:whereparams]...)})
    type_fields = Expr(:block, typed_args...)
    model_type = Expr(:type, false, type_signature, type_fields)

    # Generate a method of `foo` which spits out the appropriate struct.
    foo_method = :(foo(x...) = $type_name(x...))

    # Generate `rand` method
    rand_signature = :(Base.rand($foo_name::$type_name))
    rand_body = Expr(:block,
        [:($x = $foo_name.$x) for x in args]...,
        postwalk(x->@capture(x, lhs_ ~ rhs_) ? :($lhs = rand($rhs)) : x, body).args...,
    )
    rand_method = Expr(:function, rand_signature, rand_body)

    # Generate `logpdf` method
    logpdf_signature = Expr(:call,
        :(Distributions.logpdf),
        :($foo_name::$type_name),
        rv_names...,
    )
    logpdf_compute = postwalk(body) do x
        if @capture(x, lhs_ ~ rhs_)
            return :($l += logpdf($rhs, $lhs))
        elseif @capture(x, return out__)
            return :(return $l)
        else
            return x
        end
    end
    logpdf_body = Expr(:block,
        :($l = 0.0), # THIS IS A BAD IDEA IN GENERAL! CHANGE THIS.
        logpdf_compute.args...
    )
    logpdf_method = Expr(:function, logpdf_signature, logpdf_body)

    return Expr(:block, model_type, foo_method, rand_method, logpdf_method)
end

"""
    model(def::Expr)

A model.
"""
macro model(def::Expr)
    return esc(_model(def))
end

# A possible Bayesian linear regressor.
@model function foo(X, σw::T, σn::Real) where T<:Real
    w ~ Normal(0, σw)
    f = X * w
    return f, w
end

expr = :(
function foo(X::AbstractMatrix{T}, σw::T, σn::Real) where T<:Real
    w ~ Normal(0, σw)
    f = X * w
    y ~ Normal(f, σn)
    return y
end
)

@model function foo(X::AbstractMatrix{T}, σw::T, σn::Real) where T<:Real
    w ~ Normal(0, σw)
    f = X * w
    y ~ Normal(f, σn)
    return y
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






