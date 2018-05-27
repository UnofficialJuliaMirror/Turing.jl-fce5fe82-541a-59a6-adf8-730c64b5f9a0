using MacroTools
using MacroTools: splitdef, postwalk
using Distributions

# Remove type information from function arguments obtained using `MacroTools.splitdef`
detype_args(args::Vector) = map(x->x isa Expr ? x.args[1] : x, args)
dearg_types(args::Vector) = map(x->x isa Expr ? x.args[2] : :Any, args)

"""
    _model(def::Expr)

Internals for @model
"""
function _model(def::Expr)

    # Split up the definition using MacroTools.
    def_dict = splitdef(def)
    @assert length(def_dict[:kwargs]) === 0
    name, typed_args, body = def_dict[:name], def_dict[:args], def_dict[:body]
    args, arg_types = detype_args(typed_args), dearg_types(typed_args)

    wheres = def_dict[:whereparams]
    wheres_vals = map(x->x isa Expr ? x.args[1] : x, wheres)

    # Get the names of all of the things treated as random variables.
    rv_names = []
    postwalk(body) do x
        @capture(x, lhs_ ~ rhs_) && push!(rv_names, lhs)
        return x
    end

    # Generate some symbols.
    foo_name, l, type_name = gensym(), gensym(), gensym()

    # Construct a new type to store the arguments to foo.
    dummy_types = [gensym() for _ in eachindex(arg_types)]
    sig_types = vcat(
        reverse(wheres)...,
        [:($dummy<:$tp) for (dummy, tp) in zip(dummy_types, arg_types)],
    )
    fields = [:($arg_name::$dummy) for (arg_name, dummy) in zip(args, dummy_types)]

    # Inner constructor definition.
    inner_ctor_signature = :($type_name($(fields...)) where $(sig_types...))
    inner_ctor_body = :(new{$(reverse(wheres_vals)...), $(dummy_types...)}($(args...)))
    inner_ctor = Expr(:function, inner_ctor_signature, inner_ctor_body)

    type_signature = :($type_name{$(sig_types...)})
    type_body = Expr(:block, fields..., inner_ctor)
    model_type = Expr(:type, false, type_signature, type_body)

    # Generate a method of `foo` which spits out the appropriate struct.
    foo_signature = :($name($(typed_args...)) where $(wheres...))
    foo_body = Expr(:call, type_name, args...)
    foo_method = Expr(:function, foo_signature, foo_body)

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
        [:($x = $foo_name.$x) for x in args]...,
        :($l = 0.0),
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

@model function bar()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))
    return x1, x2
end

using BenchmarkTools
@benchmark rand(bar())
@benchmark logpdf(bar(), 1.0, 0.0, 0.5, -0.5)

#=
Outstanding Issues:
1. Limitations / brittleness of existing approach:
    The current approach is _possibly_ a bit limited / brittle as we are limited to having
    only `Symbol`s on the lhs of a `~`. There are definitely other things that you would
    want to do, such as assignment, which involves there being an expression on the rhs.
    We have also implicitly limited outselves to not doing in-place stuff.
2. Semantics for making assignments:
    Presumably we want to stick with the current mechanism in Turing. I would propose
    automatically adding keyword arguments via the @model macro which allow you to pass in
    data for arbitrary random variables.
=#


