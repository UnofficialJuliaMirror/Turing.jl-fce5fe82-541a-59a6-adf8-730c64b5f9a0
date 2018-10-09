"""
    @model(name, fbody)

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
"""
macro model(fexpr)

    modeldef = MacroTools.splitdef(fexpr)

    # Function body of the model is empty
    if all(l -> isa(l, LineNumberNode), modeldef[:body].args)
        @warn("Model definition seems empty, still continue.")
    end

    # Manipulate the function arguments.
    args = map(
        x->x isa Symbol ? Expr(:kw, x, :nothing) : x,
        vcat(modeldef[:args], modeldef[:kwargs]),
    )

    # Translate all ~ occurences in body to assume or observe calls.
    modeldef[:body] = MacroTools.postwalk(
        ex->@capture(ex, x_ ~ d_) ? tilde(x, d, modeldef[:name], args) : ex,
        modeldef[:body],
    )

    # Construct closure.
    closure_name = Symbol(modeldef[:name], :_model)
    closure = MacroTools.combinedef(
        Dict(
            :name => closure_name,
            :kwargs => [],
            :args => [
                :(vi::Turing.VarInfo=Turing.VarInfo()),
                :(sampler::Turing.AnySampler=Turing.SampleFromPrior())
            ],
            :body => Expr(:block, :(vi.logp = zero(Real)), modeldef[:body].args...)
        )
    )

    # Construct user function.
    modelfun = MacroTools.combinedef(
        Dict(
            :name => modeldef[:name],
            :args => args,
            :body => Expr(:block, 
                quote
                    closure = $closure
                    modelname = $closure_name
                end,
                # Insert argument values as kwargs to the closure
                map(data_insertion, args)...,
                # Eval the closure's methods globally and return it
                quote
                    Main.eval(Expr(:(=), modelname, closure))
                    return $closure_name
                end,
            )
        )
    )
    
    return esc(modelfun)
end

make_varname(expr::Symbol) = (expr, (), gensym())
function make_varname(expr::Expr)
    @assert expr.head ==  :ref "make_varname: Malformed variable name $(expr)"
    varname, indices = _make_varname(expr, Vector{Any}())
    index_expr = Expr(:tuple, [Expr(:vect, x) for x in indices]...)
    return varname, index_expr, gensym()
end
function _make_varname(expr::Expr, indices::Vector{Any})
    dump(expr)
    println(indices)
    if expr.args[1] isa Symbol
        return expr.args[1], vcat(Symbol(expr.args[2]), indices)
    else
        return _make_varname(expr.args[1], vcat(Symbol(expr.args[2]), indices))
    end
end

generate_observe(obs, dist) = :(vi.logp += Turing.observe(sampler, $(dist), $(obs), vi))

function generate_assume(var, dist, syms)
    return quote
        varname = Turing.VarName(vi, $syms, "")
        if $(dist) isa Vector
            $var, _lp = Turing.assume(sampler, $(dist), varname, $(var), vi)
        else
            $var, _lp = Turing.assume(sampler, $(dist), varname, vi)
        end
        vi.logp += _lp
    end
end

function generate_assume(var::Expr, dist, name)
    return quote
        sym, idcs, csym = $(make_varname(var))
        csym_str = string(Turing._compiler_[:name]) * string(csym)
        indexing = mapfoldl(string, *, idcs, init = "")
        varname = Turing.VarName(vi, Symbol(csym_str), sym, indexing)

        $var, _lp = Turing.assume(sampler, $(dist), varname, vi)
        vi.logp += _lp
    end
end

tilde(left, right, model_args) = generate_observe(left, right)

function tilde(left::Symbol, right, model_args)
    if left in model_args
        @info " Observe - `$(left)` is an observation"
        return generate_observe(left, right)
    else
        @info " Assume - `$(left)` is a parameter"

        sym, idcs, csym = make_varname(left)
        csym = Symbol(Turing._compiler_[:name], csym)
        syms = Symbol[csym, left]

        return generate_assume(left, right, syms)
    end
end

function tilde(left::Expr, right, model_args)
    vsym = getvsym(left)
    if vsym in model_args
        @info " Observe - `$(vsym)` is an observation"
        return generate_observe(left, right)
    else
        @info " Assume - `$(vsym)` is a parameter"
        return generate_assume(left, right)
    end
end

getvsym(s::Symbol) = s
function getvsym(expr::Expr)
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    return getvsym(expr.args[1])
end

function data_insertion(k)
    if k isa Symbol
        _k = k
    elseif k.head == :kw
        _k = k.args[1]
    else
        return :()
    end

    return quote
        if $_k == nothing
            # Notify the user if an argument is missing.
            @warn("Data `"*$(string(_k))*"` not provided, treating as parameter instead.")
        else
            if $(QuoteNode(_k)) ∉ Turing._compiler_[:args]
                push!(Turing._compiler_[:args], $(QuoteNode(_k)))
            end
            closure = Turing.setkwargs(closure, $(QuoteNode(_k)), $_k)
        end
    end
end

function setkwargs(fexpr::Expr, kw::Symbol, value)
    funcdef = MacroTools.splitdef(fexpr)
    push!(funcdef[:kwargs], Expr(:kw, kw, value))
    return MacroTools.combinedef(funcdef)
end
