using Base.Meta: parse
using MacroTools: postwalk

#################
# Overload of ~ #
#################

# TODO: Replace this macro, see issue #514
"""
Usage: @VarName x[1,2][1+5][45][3]
  return: (:x,[1,2],6,45,3)
"""
macro VarName(expr::Union{Expr, Symbol})
    ex = deepcopy(expr)
    isa(ex, Symbol) && return var_tuple(ex)
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    inds = :(())
    while ex.head == :ref
        if length(ex.args) >= 2
            pushfirst!(inds.args, Expr(:vect, ex.args[2:end]...))
            end
        ex = ex.args[1]
        isa(ex, Symbol) && return var_tuple(ex, inds)
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end
function var_tuple(sym::Symbol, inds::Expr=:(()))
    return esc(:($(QuoteNode(sym)), $inds, $(QuoteNode(gensym()))))
end

"""
    generate_observe(obs, dist)

Generate an observe expression.
"""
generate_observe(obs, dist) = :(vi.logp += Turing.observe(sampler, $(dist), $(obs), vi))

"""
    generate_assume(variable, distribution, syms)

Generate an assume expression.
"""
function generate_assume(var, dist, syms)
    return quote
        varname = Turing.VarName(vi, $syms, "")
        if $(distribution) isa Vector
            $var, _lp = Turing.assume(sampler, $(dist), varname, $(var), vi)
        else
            $var, _lp = Turing.assume(sampler, $(dist), varname, vi)
        end
        vi.logp += _lp
    end
end

function generate_assume(var::Expr, dist)
    return quote
        sym, idcs, csym = @VarName $variable
        csym_str = string(Turing._compiler_[:name]) * string(csym)
        indexing = mapfoldl(string, *, idcs, init = "")
        varname = Turing.VarName(vi, Symbol(csym_str), sym, indexing)

        $var, _lp = Turing.assume(sampler, $(dist), varname, vi)
        vi.logp += _lp
    end
end

"""
    macro: @~ var Distribution()

Tilde notation macro. This macro constructs Turing.observe or
Turing.assume calls depending on the left-hand argument.
Note that the macro is interconnected with the @model macro and
assumes that a `compiler` struct is available.

Example:
```julia
@~ x Normal()
```
"""
macro ~(left, right)
    return esc(tilde(left, right))
end

tilde(left, right) = generate_observe(left, right)

function tilde(left::Symbol, right)
    # Check if left-hand side is a observation.
    if left in Turing._compiler_[:args]
        if !(left in Turing._compiler_[:dvars])
            @info " Observe - `$(left)` is an observation"
            push!(Turing._compiler_[:dvars], left)
        end

        return generate_observe(left, right)
    else
        # Assume it is a parameter.
        if !(left in Turing._compiler_[:pvars])
            msg = " Assume - `$(left)` is a parameter"
            if isdefined(Main, left)
                msg  *= " (ignoring `$(left)` found in global scope)"
            end

            @info msg
            push!(Turing._compiler_[:pvars], left)
        end

        sym, idcs, csym = @VarName(left)
        csym = Symbol(Turing._compiler_[:name], csym)
        syms = Symbol[csym, left]

        return generate_assume(left, right, syms)
    end
end

function tilde(left::Expr, right)
    vsym = getvsym(left)
    @assert vsym isa Symbol

    if vsym in Turing._compiler_[:args]
        if !(vsym in Turing._compiler_[:dvars])
            @info " Observe - `$(vsym)` is an observation"
            push!(Turing._compiler_[:dvars], vsym)
        end

        return generate_observe(left, right)
    else
        if !(vsym in Turing._compiler_[:pvars])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @info msg
            push!(Turing._compiler_[:pvars], vsym)
        end

        return generate_assume(left, right)
    end
end

#################
# Main Compiler #
#################

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

    # extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(fexpr)

    # function body of the model is empty
    if all(l -> isa(l, LineNumberNode), modeldef[:body].args)
        @warn("Model definition seems empty, still continue.")
    end

    # construct compiler dictionary
    compiler = Dict(
        :name => modeldef[:name],
        :closure_name => Symbol(modeldef[:name], :_model),
        :args => [],
        :kwargs => modeldef[:kwargs],
        :dvars => Set{Symbol}(),
        :pvars => Set{Symbol}()
    )

    # Manipulate the function arguments.
    fargs = map(
        arg->arg isa Symbol ? Expr(:kw, fargs[i], :nothing) : arg,
        deepcopy(vcat(modeldef[:args], modeldef[:kwargs])),
    )

    # translate all ~ occurences to macro calls in body.
    modeldef[:body] = translate(modeldef[:body])

    # Construct closure.
    closure = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :kwargs => [],
            :args => [
                :(vi::Turing.VarInfo),
                :(sampler::Turing.AnySampler)
            ],
            # Initialise logp in VarInfo.
            :body => Expr(:block, :(vi.logp = zero(Real)), modeldef[:body].args...)
        )
    )

    # Construct aliases.
    alias1 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [:(vi::Turing.VarInfo)],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(vi, Turing.SampleFromPrior()))
        )
    )

    alias2 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [:(sampler::Turing.AnySampler)],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), Turing.SampleFromPrior()))
        )
    )

    alias3 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), Turing.SampleFromPrior()))
        )
    )

    # Add definitions to the compiler.
    compiler[:closure] = closure
    compiler[:alias1] = alias1
    compiler[:alias2] = alias2
    compiler[:alias3] = alias3

    # Construct user function.
    modelfun = MacroTools.combinedef(
        Dict(
            :name => compiler[:name],
            :kwargs => [Expr(:kw, :compiler, compiler)],
            :args => fargs,
            :body => Expr(:block, 
                quote
                    Turing.eval(:(Turing._compiler_ = deepcopy($compiler)))
                    # Copy the expr of function definition and callbacks
                    closure = Turing._compiler_[:closure]
                    alias1 = Turing._compiler_[:alias1]
                    alias2 = Turing._compiler_[:alias2]
                    alias3 = Turing._compiler_[:alias3]
                    modelname = Turing._compiler_[:closure_name]
                end,
                # Insert argument values as kwargs to the closure
                map(data_insertion, fargs)...,
                # Eval the closure's methods globally and return it
                quote
                    Main.eval(Expr(:(=), modelname, closure))
                    Main.eval(alias1)
                    Main.eval(alias2)
                    Main.eval(alias3)
                    return $(compiler[:closure_name])
                end,
            )
        )
    )
    
    return esc(modelfun)
end


####################
# Helper functions #
####################

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

getvsym(s::Symbol) = s
function getvsym(expr::Expr)
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    return getvsym(expr.args[1])
end

tilde_expr(x, d) = Expr(:macrocall, Symbol("@~"), LineNumberNode(@__LINE__), x, d)
translate(ex::Expr) = postwalk(ex->@capture(ex, x_ ~ d_) ? tilde_expr(x, d) : ex, ex)
