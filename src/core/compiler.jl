using Base.Meta: parse

"""
Usage: @varname x[1,2][1+5][45][3]
  return: VarName{:x}("[1,2][6][45][3]")
"""
macro varname(expr::Union{Expr, Symbol})
    ex = deepcopy(expr)
    (ex isa Symbol) && return quote
        Turing.VarName{$(QuoteNode(ex))}("")
    end
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    inds = :(())
    while ex.head == :ref
        if length(ex.args) >= 2
            strs = map(x -> :(string($x)), ex.args[2:end])
            pushfirst!(inds.args, :("[" * join($(Expr(:vect, strs...)), ",") * "]"))
        end
        ex = ex.args[1]
        isa(ex, Symbol) && return esc(quote
            Turing.VarName{$(QuoteNode(ex))}(foldl(*, $inds, init = ""))
        end)
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end

function wrong_dist_errormsg(l)
    return "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
        "Distributions on line $(l)."
end
function assert_dist(dist; msg)
    isdist = if isa(dist, AbstractVector)
        # Check if the right-hand side is a vector of distributions.
        all(d -> isa(d, Distribution), dist)
    else
        # Check if the right-hand side is a distribution.
        isa(dist, Distribution)
    end
    isdist || throw(ArgumentError(msg))
end

"""
    generate_observe(observation, dist, model_info)

Generate an observe expression for observation `observation` drawn from
a distribution or a vector of distributions (`dist`).
"""
function generate_observe(observation, dist, model_info)
    main_body_names = model_info[:main_body_names]
    vi = main_body_names[:vi]
    sampler = main_body_names[:sampler]
    return quote
        Turing.Core.assert_dist($dist, msg = $(wrong_dist_errormsg(@__LINE__)))
        $vi.logp += Turing.observe($sampler, $dist, $observation, $vi)
    end
end

"""
    generate_assume(var, dist, model_info)

Generate an assume expression for parameters `var` drawn from
a distribution or a vector of distributions (`dist`).
"""
function generate_assume(var::Union{Symbol, Expr}, dist, model_info)
    main_body_names = model_info[:main_body_names]
    vi = main_body_names[:vi]
    sampler = main_body_names[:sampler]
    varname = gensym(:varname)
    lp = gensym(:lp)
    return quote
        $varname = Turing.@varname $var
        Turing.Core.assert_dist($dist, msg = $(wrong_dist_errormsg(@__LINE__)))
        ($var, $lp) = if isa($dist, AbstractVector)
            Turing.assume($sampler, $dist, $varname, $var, $vi)
        else
            Turing.assume($sampler, $dist, $varname, $vi)
        end
        $vi.logp += $lp
    end
end

"""
    tilde(left, right, model_info)

The `tilde` function generates observation expression for data variables and assumption expressions for parameter variables, updating `model_info` in the process.
"""
function tilde(left, right, model_info)
    return generate_observe(left, right, model_info)
end
function tilde(left::Union{Symbol, Expr}, right, model_info)
    return _tilde(getvsym(left), left, right, model_info)
end

function _tilde(vsym, left, dist, model_info)
    main_body_names = model_info[:main_body_names]
    model_name = main_body_names[:model]

    model_info[:tent_dvars_list] = copy(model_info[:arg_syms])
    if vsym in model_info[:arg_syms]
        Turing.DEBUG && @debug " Observe - `$(vsym)` is an observation"
        return quote
            if Turing.in_pvars($(Val(vsym)), $model_name)
                $(generate_assume(left, dist, model_info))
            else
                $(generate_observe(left, dist, model_info))
            end
        end
    else
        # Assume it is a parameter.
        if !(vsym in model_info[:tent_pvars_list])
            Turing.DEBUG && @debug begin
                msg = " Assume - `$(vsym)` is a parameter"
                if isdefined(Main, vsym)
                    msg  *= " (ignoring `$(vsym)` found in global scope)"
                end
                msg
            end
            push!(model_info[:tent_pvars_list], vsym)
        end

        return generate_assume(left, dist, model_info)
    end
end

#################
# Main Compiler #
#################

"""
    @model(body)

Macro to specify a probabilistic model.

Example:

Model definition:

```julia
@model model_generator(x = default_x, y) = begin
    ...
end
```

Expanded model definition

```julia
# Allows passing arguments as kwargs
model_generator(; x = nothing, y = nothing)) = model_generator(x, y)
function model_generator(x = nothing, y = nothing)
    pvars, dvars = Turing.get_vars(Tuple{:x, :y}, (x = x, y = y))
    data = Turing.get_data(dvars, (x = x, y = y))
    defaults = Turing.get_default_values(dvars, (x = default_x, y = nothing))

    inner_function(sampler::Turing.AbstractSampler, model) = inner_function(model)
    function inner_function(model)
        return inner_function(Turing.VarInfo(), Turing.SampleFromPrior(), model)
    end
    function inner_function(vi::Turing.VarInfo, model)
        return inner_function(vi, Turing.SampleFromPrior(), model)
    end
    # Define the main inner function
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, model)
        local x
        if isdefined(model.data, :x)
            if model.data.x isa Type && (model.data.x <: AbstractFloat || model.data.x <: AbstractArray)
                x = Turing.Core.get_matching_type(sampler, vi, model.data.x)
            else
                x = model.data.x
            end
        else
            x = model_defaults.x
        end
        local y
        if isdefined(model.data, :y)
            if model.data.y isa Type && (model.data.y <: AbstractFloat || model.data.y <: AbstractArray)
                y = Turing.Core.get_matching_type(sampler, vi, model.data.y)
            else
                y = model.data.y
            end
        else
            y = model.defaults.y
        end

        vi.logp = zero(Real)
        ...
    end
    model = Turing.Model{pvars, dvars}(inner_function, data, defaults)
    return model
end
```

Generating a model: `model_generator(x_value)::Model`.
"""
macro model(input_expr)
    build_model_info(input_expr) |> translate_tilde! |> update_args! |> build_output
end

"""
    build_model_info(input_expr)

Builds the `model_info` dictionary from the model's expression.
"""
function build_model_info(input_expr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(input_expr)
    # Function body of the model is empty
    warn_empty(modeldef[:body])
    # Construct model_info dictionary

    # Extracting the argument symbols from the model definition
    arg_syms = map(modeldef[:args]) do arg
        # @model demo(x)
        if (arg isa Symbol)
            arg
        # @model demo(::Type{T}) where {T}
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            T
        # @model demo(x = 1)
        elseif MacroTools.@capture(arg, x_ = val_)
            x
        else
            throw(ArgumentError("Unsupported argument $arg to the `@model` macro."))
        end
    end
    args = map(modeldef[:args]) do arg
        if (arg isa Symbol)
            arg
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            if in(T, modeldef[:whereparams])
                S = :Any
            else
                ind = findfirst(x -> MacroTools.@capture(x, T1_ <: S_) && T1 == T, modeldef[:whereparams])
                ind != nothing || throw(ArgumentError("Please make sure type parameters are properly used. Every `Type{T}` argument need to have `T` in the a `where` clause"))
            end
            Expr(:kw, :($T::Type{<:$S}), Tval)
        else
            arg
        end
    end
    model_info = Dict(
        :name => modeldef[:name],
        :input_expr => input_expr,
        :main_body => modeldef[:body],
        :arg_syms => arg_syms,
        :args => args,
        :kwargs => modeldef[:kwargs],
        :whereparams => modeldef[:whereparams],
        :tent_dvars_list => Symbol[],
        :tent_pvars_list => Symbol[],
        :main_body_names => Dict(
            :vi => gensym(:vi),
            :sampler => gensym(:sampler),
            :model => gensym(:model),
            :pvars => gensym(:pvars),
            :dvars => gensym(:dvars),
            :data => gensym(:data),
            :inner_function => gensym(:inner_function),
            :defaults => gensym(:defaults)
        )
    )

    return model_info
end

"""
    translate_tilde!(model_info)

Translates ~ expressions to observation or assumption expressions, updating `model_info`.
"""
function translate_tilde!(model_info)
    ex = model_info[:main_body]
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, model_info) : x, ex)
    model_info[:main_body] = ex
    return model_info
end

"""
    update_args!(model_info)

Extracts default argument values and replaces them with `nothing`.
"""
function update_args!(model_info)
    fargs = Vector{Union{Symbol, Expr}}(model_info[:args])
    fargs_default_values = []
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            push!(fargs_default_values, :($(fargs[i]) = :nothing))
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        elseif isa(fargs[i], Expr) && MacroTools.@capture(fargs[i], T_::Type{<:S_} = Tval_)
            nothing
            #push!(fargs_default_values, :($T = $Tval))
        elseif isa(fargs[i], Expr) && fargs[i].head == :kw
            push!(fargs_default_values, :($(fargs[i].args[1]) = $(fargs[i].args[2])))
            fargs[i] = Expr(:kw, fargs[i].args[1], :nothing)
        else
            throw("Unsupported argument type $(fargs[i]).")
        end
    end
    if length(fargs_default_values) == 0
        tent_arg_defaults_nt = :(NamedTuple())
    else
        tent_arg_defaults_nt = :($(fargs_default_values...),)
    end
    model_info[:args] = fargs
    model_info[:tent_arg_defaults_nt] = tent_arg_defaults_nt
    return model_info
end

"""
    build_output(model_info)

Builds the output expression.
"""
function build_output(model_info)
    # Construct user-facing function
    main_body_names = model_info[:main_body_names]
    vi_name = main_body_names[:vi]
    model_name = main_body_names[:model]
    sampler_name = main_body_names[:sampler]
    data_name = main_body_names[:data]
    pvars_name = main_body_names[:pvars]
    dvars_name = main_body_names[:dvars]
    inner_function_name = main_body_names[:inner_function]
    defaults_name = main_body_names[:defaults]

    # Arguments with default values
    args = model_info[:args]
    # Argument symbols without default values
    arg_syms = model_info[:arg_syms]
    # Default values of the arguments
    whereparams = model_info[:whereparams]
    tent_arg_defaults_nt = model_info[:tent_arg_defaults_nt]
    # Model generator name
    outer_function_name = model_info[:name]
    # Tentative list of parameter variables
    tent_pvars_list = model_info[:tent_pvars_list]
    # Tentative list of data variables
    tent_dvars_list = model_info[:tent_dvars_list]
    # Main body of the model
    main_body = model_info[:main_body]

    if length(tent_dvars_list) == 0
        tent_dvars_nt = :(NamedTuple())
    else
        nt_type = Expr(:curly, :NamedTuple, 
            Expr(:tuple, QuoteNode.(tent_dvars_list)...), 
            Expr(:curly, :Tuple, [:(Turing.Core.get_type($x)) for x in tent_dvars_list]...)
        )
        tent_dvars_nt = Expr(:call, :(Turing.namedtuple), nt_type, Expr(:tuple, tent_dvars_list...))
    end
    #= Does the following for each of the tentative dvars
        local x
        if isdefined(model.data, :x)
            x = model.data.x
        else
            x = default_x
        end
    =#
    unwrap_data_expr = Expr(:block)
    for var in tent_dvars_list
        push!(unwrap_data_expr.args, quote
            local $var
            if isdefined($model_name.data, $(QuoteNode(var)))
                # The `Type{T}` arguments will always show up in the `model.data` named tuple.
                # This is because we are not replacing the default value of these arguments with `nothing` but keeping the same value defined by by the user, e.g. `Float64`.
                # So if the value is indeed correct, i.e. a type then it should just work
                # If the value is not a type, i.e. `::Type{T} = 1` and the user doesn't pass it something for this argument, then it will give an error when constructing the model, which is a correct Julia error.
                # This means that we don't need an expression for the default value of the `Type{T}` arguments in `model.defaults`.
                if $model_name.data.$var isa Type && ($model_name.data.$var <: AbstractFloat || $model_name.data.$var <: AbstractArray)
                    $var = Turing.Core.get_matching_type($sampler_name, $vi_name, $model_name.data.$var)
                else
                    $var = $model_name.data.$var
                end
            else
                $var = $model_name.defaults.$var
            end
        end)
    end
    return esc(quote
        # Allows passing arguments as kwargs
        $outer_function_name(;$(args...)) = $outer_function_name($(arg_syms...))
        # Outer function with `nothing` as default values except for Type{T} arguments
        function $outer_function_name($(args...))
            # Adds variables equal to `nothing` or `missing` or whose `eltype` is `Missing` to pvars and the rest to dvars
            # `tent_pvars_list` is the tentative list of pvars
            # `tent_dvars_nt` is the tentative named tuple of dvars
            $pvars_name, $dvars_name = Turing.get_vars($(Tuple{tent_pvars_list...}), $(tent_dvars_nt))
            # Filter out the dvars equal to `nothing` or `missing`, or whose `eltype` is `Missing`
            $data_name = Turing.get_data($dvars_name, $tent_dvars_nt)
            # Replace default values of inputs whose values are Vector{Missing} by a Vector{Real} of the same length as the input
            $defaults_name = Turing.get_default_values($tent_dvars_nt, $tent_arg_defaults_nt)

            # Define fallback inner functions
            function $inner_function_name($sampler_name::Turing.AbstractSampler, $model_name)
                return $inner_function_name($model_name)
            end
            function $inner_function_name($model_name)
                return $inner_function_name(Turing.VarInfo(), Turing.SampleFromPrior(), $model_name)
            end
            function $inner_function_name($vi_name::Turing.VarInfo, $model_name)
                return $inner_function_name($vi_name, Turing.SampleFromPrior(), $model_name)
            end

            # Define the main inner function
            function $inner_function_name(
                $vi_name::Turing.VarInfo,
                $sampler_name::Turing.AbstractSampler,
                $model_name
                )

                $unwrap_data_expr
                $vi_name.logp = zero(Real)
                $main_body
            end
            $model_name = Turing.Model{$pvars_name, $dvars_name}($inner_function_name, $data_name, $defaults_name)
            return $model_name
        end
    end)
end

# A hack for NamedTuple type specialization
# (T = Int,) has type NamedTuple{(:T,), Tuple{DataType}} by default
# With this function, we can make it NamedTuple{(:T,), Tuple{Type{Int}}}
# Both are correct, but the latter is what we want for type stability
get_type(::Type{T}) where {T} = Type{T}
get_type(t) = typeof(t)

# Replaces the default for `Vector{Missing}` inputs by `Vector{Real}` of the same length as the input.
@generated function get_default_values(tent_dvars_nt::Tdvars, tent_arg_defaults_nt::Tdefaults) where {Tdvars <: NamedTuple, Tdefaults <: NamedTuple}
    dvar_names = Tdvars.names
    dvar_types = Tdvars.parameters[2].types
    defaults = []
    for (n, t) in zip(dvar_names, dvar_types)
        if eltype(t) == Missing
            push!(defaults, :($n = similar(tent_dvars_nt.$n, Real)))
        elseif in(n, tent_arg_defaults_nt.names)
            push!(defaults, :($n = tent_arg_defaults_nt.$n))
        end
    end
    if length(defaults) == 0
        return :(NamedTuple())
    else
        return :($(defaults...),)
    end
end

@generated function get_vars(tent_pvars::Type{Tpvars}, tent_dvars_nt::NamedTuple) where {Tpvars <: Tuple}
    tent_pvar_syms = [Tpvars.types...]
    tent_dvar_syms = [tent_dvars_nt.names...]
    dvar_types = [tent_dvars_nt.types...]
    append!(tent_pvar_syms, [tent_dvar_syms[i] for i in 1:length(tent_dvar_syms) if dvar_types[i] == Nothing || dvar_types[i] == Missing || eltype(dvar_types[i]) == Missing])
    setdiff!(tent_dvar_syms, tent_pvar_syms)
    pvars_tuple = Tuple{tent_pvar_syms...}
    dvars_tuple = Tuple{tent_dvar_syms...}

    return :($pvars_tuple, $dvars_tuple)
end

@inline get_data(Tdvars::Type{<:Tuple}, nt::NamedTuple) = _get_data(Tuple(Tdvars.types), nt)
@inline function _get_data(dvars::Tuple, nt::NamedTuple)
    length(dvars) === 0 && return NamedTuple()
    n = dvars[1]
    f = getfield(nt, n)
    return ntmerge(namedtuple(NamedTuple{(n,), Tuple{get_type(f)}}, (f,)), _get_data(Base.tail(dvars), nt))
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return
end

####################
# Helper functions #
####################

getvsym(s::Symbol) = s
function getvsym(expr::Expr)
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    return getvsym(expr.args[1])
end


"""
    data(dict::Dict, keys::Vector{Symbol})
Construct a tuple with values filled according to `dict` and keys
according to `keys`.
"""
function data(dict::Dict, keys::Vector{Symbol})

    @assert mapreduce(k -> haskey(dict, k), &, keys)

    r = Expr(:tuple)
    for k in keys
        push!(r.args, Expr(:(=), k, dict[k]))
    end
    return Main.eval(r)
end

"""
    get_matching_type(spl, vi, ::Type{T}) where {T}
Get the specialized version of type `T` for sampler `spl`. For example,
if `T === Float64` and `spl::Hamiltonian`, the matching type is `eltype(vi[spl])`.
"""
get_matching_type(spl, vi, ::Type{T}) where {T} = T
