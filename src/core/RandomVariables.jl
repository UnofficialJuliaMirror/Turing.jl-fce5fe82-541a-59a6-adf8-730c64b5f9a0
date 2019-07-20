module RandomVariables

using ...Turing: Turing, CACHERESET, CACHEIDCS, CACHERANGES, Model,
    AbstractSampler, Sampler, SampleFromPrior, SampleFromUniform,
    Selector, getspace
using ...Utilities: vectorize, reconstruct, reconstruct!
using Bijectors: SimplexDistribution, link, invlink
using Distributions

import ...Turing: runmodel!
import Base:    string,
                Symbol,
                ==,
                hash,
                in,
                getindex,
                setindex!,
                push!,
                show,
                isempty,
                empty!,
                getproperty,
                setproperty!,
                keys,
                haskey

export  VarName,
        AbstractVarInfo,
        VarInfo,
        UntypedVarInfo,
        getlogp,
        setlogp!,
        acclogp!,
        resetlogp!,
        set_retained_vns_del_by_spl!,
        is_flagged,
        unset_flag!,
        setgid!,
        updategid!,
        setorder!,
        istrans,
        link!,
        invlink!

####
#### Types for typed and untyped VarInfo
####


###########
# VarName #
###########
"""
```
struct VarName{sym}
    csym      ::    Symbol
    indexing  ::    String
    counter   ::    Int
end
```

A variable identifier. Every variable has a symbol `sym`, indices `indexing`, and
internal fields: `csym` and `counter`. The Julia variable in the model corresponding to
`sym` can refer to a single value or to a hierarchical array structure of univariate,
multivariate or matrix variables. `indexing` stores the indices that can access the
random variable from the Julia variable.

Examples:

- `x[1] ~ Normal()` will generate a `VarName` with `sym == :x` and `indexing == "[1]"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName` with `sym == :x` and
 `indexing == "[Colon(), 1]"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName` with `sym == :x` and
 `indexing == "[Colon(), 1][2]"`.
"""
struct VarName{sym}
    csym      ::    Symbol        # symbol generated in compilation time
    indexing  ::    String        # indexing
    counter   ::    Int           # counter of same {csym, uid}
end

abstract type AbstractVarInfo end

####################
# VarInfo metadata #
####################

"""
The `Metadata` struct stores some metadata about the parameters of the model. This helps
query certain information about a variable, such as its distribution, which samplers
sample this variable, its value and whether this value is transformed to real space or
not.

Let `md` be an instance of `Metadata`:
- `md.vns` is the vector of all `VarName` instances.
- `md.idcs` is the dictionary that maps each `VarName` instance to its index in
 `md.vns`, `md.ranges` `md.dists`, `md.orders` and `md.flags`.
- `md.vns[md.idcs[vn]] == vn`.
- `md.dists[md.idcs[vn]]` is the distribution of `vn`.
- `md.gids[md.idcs[vn]]` is the set of algorithms used to sample `vn`. This is used in
 the Gibbs sampling process.
- `md.orders[md.idcs[vn]]` is the number of `observe` statements before `vn` is sampled.
- `md.ranges[md.idcs[vn]]` is the index range of `vn` in `md.vals`.
- `md.vals[md.ranges[md.idcs[vn]]]` is the vector of values of corresponding to `vn`.
- `md.flags` is a dictionary of true/false flags. `md.flags[flag][md.idcs[vn]]` is the
 value of `flag` corresponding to `vn`.

To make `md::Metadata` type stable, all the `md.vns` must have the same symbol
and distribution type. However, one can have a Julia variable, say `x`, that is a
matrix or a hierarchical array sampled in partitions, e.g.
`x[1][:] ~ MvNormal(zeros(2), 1.0); x[2][:] ~ MvNormal(ones(2), 1.0)`, and is managed by
a single `md::Metadata` so long as all the distributions on the RHS of `~` are of the
same type. Type unstable `Metadata` will still work but will have inferior performance.
When sampling, the first iteration uses a type unstable `Metadata` for all the
variables then a specialized `Metadata` is used for each symbol along with a function
barrier to make the rest of the sampling type stable.
"""
struct Metadata{TIdcs <: Dict{<:VarName,Int}, TDists <: AbstractVector{<:Distribution}, TVN <: AbstractVector{<:VarName}, TVal <: AbstractVector{<:Real}, TGIds <: AbstractVector{Set{Selector}}}
    # Mapping from the `VarName` to its integer index in `vns`, `ranges` and `dists`
    idcs        ::    TIdcs # Dict{<:VarName,Int}

    # Vector of identifiers for the random variables, where `vns[idcs[vn]] == vn`
    vns         ::    TVN # AbstractVector{<:VarName}

    # Vector of index ranges in `vals` corresponding to `vns`
    # Each `VarName` `vn` has a single index or a set of contiguous indices in `vals`
    ranges      ::    Vector{UnitRange{Int}}

    # Vector of values of all the univariate, multivariate and matrix variables
    # The value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`
    vals        ::    TVal # AbstractVector{<:Real}

    # Vector of distributions correpsonding to `vns`
    dists       ::    TDists # AbstractVector{<:Distribution}

    # Vector of sampler ids corresponding to `vns`
    # Each random variable can be sampled using multiple samplers, e.g. in Gibbs, hence the `Set`
    gids        ::    TGIds # AbstractVector{Set{Selector}}

    # Number of `observe` statements before each random variable is sampled
    orders      ::    Vector{Int}

    # Each `flag` has a `BitVector` `flags[flag]`, where `flags[flag][i]` is the true/false flag value corresonding to `vns[i]`
    flags       ::    Dict{String, BitVector}
end

###########
# VarInfo #
###########

"""
```
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
```

A light wrapper over one or more instances of `Metadata`. Let `vi` be an instance of
`VarInfo`. If `vi isa VarInfo{<:Metadata}`, then only one `Metadata` instance is used
for all the sybmols. `VarInfo{<:Metadata}` is aliased `UntypedVarInfo`. If
`vi isa VarInfo{<:NamedTuple}`, then `vi.metadata` is a `NamedTuple` that maps each
symbol used on the LHS of `~` in the model to its `Metadata` instance. The latter allows
for the type specialization of `vi` after the first sampling iteration when all the
symbols have been observed. `VarInfo{<:NamedTuple}` is aliased `TypedVarInfo`.

Note: It is the user's responsibility to ensure that each "symbol" is visited at least
once whenever the model is called, regardless of any stochastic branching. Each symbol
refers to a Julia variable and can be a hierarchical array of many random variables, e.g. `x[1] ~ ...` and `x[2] ~ ...` both have the same symbol `x`.
"""
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
const UntypedVarInfo = VarInfo{<:Metadata}
const TypedVarInfo = VarInfo{<:NamedTuple}

function VarInfo(model::Model)
    vi = VarInfo()
    model(vi, SampleFromUniform())
    return TypedVarInfo(vi)
end

function VarInfo(old_vi::UntypedVarInfo, spl, x::AbstractVector)
    new_vi = deepcopy(old_vi)
    new_vi[spl] = x 
    return new_vi
end
function VarInfo(old_vi::TypedVarInfo, spl, x::AbstractVector)
    md = newmetadata(old_vi.metadata, spl, x, 0)
    VarInfo(md, Base.RefValue{eltype(x)}(old_vi.logp[]), Ref(old_vi.num_produce[]))
end
function newmetadata(metadata::NamedTuple{names}, spl, x, offset) where {names}
    # Check if the named tuple is empty and end the recursion
    length(names) === 0 && return ()
    # Take the first key in the NamedTuple
    f = names[1]
    mdf = getfield(metadata, f)
    syms = getspace(spl)
    if f ∈ syms || length(syms) == 0
        idcs = mdf.idcs
        vns = mdf.vns
        ranges = mdf.ranges
        vals = x[(offset+1):(offset+length(mdf.vals))]
        dists = mdf.dists
        gids = mdf.gids
        orders = mdf.orders
        flags = mdf.flags
        md = Metadata(idcs, vns, ranges, vals, dists, gids, orders, flags)
	    nt = NamedTuple{(f,)}((md,))
        return merge(nt, newmetadata(_tail(metadata), spl, x, offset+length(mdf.vals)))
    else
        md = getfield(metadata, f)
	    nt = NamedTuple{(f,)}((md,))
        return merge(nt, newmetadata(_tail(metadata), spl, x, offset))
    end
end

####
#### Internal functions
####

"""
`Metadata()`

Constructs an empty type unstable instance of `Metadata`.
"""
function Metadata()
    vals  = Vector{Real}()
    flags = Dict{String, BitVector}()
    flags["del"] = BitVector()
    flags["trans"] = BitVector()

    return Metadata(
        Dict{VarName, Int}(),
        Vector{VarName}(),
        Vector{UnitRange{Int}}(),
        vals,
        Vector{Distributions.Distribution}(),
        Vector{Set{Selector}}(),
        Vector{Int}(),
        flags
    )
end

"""
`empty!(meta::Metadata)`

Empties all the fields of `meta`. This is useful when using a sampling algorithm that
assumes an empty `meta`, e.g. `SMC`.
"""
function empty!(meta::Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end

    return meta
end

# Removes the first element of a NamedTuple. The pairs in a NamedTuple are ordered, so this is well-defined.
if VERSION < v"1.1"
    _tail(nt::NamedTuple{names}) where names = NamedTuple{Base.tail(names)}(nt)
else
    _tail(nt::NamedTuple) = Base.tail(nt)
end

const VarView = Union{Int, UnitRange, Vector{Int}}

"""
`getval(vi::UntypedVarInfo, vview::Union{Int, UnitRange, Vector{Int}})`

Returns a view `vi.vals[vview]`.
"""
getval(vi::UntypedVarInfo, vview::VarView) = view(vi.metadata.vals, vview)

"""
`setval!(vi::UntypedVarInfo, val, vview::Union{Int, UnitRange, Vector{Int}})`

Sets the value of `vi.vals[vview]` to `val`.
"""
setval!(vi::UntypedVarInfo, val, vview::VarView) = vi.metadata.vals[vview] = val
function setval!(vi::UntypedVarInfo, val, vview::Vector{UnitRange})
    if length(vview) > 0
        vi.metadata.vals[[i for arr in vview for i in arr]] = val
    end
    return val
end

"""
`getidx(vi::UntypedVarInfo, vn::VarName)`

Returns the index of `vn` in `vi.metadata.vns`.
"""
getidx(vi::UntypedVarInfo, vn::VarName) = vi.metadata.idcs[vn]

"""
`getidx(vi::TypedVarInfo, vn::VarName{sym})`

Returns the index of `vn` in `getfield(vi.metadata, sym).vns`.
"""
function getidx(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).idcs[vn]
end

"""
`getrange(vi::UntypedVarInfo, vn::VarName)`

Returns the index range of `vn` in `vi.metadata.vals`.
"""
getrange(vi::UntypedVarInfo, vn::VarName) = vi.metadata.ranges[getidx(vi, vn)]

"""
`getrange(vi::TypedVarInfo, vn::VarName{sym})`

Returns the index range of `vn` in `getfield(vi.metadata, sym).vals`.
"""
function getrange(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).ranges[getidx(vi, vn)]
end

"""
`getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})`

Returns all the indices of `vns` in `vi.metadata.vals`.
"""
function getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return mapreduce(vn -> getrange(vi, vn), vcat, vns, init=Int[])
end

"""
`getdist(vi::VarInfo, vn::VarName)`

Returns the distribution from which `vn` was sampled in `vi`.
"""
getdist(vi::UntypedVarInfo, vn::VarName) = vi.metadata.dists[getidx(vi, vn)]
function getdist(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).dists[getidx(vi, vn)]
end

"""
`getval(vi::VarInfo, vn::VarName)`

Returns the value(s) of `vn`. The values may or may not be transformed to Eucledian space.
"""
getval(vi::UntypedVarInfo, vn::VarName) = view(vi.metadata.vals, getrange(vi, vn))
function getval(vi::TypedVarInfo, vn::VarName{sym}) where sym
    view(getfield(vi.metadata, sym).vals, getrange(vi, vn))
end

"""
`setval!(vi::VarInfo, val, vn::VarName)`

Sets the value(s) of `vn` in `vi.metadata` to `val`. The values may or may not be
transformed to Eucledian space.
"""
setval!(vi::UntypedVarInfo, val, vn::VarName) = vi.metadata.vals[getrange(vi, vn)] = val
function setval!(vi::TypedVarInfo, val, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).vals[getrange(vi, vn)] = val
end

"""
`getval(vi::VarInfo, vns::Vector{<:VarName})`

Returns all the value(s) of `vns`. The values may or may not be transformed to Eucledian
space.
"""
getval(vi::UntypedVarInfo, vns::Vector{<:VarName}) = view(vi.metadata.vals, getranges(vi, vns))
function getval(vi::TypedVarInfo, vns::Vector{VarName{sym}}) where sym
    view(getfield(vi.metadata, sym).vals, getranges(vi, vns))
end

"""
`getall(vi::VarInfo)`

Returns the values of all the variables in `vi`. The values may or may not be
transformed to Eucledian space.
"""
getall(vi::UntypedVarInfo) = vi.metadata.vals
getall(vi::TypedVarInfo) = vcat(_getall(vi.metadata)...)
@inline function _getall(metadata::NamedTuple{names}) where {names}
    # Check if NamedTuple is empty and end recursion
    length(names) === 0 && return ()
    # Take the first key of the NamedTuple
    f = names[1]
    # Recurse using the remaining of `metadata`
    return (getfield(metadata, f).vals, _getall(_tail(metadata))...)
end

"""
`setall!(vi::VarInfo, val)`

Sets the values of all the variables in `vi` to `val`. The values may or may not be
transformed to Eucledian space.
"""
setall!(vi::UntypedVarInfo, val) = vi.metadata.vals .= val
setall!(vi::TypedVarInfo, val) = _setall!(vi.metadata, val)
@inline function _setall!(metadata::NamedTuple{names}, val, start = 0) where {names}
    # Check if `metadata` is empty and end recursion
    length(names) === 0 && return nothing
    # Take the first key of `metadata`
    f = names[1]
    # Set the `vals` of the current symbol, i.e. f, to the relevant portion in `val`
    vals = getfield(metadata, f).vals
    @views vals .= val[start + 1 : start + length(vals)]
    # Recurse using the remaining of `metadata` and the remaining of `val`
    return _setall!(_tail(metadata), val, start + length(vals))
end

"""
`getsym(vn::VarName)`

Returns the symbol of the Julia variable used to generate `vn`.
"""
getsym(vn::VarName{sym}) where sym = sym

"""
`getgid(vi::VarInfo, vn::VarName)`

Returns the set of sampler selectors associated with `vn` in `vi`.
"""
getgid(vi::UntypedVarInfo, vn::VarName) = vi.metadata.gids[getidx(vi, vn)]
function getgid(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).gids[getidx(vi, vn)]
end

"""
`settrans!(vi::VarInfo, trans::Bool, vn::VarName)`

Sets the `trans` flag value of `vn` in `vi`.
"""
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end

"""
`syms(vi::VarInfo)`

Returns a tuple of the unique symbols of random variables sampled in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(vn -> vn.sym, vi.metadata.vns)))  # get all symbols
syms(vi::TypedVarInfo) = keys(vi.metadata)

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
function _getidcs(vi::UntypedVarInfo, ::SampleFromPrior)
    return filter(i -> isempty(vi.metadata.gids[i]) , 1:length(vi.metadata.gids))
end
# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
function _getidcs(vi::TypedVarInfo, ::SampleFromPrior)
    return _getidcs(vi.metadata)
end
@inline function _getidcs(metadata::NamedTuple{names}) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first key/symbol
    f = names[1]
    # Get the first symbol's metadata
    meta = getfield(metadata, f)
    # Get all the idcs of vns with empty gid
    v = filter(i -> isempty(meta.gids[i]), 1:length(meta.gids))
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of metadata
    return merge(nt, _getidcs(_tail(metadata)))
end

# Get all indices of variables belonging to a given sampler
function _getidcs(vi::AbstractVarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    # Checks if cache is valid, i.e. no new pushes were made, to return the cached idcs
    # Otherwise, it recomputes the idcs and caches it
    if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
        spl.info[:idcs]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
        spl.info[:idcs] = _getidcs(vi, spl.selector, getspace(spl.alg))
    end
end
function _getidcs(vi::UntypedVarInfo, s::Selector, space::Tuple)
    filter(i -> (s in vi.metadata.gids[i] || isempty(vi.metadata.gids[i])) &&
        (isempty(space) || in(vi.metadata.vns[i], space)), 1:length(vi.metadata.gids))
end
function _getidcs(vi::TypedVarInfo, s::Selector, space::Tuple)
    return _getidcs(vi.metadata, s, space)
end
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@inline function _getidcs(metadata::NamedTuple{names}, s::Selector, space::Tuple) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first sybmol
    f = names[1]
    # Get the first symbol's metadata
    f_meta = getfield(metadata, f)
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    v = filter((i) -> (s in f_meta.gids[i] || isempty(f_meta.gids[i])) &&
        (isempty(space) || in(f_meta.vns[i], space)), 1:length(f_meta.gids))
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of metadata
    return merge(nt, _getidcs(_tail(metadata), s, space))
end

# Get all vns of variables belonging to spl
_getvns(vi::UntypedVarInfo, spl::AbstractSampler) = view(vi.metadata.vns, _getidcs(vi, spl))
function _getvns(vi::TypedVarInfo, spl::AbstractSampler)
    # Get a NamedTuple of the indices of variables belonging to `spl`, one entry for each symbol
    idcs = _getidcs(vi, spl)
    return _getvns(vi.metadata, idcs)
end
# Get a NamedTuple for all the `vns` of indices `idcs`, one entry for each symbol
@inline function _getvns(metadata::NamedTuple{names}, idcs) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first symbol
    f = names[1]
    # Get the vector of `vns` with symbol `f`
    v = getfield(metadata, f).vns[getfield(idcs, f)]
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of `metadata`
    return merge(nt, _getvns(_tail(metadata), idcs))
end

# Get the index (in vals) ranges of all the vns of variables belonging to spl
@inline function _getranges(vi::AbstractVarInfo, spl::Sampler)
    ## Uncomment the spl.info stuff when it is concretely typed, not Dict{Symbol, Any}
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    #if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
    #    spl.info[:ranges]
    #else
        #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        ranges = _getranges(vi, spl.selector, getspace(spl.alg))
        #spl.info[:ranges] = ranges
        return ranges
    #end
end
# Get the index (in vals) ranges of all the vns of variables belonging to selector `s` in `space`
@inline function _getranges(vi::AbstractVarInfo, s::Selector, space::Tuple=())
    _getranges(vi, _getidcs(vi, s, space))
end
@inline function _getranges(vi::UntypedVarInfo, idcs::Vector{Int})
    mapreduce(i -> vi.metadata.ranges[i], vcat, idcs, init=Int[])
end
@inline _getranges(vi::TypedVarInfo, idcs::NamedTuple) = _getranges(vi.metadata, idcs)
@inline function _getranges(metadata::NamedTuple{names}, idcs::NamedTuple) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first symbol
    f = names[1]
    # Collect the index ranges of all the vns with symbol `f`
    v = mapreduce(i -> getfield(metadata, f).ranges[i], vcat, getfield(idcs, f), init=Int[])
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of `metadata`
    return merge(nt, _getranges(_tail(metadata), idcs))
end

"""
`set_flag!(vi::VarInfo, vn::VarName, flag::String)`

Sets `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.metadata.flags[flag][getidx(vi, vn)] = true
end
function set_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)] = true
end


####
#### APIs for typed and untyped VarInfo
####

# VarName

"""
`VarName(csym, sym, indexing, counter)`
`VarName{sym}(csym::Symbol, indexing::String)`

Constructs a new instance of `VarName{sym}`
"""
VarName(csym, sym, indexing, counter) = VarName{sym}(csym, indexing, counter)
function VarName(csym::Symbol, sym::Symbol, indexing::String)
    # TODO: update this method when implementing the sanity check
    return VarName{sym}(csym, indexing, 1)
end
function VarName{sym}(csym::Symbol, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
    return VarName{sym}(csym, indexing, 1)
end

"""
`VarName(syms::Vector{Symbol}, indexing::String)`

Constructs a new instance of `VarName{syms[2]}`
"""
function VarName(syms::Vector{Symbol}, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
    return VarName{syms[2]}(syms[1], indexing, 1)
end

"""
`VarName(vn::VarName, indexing::String)`

Returns a copy of `vn` with a new index `indexing`.
"""
function VarName(vn::VarName, indexing::String)
    return VarName(vn.csym, vn.sym, indexing, vn.counter)
end

function getproperty(vn::VarName{sym}, f::Symbol) where {sym}
    return f === :sym ? sym : getfield(vn, f)
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

"""
`uid(vn::VarName)`

Returns a unique tuple identifier for `vn`.
"""
uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)

hash(vn::VarName) = hash(uid(vn))

==(x::VarName, y::VarName) = hash(uid(x)) == hash(uid(y))

function string(vn::VarName; all = true)
    if all
        return "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
    else
        return "$(vn.sym)$(vn.indexing)"
    end
end
function string(vns::Vector{<:VarName})
    return replace(string(map(vn -> string(vn), vns)), "String" => "")
end

"""
`Symbol(vn::VarName)`

Returns a `Symbol` represenation of the variable identifier `VarName`.
"""
Symbol(vn::VarName) = Symbol(string(vn, all=false))  # simplified symbol

"""
`in(vn::VarName, space::Set)`

Returns `true` if `vn`'s symbol is in `space` and `false` otherwise.
"""
function in(vn::VarName, space::Tuple)::Bool
    if vn.sym in space
        return true
    else
        # String representation of `vn`
        vn_str = string(vn, all=false)
        return _in(vn_str, space)
    end
end
@inline function _in(vn_str::String, space::Tuple)
    length(space) === 0 && return false
    el = space[1]
    # Collect expressions from space
    expr = isa(el, Expr) ? el : return _in(vn_str, Base.tail(space))
    # Filter `(` and `)` out and get a string representation of `exprs`
    expr_str = replace(string(expr), r"\(|\)" => "")
    # Check if `vn_str` is in `expr_strs`
    valid = occursin(expr_str, vn_str)
    return valid || _in(vn_str, Base.tail(space))
end

# VarInfo

"""
`runmodel!(model::Model, vi::AbstractVarInfo, spl::AbstractSampler)`

Samples from `model` using the sampler `spl` storing the sample and log joint
probability in `vi`.
"""
function runmodel!(model::Model, vi::AbstractVarInfo, spl::AbstractSampler = SampleFromPrior())
    setlogp!(vi, 0)
    if spl isa Sampler && haskey(spl.info, :eval_num)
        spl.info[:eval_num] += 1
    end
    model(vi, spl)
    return vi
end

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{Real}(0.0), Ref(0))

"""
`TypedVarInfo(vi::UntypedVarInfo)`

This function finds all the unique `sym`s from the instances of `VarName{sym}` found in
`vi.metadata.vns`. It then extracts the metadata associated with each symbol from the
global `vi.metadata` field. Finally, a new `VarInfo` is created with a new `metadata` as
a `NamedTuple` mapping from symbols to type-stable `Metadata` instances, one for each
symbol.
"""
function TypedVarInfo(vi::UntypedVarInfo)
    meta = vi.metadata
    new_metas = Metadata[]
    # Symbols of all instances of `VarName{sym}` in `vi.vns`
    syms_tuple = Tuple(syms(vi))
    for s in syms_tuple
        # Find all indices in `vns` with symbol `s`
        inds = findall(vn -> vn.sym == s, vi.metadata.vns)
        n = length(inds)
        # New `vns`
        sym_vns = getindex.((vi.metadata.vns,), inds)
        # New idcs
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        # New dists
        sym_dists = getindex.((vi.metadata.dists,), inds)
        # New gids, can make a resizeable FillArray
        sym_gids = getindex.((vi.metadata.gids,), inds)
        @assert length(sym_gids) <= 1 ||
            all(x -> x == sym_gids[1], @view sym_gids[2:end])
        # New orders
        sym_orders = getindex.((vi.metadata.orders,), inds)
        # New flags
        sym_flags = Dict(a => vi.metadata.flags[a][inds] for a in keys(vi.metadata.flags))

        # Extract new ranges and vals
        _ranges = getindex.((vi.metadata.ranges,), inds)
        # `copy.()` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(vi.metadata.vals[_ranges[i]]) for i in 1:n]
        sym_ranges = Vector{eltype(_ranges)}(undef, n)
        start = 0
        for i in 1:n
            sym_ranges[i] = start + 1 : start + length(_vals[i])
            start += length(_vals[i])
        end
        sym_vals = foldl(vcat, _vals)

        push!(new_metas, Metadata(sym_idcs, sym_vns, sym_ranges, sym_vals,
                                    sym_dists, sym_gids, sym_orders, sym_flags)
            )
    end
    logp = vi.logp[]
    num_produce = vi.num_produce[]
    nt = NamedTuple{syms_tuple}(Tuple(new_metas))
    return VarInfo(nt, Ref(logp), Ref(num_produce))
end

#=
function getproperty(vi::VarInfo, f::Symbol)
    f === :logp && return getfield(vi, :logp)[]
    f === :num_produce && return getfield(vi, :num_produce)[]
    f === :metadata && return getfield(vi, :metadata)
    return getfield(getfield(vi, :metadata), f)
end
function setproperty!(vi::VarInfo, f::Symbol, x)
    f === :logp && return getfield(vi, :logp)[] = convert(typeof(vi.logp), x)
    f === :num_produce && return getfield(vi, :num_produce)[] = x
    return setfield!(vi, f, x)
end
=#

"""
`empty!(vi::VarInfo)`

Empties all the fields of `vi.metadata` and resets `vi.logp` and `vi.num_produce` to
zeros. This is useful when using a sampling algorithm that assumes an empty
`vi::VarInfo`, e.g. `SMC`.
"""
function empty!(vi::VarInfo)
    _empty!(vi.metadata)
    vi.logp[] = 0
    vi.num_produce[] = 0
    return vi
end
@inline _empty!(metadata::Metadata) = empty!(metadata)
@inline function _empty!(metadata::NamedTuple{names}) where {names}
    # Check if the named tuple is empty and end the recursion
    length(names) === 0 && return nothing
    # Take the first key in the NamedTuple
    f = names[1]
    # Empty the first instance of `Metadata`
    empty!(getfield(metadata, f))
    # Recurse using the remaining pairs of the NamedTuple
    return _empty!(_tail(metadata))
end

# Functions defined only for UntypedVarInfo
"""
`keys(vi::UntypedVarInfo)`

Returns an iterator over `vi.vns`.
"""
keys(vi::UntypedVarInfo) = keys(vi.metadata.idcs)

"""
`setgid!(vi::VarInfo, gid::Selector, vn::VarName)`

Adds `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
setgid!(vi::UntypedVarInfo, gid::Selector, vn::VarName) = push!(vi.metadata.gids[getidx(vi, vn)], gid)
function setgid!(vi::TypedVarInfo, gid::Selector, vn::VarName{sym}) where sym
    push!(getfield(vi.metadata, sym).gids[getidx(vi, vn)], gid)
end

"""
`istrans(vi::VarInfo, vn::VarName)`

Returns true if `vn`'s values in `vi` are transformed to Eucledian space, and false if
they are in the support of `vn`'s distribution.
"""
istrans(vi::AbstractVarInfo, vn::VarName) = is_flagged(vi, vn, "trans")

"""
`getlogp(vi::VarInfo)`

Returns the log of the joint probability of the observed data and parameters sampled in
`vi`.
"""
getlogp(vi::AbstractVarInfo) = vi.logp[]

"""
`setlogp!(vi::VarInfo, logp::Real)`

Sets the log of the joint probability of the observed data and parameters sampled in
`vi` to `logp`.
"""
@inline setlogp!(vi::AbstractVarInfo, logp::Real) = getfield(vi, :logp)[] = logp

"""
`acclogp!(vi::VarInfo, logp::Real)`

Adds `logp` to the value of the log of the joint probability of the observed data and
parameters sampled in `vi`.
"""
acclogp!(vi::AbstractVarInfo, logp::Real) = getfield(vi, :logp)[] += logp

"""
`resetlogp!(vi::VarInfo)`

Resets the value of the log of the joint probability of the observed data and parameters
sampled in `vi` to 0.
"""
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, 0.0)

"""
`isempty(vi::VarInfo)`

Returns true if `vi` is empty and false otherwise.
"""
isempty(vi::UntypedVarInfo) = isempty(vi.metadata.idcs)
isempty(vi::TypedVarInfo) = _isempty(vi.metadata)
@inline function _isempty(metadata::NamedTuple{names}) where {names}
    # Checks if `metadata` is empty to end the recursion
    length(names) === 0 && return true
    # Take the first key of `metadata`
    f = names[1]
    # If not empty, return false and end the recursion. Otherwise, recurse using the remaining of `metadata`.
    return isempty(getfield(metadata, f).idcs) && _isempty(_tail(metadata))
end

# X -> R for all variables associated with given sampler
"""
`link!(vi::VarInfo, spl::Sampler)`

Transforms the values of the random variables sampled by `spl` in `vi` from the support
of their distributions to the Eucledian space and sets their corresponding ``"trans"`
flag values to `true`.
"""
function link!(vi::UntypedVarInfo, spl::Sampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = _getvns(vi, spl)
    if ~istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            # TODO: Use inplace versions to avoid allocations
            setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, true, vn)
        end
    else
        @warn("[Turing] attempt to link a linked vi")
    end
end
function link!(vi::TypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    space = getspace(spl)
    return _link!(vi.metadata, vi, vns, space)
end
@inline function _link!(metadata::NamedTuple{names}, vi, vns, space) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first key/symbol of `metadata`
    f = names[1]
    # Extract the list of `vns` with symbol `f`
    f_vns = getfield(vns, f)
    # Transform only if `f` is in the space of the sampler or the space is void
    if f ∈ space || length(space) == 0
        if ~istrans(vi, f_vns[1])
            # Iterate over all `f_vns` and transform
            for vn in f_vns
                dist = getdist(vi, vn)
                setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
                settrans!(vi, true, vn)
            end
        else
            @warn("[Turing] attempt to link a linked vi")
        end
    end
    # Recurse using the remaining of `metadata`
    return _link!(_tail(metadata), vi, vns, space)
end

# R -> X for all variables associated with given sampler
"""
`invlink!(vi::VarInfo, spl::Sampler)`

Transforms the values of the random variables sampled by `spl` in `vi` from the
Eucledian space back to the support of their distributions and sets their corresponding
``"trans"` flag values to `false`.
"""
function invlink!(vi::UntypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    if istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, false, vn)
        end
    else
        @warn("[Turing] attempt to invlink an invlinked vi")
    end
end
function invlink!(vi::TypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    space = getspace(spl)
    return _invlink!(vi.metadata, vi, vns, space)
end
@inline function _invlink!(metadata::NamedTuple{names}, vi, vns, space) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first key/symbol of `metadata`
    f = names[1]
    # Extract the list of `vns` with symbol `f`
    f_vns = getfield(vns, f)
    # Transform only if `f` is in the space of the sampler or the space is void
    if f ∈ space || length(space) == 0
        if istrans(vi, f_vns[1])
            # Iterate over all `f_vns` and transform
            for vn in f_vns
                dist = getdist(vi, vn)
                setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
                settrans!(vi, false, vn)
            end
        else
            @warn("[Turing] attempt to invlink an invlinked vi")
        end
    end
    return _invlink!(_tail(metadata), vi, vns, space)
end

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
"""
`getindex(vi::VarInfo, vn::VarName)`
`getindex(vi::VarInfo, vns::Vector{<:VarName})`

Returns the current value(s) of `vn` (`vns`) in `vi` in the support of its (their)
distribution(s). If the value(s) is (are) transformed to the Eucledian space, it is
(they are) transformed back.
"""
function getindex(vi::AbstractVarInfo, vn::VarName)
    @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vn)
    return copy(istrans(vi, vn) ?
        invlink(dist, reconstruct(dist, getval(vi, vn))) :
        reconstruct(dist, getval(vi, vn)))
end
function getindex(vi::AbstractVarInfo, vns::Vector{<:VarName})
    @assert haskey(vi, vns[1]) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vns[1])
    return copy(istrans(vi, vns[1]) ?
        invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
        reconstruct(dist, getval(vi, vns), length(vns)))
end

"""
`getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})`

Returns the current value(s) of the random variables sampled by `spl` in `vi`. The
value(s) may or may not be transformed to Eucledian space.
"""
getindex(vi::AbstractVarInfo, spl::SampleFromPrior) = copy(getall(vi))
getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, _getranges(vi, spl)))
function getindex(vi::TypedVarInfo, spl::Sampler)
    # Gets the ranges as a NamedTuple
    ranges = _getranges(vi, spl)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return vcat(_getindex(vi.metadata, ranges)...)
end
# Recursively builds a tuple of the `vals` of all the symbols
@inline function _getindex(metadata::NamedTuple{names}, ranges) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return ()
    # Take the first key of `metadata`
    f = names[1]
    # Get the `vals` and `ranges` of symbol `f`
    f_vals = getfield(metadata, f).vals
    f_range = getfield(ranges, f)
    # Get the values from `f_vals` that were sampled by `spl` and recurse using the remaining of `metadata`
    return (f_vals[f_range], _getindex(_tail(metadata), ranges)...)
end

"""
`setindex!(vi::VarInfo, val, vn::VarName)`

Sets the current value(s) of the random variable `vn` in `vi` to `val`. The value(s) may
or may not be transformed to Eucledian space.
"""
setindex!(vi::AbstractVarInfo, val::Any, vn::VarName) = setval!(vi, val, vn)

"""
`setindex!(vi::VarInfo, val, spl::Union{SampleFromPrior, Sampler})`

Sets the current value(s) of the random variables sampled by `spl` in `vi` to `val`. The
value(s) may or may not be transformed to Eucledian space.
"""
setindex!(vi::AbstractVarInfo, val::Any, spl::SampleFromPrior) = setall!(vi, val)
setindex!(vi::UntypedVarInfo, val::Any, spl::Sampler) = setval!(vi, val, _getranges(vi, spl))
function setindex!(vi::TypedVarInfo, val, spl::Sampler)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = _getranges(vi, spl)
    _setindex!(vi.metadata, val, ranges)
    return val
end
# Recursively writes the entries of `val` to the `vals` fields of all the symbols as if they were a contiguous vector.
@inline function _setindex!(metadata::NamedTuple{names}, val, ranges, start = 0) where {names}
    length(names) === 0 && return nothing
    f = names[1]
    # The `vals` field of symbol `f`
    f_vals = getfield(metadata, f).vals
    # The indices in `f_vals` corresponding to sampler `spl`
    f_range = getfield(ranges, f)
    n = length(f_range)
    # Writes the portion of `val` corresponding to the symbol `f`
    @views f_vals[f_range] .= val[start+1:start+n]
    # Increment the global index and move to the next symbol
    start += n
    return _setindex!(_tail(metadata), val, ranges, start)
end

function Base.eltype(vi::AbstractVarInfo, spl::Union{AbstractSampler, SampleFromPrior})
    return eltype(Core.Compiler.return_type(getindex, Tuple{typeof(vi), typeof(spl)}))
end

"""
`haskey(vi::VarInfo, vn::VarName)`

Returns `true` if `vn` has been sampled in `vi` and `false` otherwise.
"""
haskey(vi::UntypedVarInfo, vn::VarName) = haskey(vi.metadata.idcs, vn)
function haskey(vi::TypedVarInfo, vn::VarName{sym}) where {sym}
    metadata = vi.metadata
    Tmeta = typeof(metadata)
    return sym in fieldnames(Tmeta) && haskey(getfield(metadata, sym).idcs, vn)
end

function show(io::IO, vi::UntypedVarInfo)
    vi_str = """
    /=======================================================================
    | VarInfo
    |-----------------------------------------------------------------------
    | Varnames  :   $(string(vi.metadata.vns))
    | Range     :   $(vi.metadata.ranges)
    | Vals      :   $(vi.metadata.vals)
    | GIDs      :   $(vi.metadata.gids)
    | Orders    :   $(vi.metadata.orders)
    | Logp      :   $(vi.logp[])
    | #produce  :   $(vi.num_produce[])
    | flags     :   $(vi.metadata.flags)
    \\=======================================================================
    """
    print(io, vi_str)
end

# Add a new entry to VarInfo
"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution)`

Pushes a new random variable `vn` with a sampled value `r` from a distribution `dist` to
the `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution)
    return push!(vi, vn, r, dist, Set{Selector}([]))
end

"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler)`

Pushes a new random variable `vn` with a sampled value `r` sampled with a sampler `spl`
from a distribution `dist` to `VarInfo` `vi`. The sampler is passed here to invalidate
its cache where defined.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, spl::Sampler)
    spl.info[:cache_updated] = CACHERESET
    return push!(vi, vn, r, dist, spl.selector)
end
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, spl::AbstractSampler)
    return push!(vi, vn, r, dist)
end

"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution, gid::Selector)`

Pushes a new random variable `vn` with a sampled value `r` sampled with a sampler of
selector `gid` from a distribution `dist` to `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, gid::Selector)
    return push!(vi, vn, r, dist, Set([gid]))
end
function push!(vi::UntypedVarInfo, vn::VarName, r::Any, dist::Distribution, gidset::Set{Selector})
    @assert ~(vn in keys(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

    val = vectorize(dist, r)

    vi.metadata.idcs[vn] = length(vi.metadata.idcs) + 1
    push!(vi.metadata.vns, vn)
    l = length(vi.metadata.vals); n = length(val)
    push!(vi.metadata.ranges, l+1:l+n)
    append!(vi.metadata.vals, val)
    push!(vi.metadata.dists, dist)
    push!(vi.metadata.gids, gidset)
    push!(vi.metadata.orders, vi.num_produce[])
    push!(vi.metadata.flags["del"], false)
    push!(vi.metadata.flags["trans"], false)

    return vi
end
function push!(
            vi::TypedVarInfo,
            vn::VarName{sym},
            r::Any,
            dist::Distributions.Distribution,
            gidset::Set{Selector}
            ) where sym

    @assert ~(haskey(vi, vn)) "[push!] attempt to add an exisitng variable $(vn.sym) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gidset"

    val = vectorize(dist, r)

    meta = getfield(vi.metadata, sym)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals); n = length(val)
    push!(meta.ranges, l+1:l+n)
    append!(meta.vals, val)
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, vi.num_produce[])
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)

    return meta
end

"""
`setorder!(vi::VarInfo, vn::VarName, index::Int)`

Sets the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe
statements run before sampling `vn`.
"""
function setorder!(vi::UntypedVarInfo, vn::VarName, index::Int)
    if vi.metadata.orders[vi.metadata.idcs[vn]] != index
        vi.metadata.orders[vi.metadata.idcs[vn]] = index
    end
    return vi
end
function setorder!(mvi::TypedVarInfo, vn::VarName{sym}, index::Int) where {sym}
    vi = getfield(mvi.metadata, sym)
    if vi.metadata.orders[vi.metadata.idcs[vn]] != index
        vi.metadata.orders[vi.metadata.idcs[vn]] = index
    end
    return mvi
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

"""
`is_flagged(vi::VarInfo, vn::VarName, flag::String)`

Returns `true` if `vn` has a true value for `flag` in `vi`, and `false` otherwise.
"""
function is_flagged(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.metadata.flags[flag][getidx(vi, vn)]
end
function is_flagged(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)]
end

"""
`unset_flag!(vi::VarInfo, vn::VarName, flag::String)`

Sets `vn`'s value for `flag` to `false` in `vi`.
"""
function unset_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.metadata.flags[flag][getidx(vi, vn)] = false
end
function unset_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)] = false
end

"""
`set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)`

Sets the `"del"` flag of variables in `vi` with `order > vi.num_produce` to `true`.
"""
function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = _getidcs(vi, spl)
    if vi.num_produce[] == 0
        for i = length(gidcs):-1:1
          vi.metadata.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.metadata.orders)
            if i in gidcs && vi.metadata.orders[i] > vi.num_produce[]
                vi.metadata.flags["del"][i] = true
            end
        end
    end
    return nothing
end
function set_retained_vns_del_by_spl!(vi::TypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = _getidcs(vi, spl)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, vi.num_produce[])
end
@inline function _set_retained_vns_del_by_spl!(metadata::NamedTuple{names}, gidcs, num_produce) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first symbol
    f = names[1]
    # Get the idcs, orders and flags with symbol `f`
    f_gidcs = getfield(gidcs, f)
    f_orders = getfield(metadata, f).orders
    f_flags = getfield(metadata, f).flags
    # Set the flag for variables with symbol `f`
    if num_produce == 0
        for i = length(f_gidcs):-1:1
            f_flags["del"][f_gidcs[i]] = true
        end
    else
        for i in 1:length(f_orders)
            if i in f_gidcs && f_orders[i] > num_produce
                f_flags["del"][i] = true
            end
        end
    end
    # Recurse using the remaining of `metadata`
    return _set_retained_vns_del_by_spl!(_tail(metadata), gidcs, num_produce)
end

"""
`updategid!(vi::VarInfo, vn::VarName, spl::Sampler)`

If `vn` doesn't have a sampler selector linked and `vn`'s symbol is in the space of
`spl`, this function will set `vn`'s `gid` to `Set([spl.selector])`.
"""
function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler)
    if ~isempty(getspace(spl.alg)) && isempty(getgid(vi, vn)) && getsym(vn) in getspace(spl.alg)
        setgid!(vi, spl.selector, vn)
    end
end

end # end of module
