# module Traces

mutable struct Trace{Tspl <: AbstractSampler, Tvi <: AbstractVarInfo, Tmodel <: Model}
    task  ::  Task
    vi    ::  Tvi
    spl   ::  Tspl
    model ::  Tmodel
    Trace{Tspl, Tvi, Tmodel}() where {Tspl, Tvi, Tmodel} = new()
    function Trace{SampleFromPrior}(m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
        res = new{SampleFromPrior, typeof(vi), typeof(m)}()
        res.vi = vi
        res.model = m
        res.spl = SampleFromPrior()
        return res
    end
    function Trace{T}(m::Model, spl::AbstractSampler, vi::AbstractVarInfo) where T <: Sampler
        res = new{T, typeof(vi), typeof(m)}()
        res.model = m
        res.spl = spl
        res.vi = vi
        return res
    end
end
function Base.copy(trace::Trace)
    newtrace = typeof(trace)()
    newtrace.vi = deepcopy(trace.vi)
    newtrace.task = Base.copy(trace.task)
    newtrace.spl = trace.spl
    newtrace.model = trace.model
    return newtrace
end

# NOTE: this function is called by `forkr`
function Trace(
    f::Function,
    m::Model,
    spl::T,
    vi::AbstractVarInfo
) where {T <: AbstractSampler}
    res = Trace{T}(m, spl, deepcopy(vi));
    # CTask(()->f());
    res.task = CTask( () -> begin res=f(); produce(Val{:done}); res; end )
    if isa(res.task.storage, Nothing)
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end
function Trace(
    m::Model,
    spl::T,
    vi::AbstractVarInfo
) where {T <: AbstractSampler}
    res = Trace{T}(m, spl, deepcopy(vi));
    # CTask(()->f());
    res.vi.num_produce = 0
    res.task = CTask( () -> begin vi_new=m(vi, spl); produce(Val{:done}); vi_new; end )
    if isa(res.task.storage, Nothing)
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end

# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (t.vi.num_produce += 1; consume(t.task))

# Task copying version of fork for Trace.
function fork(trace :: Trace, is_ref :: Bool = false)
    newtrace = copy(trace)
    is_ref && set_retained_vns_del_by_spl!(newtrace.vi, newtrace.spl)
    newtrace.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace :: Trace)
    newtrace = Trace(trace.task.code, trace.model, trace.spl, deepcopy(trace.vi))
    newtrace.spl = trace.spl
    newtrace.vi.num_produce = 0
    return newtrace
end

current_trace() = current_task().storage[:turing_trace]

# end # end module
