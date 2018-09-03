"""
    gradient(vi::VarInfo, model::Function, spl::Union{Nothing, Sampler})

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint probibilioty. This function uses chunk-wise forward AD with a chunk of size $(length(Turing.FADCfg.seeds)) as default.

Example:

```julia
grad = gradient(vi, model, spl)
end
```
"""
gradient(vi::VarInfo, model::Function) = gradient_f(realpart(vi[nothing]), vi, model, nothing)
gradient(vi::VarInfo, model::Function, spl::Union{Nothing, Sampler}) = gradient_f(realpart(vi[spl]), vi, model, spl)

gradient_f(θ::Vector{Float64}, vi::VarInfo, model::Function) = gradient_f(vi, model, nothing)
gradient_f(θ::Vector{Float64}, _vi::VarInfo, model::Function, spl::Union{Nothing, Sampler}) = begin

  vi      = deepcopy(_vi)
  vi[spl] = θ

  f(x::Vector) = begin
    vi[spl] = x
    -getlogp(runmodel(model, vi, spl))
  end

  if length(θ) != size(FADCfg.duals)[1]
      setchunksize(length(Turing.FADCfg.seeds), θ);
  end

  g = x -> ForwardDiff.gradient(f, x, FADCfg)

  grad = g(vi[spl])
end

gradient_r(theta::Vector{Float64}, vi::VarInfo, model::Function) = gradient_r(theta, vi, model, nothing)
gradient_r(theta::Vector{Float64}, vi::Turing.VarInfo, model::Function, spl::Union{Nothing, Sampler}) = begin
    f_r(ipts) = begin
      vi[spl] = ipts
      -runmodel(model, vi, spl).logp
    end

    grad = Tracker.gradient(f_r, theta)
    vi.logp = vi.logp.data
    vi_spl = vi[spl]
    for i = 1:length(theta)
      vi_spl[i] = vi_spl[i].data
    end

    first(grad).data
end


"""
   Function for checking numerical errors in gradients
"""
verifygrad(grad::Vector{Float64}) = begin
  if any(isnan.(grad)) || any(isinf.(grad))
    @warn("Numerical error has been found in gradients.")
    @warn("grad = $(grad)")
    false
  else
    true
  end
end
