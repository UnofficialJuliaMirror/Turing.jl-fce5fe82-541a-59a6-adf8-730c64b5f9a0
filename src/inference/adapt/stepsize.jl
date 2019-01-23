######################
### Mutable states ###
######################

mutable struct DAState{TI<:Integer,TF<:Real}
    m     :: TI
    ϵ     :: TF
    μ     :: TF
    x_bar :: TF
    H_bar :: TF
end

function DAState(ϵ::Real)
    μ = computeμ(ϵ) # NOTE: this inital values doesn't affect anything as they will be overwritten
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function computeμ(ϵ::Real)
    return log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function reset!(dastate::DAState{TI,TF}) where {TI<:Integer,TF<:Real}
    dastate.m = zero(TI)
    dastate.x_bar = zero(TF)
    dastate.H_bar = zero(TF)
end

mutable struct MSSState{T<:Real}
    ϵ :: T
end

################
### Adapters ###
################

abstract type StepSizeAdapter <: AbstractAdapter end

struct FixedStepSize{T<:Real} <: StepSizeAdapter
    ϵ :: T
end

function getss(fss::FixedStepSize)
    return fss.ϵ
end

struct DualAveraging{TI<:Integer,TF<:Real} <: StepSizeAdapter
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
end

function DualAveraging(spl::Sampler{<:AdaptiveHamiltonian}, ::Nothing, ϵ::Real)
    return DualAveraging(0.05, 10.0, 0.75, spl.alg.delta, DAState(ϵ))
end

function getss(da::DualAveraging)
    return da.state.ϵ
end

struct ManualSSAdapter{T<:Real} <:StepSizeAdapter
    state :: MSSState{T}
end

function getss(mssa::ManualSSAdapter)
    return mssa.state.ϵ
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_stepsize!(da::DualAveraging, stats::Real)
    @debug "adapting step size ϵ..."
    @debug "current α = $(stats)"
    da.state.m += 1
    m = da.state.m

    # Clip average MH acceptance probability.
    stats = stats > 1 ? 1 : stats

    γ = da.γ; t_0 = da.t_0; κ = da.κ; δ = da.δ
    μ = da.state.μ; x_bar = da.state.x_bar; H_bar = da.state.H_bar

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - stats)

    x = μ - H_bar * sqrt(m) / γ            # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    @debug "new ϵ = $(ϵ), old ϵ = $(da.state.ϵ)"

    if isnan(ϵ) || isinf(ϵ)
        @warn "Numerical error has been found: ϵ = $ϵ, rejectting..."
        @warn "Previous valid ϵ = $(da.state.ϵ) will be used instead."
    else
        da.state.ϵ = ϵ
    end
    da.state.x_bar = x_bar
    da.state.H_bar = H_bar
end

function adapt!(da::DualAveraging, stats::Real, is_updateμ::Bool)
    adapt_stepsize!(da, stats)
    if is_updateμ
        da.state.μ = computeμ(da.state.ϵ)
        reset!(da.state)
    end
end


# TODO: remove used Turing-wrapper functions

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/base_hmc.hpp
function find_good_eps(model, spl::Sampler{T}, vi::VarInfo) where T
    # Negative potential energy func, Hamiltonian energy func
    Uf, Hf = gen_lj_func(vi, spl, model), gen_H_func()
    momentum_sampler = gen_momentum_sampler(vi, spl)

    @info "[Turing] looking for good initial eps..."
    ϵ = 1.0

    θ, p = vi[spl], momentum_sampler()
    H0 = Hf(θ, p, Uf(θ))

    θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
    h = τ == 0 ? Inf : Hf(θ_prime, p_prime, Uf(θ_prime))

    delta_H = H0 - h
    direction = delta_H > log(0.8) ? 1 : -1

    iter_num = 1

    # Heuristically find optimal ϵ
    while (iter_num <= 12)

        p = momentum_sampler()
        H0 = Hf(vi[spl], p, Uf(vi[spl]))

        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        τ == 0 &&
            @info "\r[$T] Numerical error occured in initial step size serarch."
        h = τ == 0 ? Inf : Hf(θ_prime, p_prime, Uf(θ_prime))
        @debug "direction = $direction, h = $h"

        delta_H = H0 - h

        if ((direction == 1) && !(delta_H > log(0.8)))
            break
        elseif ((direction == -1) && !(delta_H < log(0.8)))
            break
        else
            ϵ = direction == 1 ? 2.0 * ϵ : 0.5 * ϵ
        end

        iter_num += 1
    end

    while h == Inf  # revert if the last change is too big
        ϵ = ϵ / 2               # safe is more important than large
        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        h = τ == 0 ? Inf : Hf(θ_prime, p_prime, Uf(θ_prime))
    end
    h == Inf && @info "\r[$T] Numerical error occured in initial step size serarch."
    @info "\r[$T] found initial ϵ: $ϵ"

    return ϵ
end
