#######################################
# Normalised Inverse Gaussian Process #
#######################################

immutable T_NIGP <: TotalMassDistribution
    tau           ::  Real
end

Distributions.rand(d::T_NIGP) = rand(ExpTiltedSigma(0.5, d.tau))
Distributions.logpdf{T<:Real}(d::T_NIGP, x::T) = 0

immutable SBS_NIGP <: SizeBiasedDistribution
    T_surplus     ::  Real
end

Distributions.rand(d::SBS_NIGP) = begin
  X = sqrt(rand(Gamma(3/4, 1)))
  Y = sqrt(rand(InverseGamma(1/4, 1/(64 * d.T_surplus^2))))
  d.T_surplus * X / (X + Y)
end

# NOTE: Yet derived (compute CDF, then take the gradient)
Distributions.logpdf{T<:Real}(d::SBS_NIGP, x::T) = 0
