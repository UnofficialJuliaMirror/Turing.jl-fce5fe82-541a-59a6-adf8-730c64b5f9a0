#####################
# Dirichlet Process #
#####################

immutable SBS_DP <: SizeBiasedSamplingDistribution
    alpha         ::  Real
    T_surplus     ::  Real
end

Distributions.rand(d::SBS_DP) = d.T_surplus*rand(Beta(1, d.alpha))
Distributions.logpdf{T<:Real}(d::SBS_DP, x::T) = logpdf(Beta(1, d.alpha),x/d.T_surplus)

######################
# Pitman-Yor Process #
######################

immutable SBS_PYP <: SizeBiasedSamplingDistribution
    alpha         ::  Real
    theta         ::  Real
    index         ::  Int
    T_surplus     ::  Real
end

Distributions.rand(d::SBS_PYP) = d.T_surplus*rand(Beta(1 - d.alpha, d.theta + d.index * d.alpha))
Distributions.logpdf{T<:Real}(d::SBS_PYP, x::T) = logpdf(Beta(1 - d.alpha, d.theta + d.index * d.alpha),x/d.T_surplus)
