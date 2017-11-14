abstract SizeBiasedDistribution   <: ContinuousUnivariateDistribution
abstract TotalMassDistribution    <: ContinuousUnivariateDistribution

Distributions.minimum(d::SizeBiasedDistribution) = 0.0
Distributions.maximum(d::SizeBiasedDistribution) = d.T_surplus

Distributions.minimum(d::TotalMassDistribution) = 0.0
Distributions.maximum(d::TotalMassDistribution) = Inf

include("exptiltedsigma.jl")
include("pyp.jl")
include("ngip.jl")
