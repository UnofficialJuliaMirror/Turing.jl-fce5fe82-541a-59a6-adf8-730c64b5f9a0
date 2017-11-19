abstract type StickSizeBiasedDistribution <: ContinuousUnivariateDistribution end
abstract type SizeBiasedDistribution      <: ContinuousUnivariateDistribution end
abstract type TotalMassDistribution       <: ContinuousUnivariateDistribution end

Distributions.minimum(d::StickSizeBiasedDistribution) = 0.0
Distributions.maximum(d::StickSizeBiasedDistribution) = 1.0
init(dist::StickSizeBiasedDistribution) = rand(dist)

Distributions.minimum(d::SizeBiasedDistribution) = 0.0
Distributions.maximum(d::SizeBiasedDistribution) = d.T_surplus

Distributions.minimum(d::TotalMassDistribution) = 0.0
Distributions.maximum(d::TotalMassDistribution) = Inf

init(dist::SizeBiasedDistribution) = rand(dist)

include("exptiltedsigma.jl")
include("pyp.jl")
include("nigp.jl")
