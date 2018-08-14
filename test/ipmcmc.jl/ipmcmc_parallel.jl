addprocs(1)

@everywhere using Turing
using Base.Test

srand(125)

@everywhere x = [1.5 2.0]

@everywhere @model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

# run particle MCMC sampler in parallel
@everywhere inference = IPMCMC(30, 500, 4)
@everywhere mf = gdemo(x)
chain = sample(mf, inference)

println(mean(chain[:s]))
println(mean(chain[:m]))

@test mean(chain[:s]) ≈ 49/24 atol=0.1
@test mean(chain[:m]) ≈ 7/6 atol=0.1
