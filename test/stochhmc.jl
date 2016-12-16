using Distributions
using Turing
using Base.Test

obs = [0,1,0,1,1,1,1,1,1,1]

@model stoch_test begin
  @assume p ~ Beta(2,2)
  for i = 1:length(obs)
    @observe obs[i] ~ Bernoulli(p)
  end
  @predict p
end

# chain = sample(stoch_test, HMC(1000, 1.0, 3))
chain = sample(stoch_test, StochHMC(1000, 1.0, 3, 0.7, 0.0))
println("[stochhmc] mean(chain[:p]):", mean(chain[:p]), ", analytical:", 10/14)
@test_approx_eq_eps mean(chain[:p]) 10/14 0.05
