using Turing
using Turing: SBS_DP, SBS_PYP, T_NIGP, SBS_NIGP, V_SBS_DP
using Turing: realpart, CHUNKSIZE
srand(100)

data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
mu_0 = mean(data); sigma_0 = 1/sqrt(0.635); sigma_1 = sigma_0/15

@model infiniteMixture(y) = begin
  N = length(y)
  H = Normal(mu_0, sigma_0)

  x = tzeros(Real, N);
  z = tzeros(Real, N)
  V = tzeros(Real, N);

  k = 0
  T = 1
  T_surplus = tzeros(Real, N+1);
  T_surplus[1] = T

  for i in 1:N
    ps = Array{Real}(vcat(T_surplus[1:k].*V[1:k]/T, T_surplus[k+1]/T))
    z[i] ~ Categorical(ps)
    if z[i] > k
      k = k + 1
      V[k] ~ V_SBS_DP(10)
      x[k] ~ H
      T_surplus[k+1] = T_surplus[k] - V[k]*T_surplus[k]
    end
    y[i] ~ Normal(x[z[i]], sigma_1)
  end
end

permutation = randperm(length(data))
model = infiniteMixture(data[permutation])
# vi = model()
sampler = HMC(100, 0.03, 5,:V,:x)
results = sample(infiniteMixture(data[permutation]), sampler)
