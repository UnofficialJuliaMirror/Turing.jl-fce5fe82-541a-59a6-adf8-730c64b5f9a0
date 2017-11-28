using Turing
using Turing: V_SBS_PYP, V_SBS_DP
using Base.Test

### Mixture model with PY prior and shared variance
# Compute posterior distribution over partitions

# Model parameters
data = [-2,2,-1.5,1.5]
mu_0 = mean(data); sigma_0 = 4; sigma_1 = 0.5
tau0 = 1/sigma_0^2
tau1 = 1/sigma_1^2

# PYP parameters
sigma = 0.25 # = alpha
theta = 0.1

# Model definition
@model py_mixture_model(y) = begin
  N = length(y)
  H = Normal(mu_0, sigma_0)

  x = tzeros(Real, N); V = tzeros(Real, N); z = tzeros(Int, N)
  k = 0
  T = 1
  T_surplus = tzeros(Real, N+1)
  T_surplus[1] = T

  for i in 1:N
    ps = Array{Real}(vcat(T_surplus[1:k].*V[1:k]/T, T_surplus[k+1]/T))
    z[i] ~ Categorical(ps)
    if z[i] > k
      k = k + 1
      V[k] ~ V_SBS_PYP(sigma, theta, k)
      x[k] ~ H
      T_surplus[k+1] = T_surplus[k] - V[k]*T_surplus[k]
    end
    y[i] ~ Normal(x[z[i]], sigma_1)
  end
end

### Compute empirical posterior distribution over partitions
sampler = SMC(1000)
samples = sample(py_mixture_model(data), sampler) # not Chain type but samples

# Check that there is no NaN value associated
@test sum([sum(sample[:x][sample[:z]].== NaN)+sum(sample[:V][sample[:z]].== NaN) for sample in samples]) == 0

# using Combinatorics; Partitions = collect(partitions(1:length(data)))
Partitions = [[[1, 2, 3, 4]], [[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]], [[1, 2], [3], [4]], [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 3], [2], [4]], [[1, 4], [2, 3]], [[1], [2, 3, 4]], [[1], [2, 3], [4]], [[1, 4], [2], [3]], [[1], [2, 4], [3]], [[1], [2], [3, 4]], [[1], [2], [3], [4]]]
empirical_probs = zeros(length(Partitions))

sum_weights = sum([sample.weight for sample in samples])
for sample in samples
  cluster = [find(sample[:z] .== c) for c in 1:maximum(sample[:z])]
  partition_idx = [i for i in 1:length(Partitions) if Partitions[i] == cluster][1]
  if sum_weights == 0
    empirical_probs[partition_idx] += 1
  else
    empirical_probs[partition_idx] += sample.weight
  end
end
if sum_weights == 0 empirical_probs /= length(samples) end

### Compute theoretical posterior distribution over partitions
# cf Maria Lomeli's thesis, Section 2.7.4, page 46.

function compute_log_joint(observations, partition, tau0, tau1)
  n = length(observations)
  k = length(partition)
  prob = k*log(sigma) + lgamma(theta) + lgamma(theta/sigma + k) - lgamma(theta/sigma) - lgamma(theta + n)
  for cluster in partition
    prob += lgamma(length(cluster) - sigma) - lgamma(1 - sigma)
    prob += compute_log_conditonal_observations(observations, cluster, tau0, tau1)
  end
  prob
end

function compute_log_conditonal_observations(observations, cluster, tau0, tau1)
  nl = length(cluster)
  prob = (nl/2)*log(tau1) - (nl/2)*log(2*pi) + 0.5*log(tau0) + 0.5*log(tau0+nl)
  prob += -tau1/2*(sum(observations)) + 0.5*(tau0*mu_0+tau1*sum(observations[cluster]))^2/(tau0+nl*tau1)
  prob
end

true_log_probs = [compute_log_joint(data, partition, tau0, tau1) for partition in Partitions]
true_probs = exp.(true_log_probs)
true_probs /= sum(true_probs)

######################### Test of similarity between distributions
# score = length(samples)*sum((empirical_probs - true_probs).^2 ./true_probs) # chi-square score
score = sum((empirical_probs - true_probs).^2)
println("L2 norm score: ", score)
@test score < 0.05
