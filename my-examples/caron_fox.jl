using Turing
using Turing: T_NIGP, SBS_NIGP
using Plots
using Gallium
#srand(240013)

@model caronFoxGGP(ct, sparse_graph, D_alpha_star, n_nodes, perm) = begin

  alpha ~ Exponential(1)
  W_alpha_star ~ T_NIGP(alpha)

  W_rem = W_alpha_star
  J = zeros(Float64, n_nodes);
  for i=1:n_nodes
    J[i] ~ SBS_NIGP(W_rem)
    W_rem -= J[i]
  end

  unobserved_mass = W_alpha_star^2

  for (i, j, ct) in sparse_graph
    correspondent_mass = J[perm(i)]*J[perm(j)]
    ct ~ Poisson(correspondent_mass)
    unobserved_mass -= correspondent_mass
  end

  ct = 0
  ct ~ Poisson(unobserved_mass)

end

graph = [(1, 1, 2), (1, 2, 1), (1, 3, 1), (2, 1, 1)]
n_nodes = 3
D_alpha_star = 5

function perm(x)
  if x == 1
    return 3
  elseif x == 3
    return 1
  end
  return x
end

sampler = SMC(100000)
mdl = caronFoxGGP(0, graph, D_alpha_star, n_nodes, perm)
samples = sample(mdl, sampler)
println("J[1] = ", mean([sample[Symbol("J[1]")]/sample[:W_alpha_star] for sample in samples]))
println("J[2] = ", mean([sample[Symbol("J[2]")]/sample[:W_alpha_star] for sample in samples]))
println("J[3] = ", mean([sample[Symbol("J[3]")]/sample[:W_alpha_star] for sample in samples]))
