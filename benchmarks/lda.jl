# Simulate data
using Distributions

V = 5   # vocab
K = 3   # topic
M = 25  # doc

alpha = collect(ones(K) / K);
beta = collect(ones(V) / V);

theta = rand(Dirichlet(alpha), M)
phi = rand(Dirichlet(beta), K)
@info "true vals" phi theta sum(phi * theta, dims=1)

avg_doc_length = 50

doc_length = rand(Poisson(avg_doc_length), M)

N = sum(doc_length)

w = Vector{Int}(undef, N)
doc = Vector{Int}(undef, N)

n = 1
for m = 1:M, i=1:doc_length[m]
    global n
    z = rand(Categorical(theta[:,m]))
    w[n] = rand(Categorical(phi[:,z]))
    doc[n] = m
    n = n + 1
end

# LDA
using Turing

struct CategoricalReal<: Distributions.DiscreteUnivariateDistribution
    p
end

function Distributions.logpdf(d::CategoricalReal, k::Int64)
    return log(d.p[k])
end

@model lda_collapsed(K, V, M, N, w, doc, beta, alpha, ::Type{T}=Float64) where {T<:Real} = begin
    theta = Matrix{T}(undef, K, M)
    for m = 1:M
        theta[:,m] ~ Dirichlet(alpha)
    end

    phi = Matrix{T}(undef, V, K)
    for k = 1:K
        phi[:,k] ~ Dirichlet(beta)
    end

    word_dist = phi * theta

    for n = 1:N
        w[n] ~ CategoricalReal(word_dist[:,doc[n]])
    end
end

mf = lda_collapsed(K, V, M, N, w, doc, beta, alpha)

# c3= sample(mf, NUTS(2000, 0.65));
# c3= sample(mf, HMCDA(2000, 0.65, 1.0));
c3 = sample(mf, HMC(2000, 0.01, 1));

# julia> c3= sample(mf, HMC(2000, 0.01, 10))
# ┌ Info: Finished 2000 sampling steps in 574.492034088 (s)
# │   typeof(h.metric) = AdvancedHMC.Adaptation.UnitEuclideanMetric{Float64}
# │   typeof(τ) = AdvancedHMC.StaticTrajectory{AdvancedHMC.Leapfrog{Float64}}
# │   EBFMI(Hs) = 125.1832457754953
# └   mean(αs) = 0.9988664272956684
