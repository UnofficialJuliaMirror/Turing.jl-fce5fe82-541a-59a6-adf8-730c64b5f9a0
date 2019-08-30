using Turing, Random, Test
using StatsFuns

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "smc.jl" begin
  @model normal() = begin
    a ~ Normal(4,5)
    3 ~ Normal(a,2)
    b ~ Normal(a,1)
    1.5 ~ Normal(b,2)
    a, b
  end

  tested = sample(normal(), SMC(10))
end
