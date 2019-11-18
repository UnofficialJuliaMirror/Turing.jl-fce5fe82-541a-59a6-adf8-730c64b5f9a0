using Turing, Random, Test
using Turing: @varname
using Distributions: Normal
using StatsFuns

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "util.jl" begin
    i = 1
    vn = @varname s
    @test vn.sym == :s
    @test vn.indexing == ""

    vn = @varname x[1,2][1+5][45][3][i]
    @test vn.sym == :x
    @test vn.indexing == "[1,2][6][45][3][1]"
    @test StatsFuns.logistic(1.1) == 1.0 / (exp(-1.1) + 1.0)
    @test StatsFuns.logit(0.3) ≈ -0.8472978603872036 atol=1e-9
    @test !isnan(StatsFuns.logit(1.0))
    @test isinf(StatsFuns.logit(1.0))
    @test !isnan(StatsFuns.logit(0.0))
    @test isinf(StatsFuns.logit(0.0))

    @test Turing.Core.data(Dict(:y => 1, :x => 2, :z => 1), [:x, :y]) ==
        (x = 2, y = 1)
end
