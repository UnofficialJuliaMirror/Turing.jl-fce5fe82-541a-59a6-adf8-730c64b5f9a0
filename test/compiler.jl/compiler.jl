using Test
using Turing: make_varname

@testset "compiler.jl" begin

    # Test make_varname.
    @test make_varname(:x)[1] == :x
    @test make_varname(:x)[2] == ()

    @test make_varname(:(x[1]))[1] == :x
    @test make_varname(:(y[n]))[2] == :(([n],))

    @test make_varname(:(y[n][m]))[1] == :y
    @test make_varname(:(z[n][m]))[2] == :(([n], [m]))

    @test make_varname(:(z[n, m]))[1] == :z
    @test make_varname(:(z[n, m]))[2] == :(([n, m],))
end
