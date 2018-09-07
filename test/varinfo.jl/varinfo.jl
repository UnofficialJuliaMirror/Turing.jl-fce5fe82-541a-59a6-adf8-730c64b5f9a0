using Turing, Test
using Turing: reconstruct, invlink, step
using Turing.VarReplay
using Turing.VarReplay: getvals, getidcs, set_retained_vns_del_by_spl!, is_flagged, unset_flag!

function randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Turing.Sampler, count::Bool)
  if ~haskey(vi, vn)
    r = rand(dist)
    Turing.push!(vi, vn, r, dist, spl.alg.gid)
    r
  elseif is_flagged(vi, vn, "del")
    unset_flag!(vi, vn, "del")
    r = rand(dist)
    Turing.setval!(vi, Turing.vectorize(dist, r), vn)
    r
  else
    if count Turing.checkindex(vn, vi, spl) end
    Turing.updategid!(vi, vn, spl)
    vi[vn]
  end
end

# generate unique model id
modelid = gensym(:model)

# generate unique compiletime variable id
varid = gensym()

# construct VarName object
vn1 = VarName(modelid, varid, :x, ([1],), 1)

@test vn1.modelid == modelid
@test vn1.varid == varid
@test vn1.varname == :x
@test vn1.indexing == ([1],)
@test vn1.counter == 1

vn11 = VarName(modelid, varid, :x, ([1],), 1)
@test vn11 == vn1

vi = VarInfo()
dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]

alg = PG(PG(5,5),2)
spl2 = Turing.Sampler(alg)

# model unique id
modelid = gensym(:model)

vn_w = VarName(modelid, gensym(), :w, (), 1)
randr(vi, vn_w, dists[1], spl2, true)

vn_x = VarName(modelid, gensym(), :x, (), 1)
vn_y = VarName(modelid, gensym(), :y, (), 1)
vn_z = VarName(modelid, gensym(), :z, (), 1)
vns = [vn_x, vn_y, vn_z]

alg = PG(PG(5,5),1)
spl1 = Turing.Sampler(alg)
for i = 1:3
  r = randr(vi, vns[i], dists[i], spl1, false)
  val = vi[vns[i]]
  @test sum(val - r) <= 1e-9
end

# println(vi)

@test length(getvals(vi, spl1)) == 3
@test length(getvals(vi, spl2)) == 1


vn_u = VarName(modelid, gensym(), :u, (), 1)
randr(vi, vn_u, dists[1], spl2, true)

# println(vi)
vi.num_produce = 1
set_retained_vns_del_by_spl!(vi, spl2)

# println(vi)

vals_of_1 = collect(getvals(vi, spl1))
# println(vals_of_1)
filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_1)
@test length(vals_of_1) == 3

vals_of_2 = collect(getvals(vi, spl2))
# println(vals_of_2)
filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_2)
@test length(vals_of_2) == 1

@model gdemo() = begin
  x ~ InverseGamma(2,3)
  y ~ InverseGamma(2,3)
  z ~ InverseGamma(2,3)
  w ~ InverseGamma(2,3)
  u ~ InverseGamma(2,3)
end

# Test the update of group IDs
g_demo_f = gdemo()
g = Turing.Sampler(Gibbs(1000, PG(10, 2, :x, :y, :z), HMC(1, 0.4, 8, :w, :u)))

pg, hmc = g.info[:samplers]

vi = g_demo_f(Turing.VarInfo(), nothing)
vi = step(g_demo_f, pg, vi)
@test vi.gids == [1,1,1,0,0]

vi = g_demo_f(vi, hmc)
@test vi.gids == [1,1,1,2,2]
