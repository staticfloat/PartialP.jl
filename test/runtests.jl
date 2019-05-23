using PartialP, Test
using PartialP: gradient

if PartialP.usetyped
  @info "Testing PartialP in type-hacks mode."
else
  @info "Testing PartialP in normal mode."
end

@testset "PartialP" begin

@testset "Features" begin
  include("features.jl")
end

@testset "Gradients" begin
  include("gradcheck.jl")
end

@testset "Complex" begin
  include("complex.jl")
end

@testset "Compiler" begin
  include("compiler.jl")
end

end
