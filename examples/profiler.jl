# Initialize environment in current directory
@info("Ensuring example environment instantiated...")
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

@info("Loading PartialP...")
using PartialP

function f(x)
  for i = 1:5
    x = sin(cos(x))
  end
  return x
end

function loop(x, n)
  r = x/x
  for i = 1:n
    r *= f(x)
  end
  return sin(cos(r))
end

gradient(loop, 2, 3)

PartialP.@profile loop(2, 3)

function logsumexp(x::Array{Float64,1})
  A = maximum(x);
  ema = exp.(x .- A);
  sema = sum(ema);
  log(sema) + A;
end

gradient(logsumexp, rand(100))

PartialP.@profile logsumexp(rand(100))
