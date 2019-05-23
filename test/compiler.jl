using PartialP, Test
using PartialP: forward, @adjoint

macro test_inferred(ex)
  :(let res = nothing
    @test begin
      res = @inferred $ex
      true
    end
    res
  end) |> esc
end

trace_contains(st, func, file, line) = any(st) do fr
  func in (nothing, fr.func) && endswith(String(fr.file), file) &&
    fr.line == line
end

bad(x) = x
@adjoint bad(x) = x, Δ -> error("bad")

PartialP.usetyped && PartialP.refresh()

function badly(x)
  x = x + 1
  x = bad(x)
  return x
end

y, back = forward(badly, 2)
@test y == 3
@test_throws Exception back(1)
bt = try back(1) catch e stacktrace(catch_backtrace()) end

@test trace_contains(bt, nothing, "compiler.jl", 20)
@test trace_contains(bt, :badly, "compiler.jl", 26)

# Type inference checks

PartialP.refresh()

y, back = @test_inferred forward(*, 2, 3)
@test_inferred(back(1))

_sincos(x) = sin(cos(x))

y, back = @test_inferred forward(_sincos, 0.5)
@test_inferred back(1)

f(x) = 3x^2 + 2x + 1

y, back = @test_inferred forward(f, 5)
@test y == 86
@test_inferred(back(1))

y, back = @test_inferred forward(Core._apply, +, (1, 2, 3))
@test_inferred back(1)

# TODO fix bcast inference
# bcast(x) = x .* 5
# y, back = @test_inferred forward(bcast, [1,2,3])
# @test_inferred back([1,1,1])

foo = let a = 4
  x -> x*a
end

@test_inferred gradient(f -> f(5), foo)

getx(x) = x.x
y, back = @test_inferred forward(getx, (x=1,y=2.0))
@test_inferred back(1)

# TODO
# MRP:
#     foo(f) = Ref((f,))
#     @code_typed foo(Complex)
# @test_inferred forward(Complex, 1, 2)

# Checks that use control flow
if PartialP.usetyped
  include("typed.jl")
end
