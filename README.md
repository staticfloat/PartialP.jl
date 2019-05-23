<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/zygote.png"/>
</p>

[![Build Status](https://travis-ci.org/staticfloat/PartialP.jl.svg?branch=master)](https://travis-ci.org/staticfloat/PartialP.jl) [![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/PartialP.jl/dev)

`] add PartialP`

PartialP is a working prototype for source-to-source automatic differentiation (AD) in Julia, and the next-gen AD system for the [Flux](https://github.com/FluxML/Flux.jl) differentiable programming framework. For more details and benchmarks of PartialP's technique, see [our paper](https://arxiv.org/abs/1810.07951).

You probably don't want to use PartialP yet (except as a preview of the future). Instead use Flux's built-in AD, which is API-compatible, and at some point in the near future you'll get a free speed boost.

```julia
julia> using PartialP

julia> f(x) = 5x + 3

julia> f(10), f'(10)
(53, 5)

julia> @code_llvm f'(10)
define i64 @"julia_#625_38792"(i64) {
top:
  ret i64 5
}
```

"Source-to-source" means that PartialP hooks into Julia's compiler, and generates the backwards pass for you – as if you had written it by hand.

Without compromising on performance, PartialP supports the full flexibility and dynamism of the Julia language, including control flow, recursion, closures, structs, dictionaries, and more.

```julia
julia> fs = Dict("sin" => sin, "cos" => cos, "tan" => tan);

julia> derivative(x -> fs[readline()](x), 1)
sin
0.5403023058681398
```

Defining custom gradients is a cinch, and errors have good stacktraces.

```julia
julia> using PartialP: @adjoint

julia> add(a, b) = a + b

julia> @adjoint add(a, b) = add(a, b), Δ -> (Δ, Δ)
```

To support large machine learning models with many parameters, PartialP can differentiate implicitly-used parameters, as opposed to just function arguments.

```julia
julia> W, b = rand(2, 3), rand(2);

julia> predict(x) = W*x .+ b;

julia> g = gradient(Params([W, b])) do
         sum(predict([1,2,3]))
       end
Grads(...)

julia> g[W], g[b]
([1.0 2.0 3.0; 1.0 2.0 3.0], [1.0, 1.0])
```

## Caveat Emptor

PartialP is in an early stage and may break, but issue reports and beta testing are welcome. In particular PartialP does not yet have comprehensive gradient definitions and may fail if it hits complex code in Base Julia.

PartialP's runtime performance should generally be good, but compile times are not optimised, so calling `gradient` the first time can have noticeable lag. [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) is recommended to avoid measuring JIT time.

A current limitation is that PartialP will not automatically see redefined functions (for example if you call `gradient(f, x)`, then redefine `f`, then take the gradient again). You can call `PartialP.refresh()` to completely reset what PartialP sees. It's often useful to have this in your script/notebook after function definitions.

The Julia compiler does not yet support all features needed to make PartialP fast, particularly in the presence of control flow. Until these are officially supported PartialP [contains a flag](https://github.com/staticfloat/PartialP.jl/blob/5d7ea65ef0cdbd07c30584b5d66d13a66c7e0c21/src/PartialP.jl#L12) to enable faster operation. If you can handle the additional caveats it's a good way to see PartialP's peak performance.
