# Stacks

mutable struct Stack{T}
  idx::Int
  data::Vector{T}
end

Stack(data::Vector{T}) where T =
  Stack{T}(length(data), data)

function Base.pop!(stk::Stack)
  i = stk.idx
  stk.idx = i == 1 ? length(stk.data) : i-1
  @inbounds return stk.data[i]
end

function _push!(a::Vector{T}, x::T) where T
  Base._growend!(a, 1)
  @inbounds a[end] = x
  return
end

# Emit

xstack(T) = stmt(Expr(:call, Vector{T}), type = Vector{T})

function alphauses(b)
  us = Set{Alpha}()
  postwalk(x -> x isa Alpha && push!(us, x), b)
  return us
end

xtuple(xs...) = xcall(:tuple, xs...)

concrete(T::DataType) = T
concrete(::Type{Type{T}}) where T = typeof(T)
concrete(T) = Any

runonce(b) = b.id in (1, length(b.ir.blocks))

function forward_stacks!(adj, F)
  stks, recs = [], []
  pr = adj.primal
  for b in blocks(pr), α in alphauses(block(adj.adjoint, b.id))
    if runonce(b)
      push!(recs, Variable(α))
    else
      T = exprtype(pr, Variable(α))
      stk = pushfirst!(pr, xstack(T))
      push!(recs, stk)
      push!(b, xcall(PartialP, :_push!, stk, Variable(α)))
    end
    push!(stks, (b.id, alpha(α)))
  end
  args = arguments(pr)[3:end]
  T = Tuple{concrete.(exprtype.((pr,), recs))...}
  isconcretetype(T) || (T = Any)
  rec = push!(pr, xtuple(recs...))
  if usetyped && length(pr.blocks) > 1
    rec = push!(pr, Expr(:call, Pullback{F,T}, rec))
  else
    P = length(pr.blocks) == 1 ? Pullback{F} : Pullback{F,Any}
    rec = push!(pr, Expr(:call, P, rec))
  end
  ret = xtuple(pr.blocks[end].branches[end].args[1], rec)
  ret = push!(pr, ret)
  pr.blocks[end].branches[end].args[1] = ret
  return pr, stks
end

function reverse_stacks!(adj, stks)
  ir = adj.adjoint
  entry = blocks(ir)[end]
  self = argument!(entry, at = 1)
  t = pushfirst!(blocks(ir)[end], xcall(:getfield, self, QuoteNode(:t)))
  repl = Dict()
  runonce(b) = b.id in (1, length(ir.blocks))
  for b in blocks(ir)
    for (i, (b′, α)) in enumerate(stks)
      b.id == b′ || continue
      if runonce(b)
        val = insertafter!(ir, t, xcall(:getindex, t, i))
      else
        stk = push!(entry, xcall(:getindex, t, i))
        stk = push!(entry, xcall(PartialP, :Stack, stk))
        val = pushfirst!(b, xcall(:pop!, stk))
      end
      repl[α] = val
    end
  end
  return IRTools.prewalk!(x -> get(repl, x, x), ir)
end

function stacks!(adj, T)
  forw, stks = forward_stacks!(adj, T)
  back = reverse_stacks!(adj, stks)
  permute!(back, length(back.blocks):-1:1)
  IRTools.domorder!(back)
  return forw, back
end

varargs(m::Method, n) = m.isva ? n - m.nargs + 1 : nothing

meta(T) = (usetyped ? IRTools.typed_meta : IRTools.meta)(T)

function getmeta(T)
  m = meta(T)
  (usetyped && m != nothing) || return m
  any(x -> isexpr(x, :goto, :gotoifnot), m.code.code) || return IRTools.meta(T)
  return m
end

function _lookup_grad(T)
  (m = getmeta(T)) == nothing && return
  m isa IRTools.TypedMeta && m.ret == Union{} && return
  va = varargs(m.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IR(m), varargs = va, normalise = false), T)
  m, forw, back
end

function stacklines(T::Type)
  adj = Adjoint(IR(meta(T)), normalise = false)
  recs = []
  for b in blocks(adj.adjoint), α in alphauses(b)
    push!(recs, IRTools.exprline(adj.primal, Variable(α)))
  end
  return recs
end
