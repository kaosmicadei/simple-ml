#=
This file contains a simple implementation of a linear computational graph.
Since it doesn't contain the AutoGrad implementation, a `Chain` object
cannot be trained. Usually the AutoGrad is implemented using
*Automatic Differentiation* and *backpropagration*.

Tip #1: You probably saw in the documentation that `;` in Julia is a way to
indicate a newline on the code (and on the REPL means "suppress the output")
So, write `(a; b; c)` is equivalent to

    begin
      a
      b
      c
    end

Tip #2: In Julia, you can make any object callable (everything can become a
function). In documentation, take a look at the "Function-like objects" in
the "Methods" section.

Tip #3: To simplify itearions like `(x=x₀; for xᵢ in values x=f(x,xᵢ) end; x)`,
functional programming languages implements a functionis called: `reduce`,
`foldl` and `foldr` being `l` and `r` indicating the associative direction.

    foldl(•, [a,b,c]) => (a•b)•c
    foldr(•, [a,b,c]) => a•(b•c)

You can, as well, set a particular initial value.

    foldl(•,[b,c,d]; init=a) => ((a•b)•c)•d
=#

# LAYERS
abstract type Layer end

"""
A sequence of layers of neuralnets or functions that can be applied to a layer.
"""
struct Chain <: Layer
  layers::Array{Union{Layer,Function},1}
end
Chain(layers...) = (Chain ∘ collect)(layers)
(chain::Chain)(x) = foldl( (i,l) -> l(i), chain.layers; init=x )

"""
The basic layer to build a neural network. The `σ` is the activation function
which has the default value `σ=identity`.
"""
mutable struct Dense <: Layer
  weights::Array{Float32,2}
  linearcoef::Array{Float32,1}
  activation
end
Dense(ins::Int, outs::Int, σ) = Dense(rand(Float32, (outs, ins)), rand(Float32, outs), σ)
Dense(ins::Int, outs::Int) = Dense(ins, outs, identity)
(layer::Dense)(x) = layer.activation(layer.weights * x .+ layer.linearcoef)

# FUNCTIONS
"""
Normalisation of `xs`.
"""
softmax(x) = (w=exp.(x); w ./ sum(w; dims=1))

"""
Replaces negative values on the input by zero.
"""
relu(x) = max.(0, x)

# TEST
# ℝ³ -> ℝ²
model = Chain(
    Dense(3, 10, relu),
    Dense(10,2),
    softmax)

x = rand(3,2)
@show x
@info "Single input" model(x[:,1])
@info "Batched input (columns, batchsize=2)" model(x)
