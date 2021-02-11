# LAYERS
abstract type Layer end

struct Chain <: Layer
  layers::Array{Union{Layer,Function},1}
end
Chain(layers...) = (Chain ∘ collect)(layers)
(chain::Chain)(x) = foldl( (i,l) -> l(i), chain.layers; init=x )

mutable struct Dense <: Layer
  weights::Array{Float32,2}
  linearcoef::Array{Float32,1}
  activation
end
Dense(ins::Int, outs::Int, σ) = Dense(rand(Float32, (outs, ins)), rand(Float32, outs), σ)
Dense(ins::Int, outs::Int) = Dense(ins, outs, identity)
(layer::Dense)(x) = layer.activation(layer.weights * x .+ layer.linearcoef)

# FUNCTIONS
softmax(x) = (w=exp.(x); w ./ sum(w; dims=1))

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
