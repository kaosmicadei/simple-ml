@info "Loading packages"
using Statistics: mean

@info "Preparing data and creating model"
# Inputs
# I'm using the columns as inputs because is more easy to create batches and
# it's also the way Flux uses data.
x = [0 0 1 1;
     0 1 0 1]

# Outputs
y = [0 1 1 0]

# Activation fuctions
# ReLU
relu(x) = max(0, x)
drelu(x) = (1 + sign(x)) / 2         # dσ/dx
# Sigmoid
sigmoid(x) = 1 / (1+exp(-x))
dsigmoid(x) = let s=sigmoid(x); s*(1-s) end     # dσ/dx

# Model a XOR gate we need two dense layers with the following parameters.
# Hidden layer: two percptrons.
W₁ = rand(2,2)   # 2-inputs, 2-outputs (NAND and OR)
b₁ = rand(2,1)
σ₁, dσ₁ = sigmoid, dsigmoid
layer1(x) = σ₁.(W₁*x .+ b₁)
# Output layer: one percptron.
W₂ = rand(1,2)   # 2-inputs, 1-output (AND)
b₂ = rand(1,1)
σ₂, dσ₂ = relu, drelu
layer2(x) = σ₂.(W₂*x .+ b₂)
model(x) = (layer2 ∘ layer1)(x)

# As a cost function we'll use mean((ŷ-y)²) where `y` is the training output
# data and `ŷ` is the prediction. However, here we are intesrest in the
# derivative of the cost function.

# Training
for epoch in 1:2_000
  print("Epoch $(epoch)\r")
  # Calulate the gradient of the loss function
  y₁ = layer1(x)
  δ = 2 * mean(model(x) .- y; dims=1)
  g₂ = δ .* dσ₂.(W₂*y₁ .+ b₂)
  g₁ = (transpose(W₂) * g₂) .*  dσ₁.(W₁*x .+ b₁)
  ∇W₁ = let (_, lx) = size(x)
    foldl((m,i) -> m + g₁[:,i] * transpose(x[:,i]), 1:lx; init=zeros(size(W₁)))
  end
  ∇b₁ = mean(g₁; dims=2)
  ∇W₂ = let (_, ly) = size(y₁)
    foldl((m,i) -> m + g₂[:,i] * transpose(y₁[:,i]), 1:ly; init=zeros(size(W₂)))
  end
  ∇b₂ = mean(g₂; dims=2)
  # Learning rate
  η = 0.1
  # Update parameters
  W₁ .-= η * ∇W₁
  b₁ .-= η * ∇b₁
  W₂ .-= η * ∇W₂
  b₂ .-= η * ∇b₂
end

# As final result we got...
@info "Final result" model(x)
@debug "Hidden layer" W₁ b₁
@debug "Ootput layer" W₂ b₂
