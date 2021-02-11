@info "Loading packages"
using Statistics: mean

@info "Preparing data and creating model"
# Inputs
x = [0 0 1 1;
     0 1 0 1]

# Outputs
y = [0 0 0 1]

# Activation fuctions
# ReLU
σ(x) = max(0, x)
dσ(x) = (1 + sign(x)) / 2         # dσ/dx
# Sigmoid
# σ(x) = 1 / (1+exp(-x))
# dσ(x) = let s=σ(x); s*(1-s) end     # dσ/dx

# Model an AND gate we need only one dense layer with the following parameters.
W = rand(1,2)   # 2-inputs, 1-output
b = rand(1,1)
model(x) = σ.(W*x .+ b)

# As a cost function we'll use mean((ŷ-y)²) where `y` is the training output
# data and `ŷ` is the prediction.

# Training
for epoch in 1:500
  print("Epoch $(epoch)\r")
  # Calulate the gradient of the loss function
  δ = 2 * mean(model(x) .- y; dims=1)
  gs = δ .* dσ.(W*x .+ b)
  ∇W = let (_, len) = size(x)
    foldl((m,i) -> m + gs[:,i] * transpose(x[:,i]), 1:len; init=zeros(size(W)))
  end
  ∇b = mean(gs; dims=2)
  # Learning rate
  η = 0.1
  # Update parameters
  W .-= η * ∇W
  b .-= η * ∇b
end

# As final result we got...
@info "Final result" model(x)
@debug "Parameters" W b
