@info "Loading packages"
using Flux
using Flux: @epochs
using Flux: Data.DataLoader

@info "Preparing data and creating model"
# Inputs
x_train = [0 0 1 1;
           0 1 0 1]

# Outputs
y_train = [0 1 1 0]

# Combine the data into batches with size=1 (stochastic optimisation)
data_train = DataLoader(x_train, y_train)

# Model a XOR gate requires two layers
model = Chain(
    Dense(2,2,σ),       # 2-inputs, 2-outputs (NAND and OR)
    Dense(2,1,σ))       # 2-inputs, 1-output (AND)

# As a cost function we'll use mean((ŷ-y)²) where `y` is the training output
# data and `ŷ` is the prediction.
loss(x,y) = Flux.Losses.mse(model(x), y)

# As optimiser maybe would be a good idea to use momentum for fast convergence
# with the default learning rate (0.1).
opt = Momentum()

# Since the dataset is too small for a proper training we'll a large number of
# epochs.
@epochs 20_000 Flux.train!(loss, params(model), data_train, opt)

# As final result we got...
@info "Final result" model(x_train)
@debug "Parameters" params(model)
