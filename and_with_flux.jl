@info "Loading packages"
using Flux
using Flux: @epochs
using Flux: Data.DataLoader

@info "Preparing data and creating model"
# Inputs
x_train = [0 0 1 1;
           0 1 0 1]

# Outputs
y_train = [0 0 0 1]

# Combine the data into batches
data_train = DataLoader(x_train, y_train; batchsize=4)

# Model an AND gate requires only one layer
model = Dense(2,1,σ)

# As a cost function we'll use mean((ŷ-y)²) where `y` is the training output
# data and `ŷ` is the prediction.
loss(x,y) = Flux.Losses.mse(model(x), y)

# As optimiser we'll use the standard gradient descent with the default
# learning rate (0.1).
opt = Descent()

# Since the dataset is too small for a proper training we'll a large number of
# epochs.
@epochs 5_000 Flux.train!(loss, params(model), data_train, opt)

# As final result we got...
@info "Final result" model(x_train)
@debug "Parameters" params(model)
