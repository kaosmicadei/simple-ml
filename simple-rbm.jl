samples = [[1,-1,1,-1,1],
           [-1,1,-1,1,1],
           [1,1,-1,1,-1]]
@info "Samples" samples

batchsize = 128
training_data = hcat(rand(samples, batchsize)...)

visible_nodes = 5
hidden_nodes = 10

W = rand(hidden_nodes, visible_nodes)
h = rand(hidden_nodes, 1)
v = rand(visible_nodes, 1)

σ = tanh
sample(x) = (x .> rand(size(x)...)) |> m -> 2 .* m .- 1

reconstruct(x) = let
  hidden = σ.(W*x .+ h) |> sample
  σ.(W'hidden .+ v)
end

for epoch in 1:20_000
  print("Epoch: $(epoch)\r")
  # Contrastive Divergence
  positive_hidden = σ.(W*training_data .+ h)
  hidden = sample(positive_hidden)
  positive_assocs = positive_hidden * training_data'

  negative_visible = σ.(W'hidden .+ v) |> sample
  negative_hidden = σ.(W*negative_visible .+ h)
  negative_assocs = negative_hidden * negative_visible'

  η = 0.1
  W .+= η * (positive_assocs - negative_assocs)
  h .+= η * (sum(positive_hidden; dims=2) - sum(negative_hidden; dims=2))
  v .+= η * (sum(training_data; dims=2) - sum(negative_visible; dims=2))
end

test_data = hcat(rand(samples, 4)...)
let l=length(test_data)
  test_data[rand(1:l, l÷2)] .= 0
end
@info "Input test" test_data
@info "Reconstructed data" round.(Int, reconstruct(test_data))
