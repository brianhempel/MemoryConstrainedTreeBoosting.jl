push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MemoryConstrainedTreeBoosting

Random.seed!(123456)

# 10x4 array
X = Float32[0.314421   0.622812  0.0 0.758382
            0.592915   0.799313  1.0 0.82079
            0.121827   0.946241  0.0 0.00250338
            0.248926   0.44318   0.0 0.375335
            0.0302476  0.365399  0.0 0.18079
            0.0598305  0.463519  0.0 0.0609047
            0.510357   0.488909  1.0 0.655259
            0.358932   0.98549   1.0 0.472883
            0.443199   0.4372    1.0 0.476574
            0.195279   0.576752  1.0 0.436448]

y = Float32[0.0
            1.0
            0.0
            0.0
            0.0
            0.0
            1.0
            1.0
            1.0
            1.0]

weights = Float32[10.0
                  0.1
                  0.1
                  1.0
                  1.0
                  1.0
                  1.0
                  1.0
                  10.0
                  1.0]

validation_X       = X[1:3, :]
validation_y       = y[1:3]
validation_weights = weights[1:3]

# weights = weights ./ weights

(bin_splits, trees) =
  train(
      X, y,
      weights = weights,
      bin_count = 4,
      iteration_count = 20,
      min_data_weight_in_leaf = 2.0,
      learning_rate = 0.3,
      bagging_temperature = 0.0,
      exclude_features   = [3],
      validation_X       = validation_X,
      validation_y       = validation_y,
      validation_weights = validation_weights
    )

bin_splits :: Vector{MemoryConstrainedTreeBoosting.BinSplits{Float32}}

save_path = tempname()

save(save_path, bin_splits, trees)

bin_splits2, trees2 = load(save_path)

# Make sure load gives the types back correctly.
bin_splits2 :: Vector{MemoryConstrainedTreeBoosting.BinSplits{Float32}}
trees2      :: Vector{MemoryConstrainedTreeBoosting.Tree}

@assert bin_splits == bin_splits2
@assert repr(trees) == repr(trees2)
@assert MemoryConstrainedTreeBoosting.tree_to_dict.(trees) == MemoryConstrainedTreeBoosting.tree_to_dict.(trees2)

X_binned = apply_bins(X, bin_splits2)

println(predict_on_binned(X_binned, trees2))
println(y)

println("Unbinned predictor")
unbinned_predict = load_unbinned_predictor(save_path)

println(unbinned_predict(X))
println(y)

@assert sum(abs.(predict_on_binned(X_binned, trees2) - unbinned_predict(X))) < 0.00001
