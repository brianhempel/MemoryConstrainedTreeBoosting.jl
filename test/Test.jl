push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using MagicTreeBoosting

# 10x3 array
X = Float32[0.314421   0.622812  0.758382
            0.592915   0.799313  0.82079
            0.121827   0.946241  0.00250338
            0.248926   0.44318   0.375335
            0.0302476  0.365399  0.18079
            0.0598305  0.463519  0.0609047
            0.510357   0.488909  0.655259
            0.358932   0.98549   0.472883
            0.443199   0.4372    0.476574
            0.195279   0.576752  0.436448]

y = [0.0
     1.0
     0.0
     0.0
     0.0
     0.0
     1.0
     1.0
     1.0
     1.0]

weights = [10.0
           0.1
           0.1
           1.0
           1.0
           1.0
           1.0
           1.0
           10.0
           1.0]

# weights = weights ./ weights

(bin_splits, trees) = train(X, y, weights = weights, bin_count = 4, iteration_count = 20, min_data_weight_in_leaf = 2.0, learning_rate = 0.3)

bin_splits :: Vector{MagicTreeBoosting.BinSplits{Float32}}

save_path = tempname()

save(save_path, bin_splits, trees)

bin_splits, trees = load(save_path)

# Make sure load gives the types back correctly.
bin_splits :: Vector{MagicTreeBoosting.BinSplits{Float32}}
trees      :: Vector{MagicTreeBoosting.Tree}

X_binned = apply_bins(X, bin_splits)

println(predict_on_binned(X_binned, trees))
println(y)

bin_splits          = prepare_bin_splits(X, 4)
X_binned_compressed = bin_and_compress(X[1:3,:], bin_splits)
X_binned_compressed = bin_and_compress(X[4:10,:], bin_splits, prior_data = X_binned_compressed)
X_binned_compressed = finalize_loading(X_binned_compressed)

trees = train_on_binned(X_binned_compressed, y, weights = weights, iteration_count = 20, min_data_weight_in_leaf = 2.0, learning_rate = 0.3)

println(predict_on_binned(X_binned_compressed, trees))
println(y)
