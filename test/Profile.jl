push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MemoryConstrainedTreeBoosting

using Profile

Random.seed!(123456)

# For SREF, datapoint/feature ratio is ~370:1
feature_count = 5000
point_count   = feature_count*200

# X       = randn(Float32, (point_count, feature_count))
y       = round.(rand(MemoryConstrainedTreeBoosting.Prediction, point_count))
weights = rand(MemoryConstrainedTreeBoosting.DataWeight, point_count)


# bin_splits = prepare_bin_splits(X)
# X_binned   = apply_bins(X, bin_splits)
X_binned   = rand(UInt8(1):UInt8(255), (point_count, feature_count))
# X          = nothing

validation_range = 1:div(point_count,3)

iteration_callback =
  make_callback_to_track_validation_loss(
      X_binned[validation_range, :],
      y[validation_range];
      validation_weights = weights[validation_range]
    )

trees = train_on_binned(X_binned, y, weights = weights, iteration_count = 2, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
# trees = train_on_binned(X_binned, y, weights = weights, iteration_count = 100, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
#
# save_path = tempname()
# save(save_path, bin_splits, trees)
#
# X = randn(Float32, (point_count, feature_count))
#
# println("Binned predictor")
# bin_splits, trees = load(save_path)
# @time predict(X, bin_splits, trees)
# @time predict(X, bin_splits, trees)
#
# println("Unbinned predictor")
# unbinned_predict = load_unbinned_predictor(save_path)
#
# @time unbinned_predict(X)
# @time unbinned_predict(X)
#
# println(sum(abs.(predict(X, bin_splits, trees) - unbinned_predict(X))))
# @assert sum(abs.(predict(X, bin_splits, trees) - unbinned_predict(X))) < (0.00001 * point_count)

# @profile trees = train_on_binned(X_binned, y, weights = weights)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5, iteration_callback = iteration_callback)
