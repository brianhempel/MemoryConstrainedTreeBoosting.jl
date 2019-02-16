push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MagicTreeBoosting

using Profile

Random.seed!(123456)

feature_count = 150
point_count   = 12_000_000

X       = randn(Float32, (point_count, feature_count))
y       = round.(rand(MagicTreeBoosting.Prediction, point_count))
weights = rand(MagicTreeBoosting.DataWeight, point_count)


bin_splits = prepare_bin_splits(X)
X_binned   = apply_bins(X, bin_splits)

trees = train_on_binned(X_binned, y, weights = weights, iteration_count = 2, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)

# @profile trees = train_on_binned(X_binned, y, weights = weights)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
