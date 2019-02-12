push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MagicTreeBoosting

using Profile

Random.seed!(123456)

feature_count = 200
point_count   = 100_000

X       = randn(Float32, (point_count, feature_count))
y       = round.(rand(MagicTreeBoosting.Prediction, point_count))
weights = rand(MagicTreeBoosting.DataWeight, point_count)


bin_splits = prepare_bin_splits(X)
X_binned   = apply_bins(X, bin_splits)

trees = train_on_binned(X_binned, y, weights = weights, iteration_count = 2)

# @profile trees = train_on_binned(X_binned, y, weights = weights)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

@time train_on_binned(X_binned, y, weights = weights)
@time train_on_binned(X_binned, y, weights = weights)
@time train_on_binned(X_binned, y, weights = weights)
@time train_on_binned(X_binned, y, weights = weights)
@time train_on_binned(X_binned, y, weights = weights)
