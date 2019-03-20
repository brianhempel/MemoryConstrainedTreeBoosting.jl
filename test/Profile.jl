push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
import MemoryConstrainedTreeBoosting

using Profile

transpose = "--transposed" in ARGS

# Random.seed!(123456)

feature_count = 1500
point_count   = 1_300_000
if transpose
  X = randn(Float32, (feature_count, point_count))
else
  X = randn(Float32, (point_count, feature_count))
end

Module = transpose ? MemoryConstrainedTreeBoosting.Transposed : MemoryConstrainedTreeBoosting

y       = round.(rand(Module.Prediction, point_count))
weights = rand(Module.DataWeight, point_count)


bin_splits = Module.prepare_bin_splits(X)
X_binned   = Module.apply_bins(X, bin_splits)
X          = nothing


trees = Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 2, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)

# @profile trees = Module.train_on_binned(X_binned, y, weights = weights)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

@time Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
@time Module.train_on_binned(X_binned, y, weights = weights, iteration_count = 10, feature_fraction = 0.5, max_leaves = 6, bagging_temperature = 0.5)
