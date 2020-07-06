push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MemoryConstrainedTreeBoosting

using Profile

scores  = collect(reinterpret(Float32, read(joinpath(@__DIR__, "calibration_test", "110_trees_loss_0.0019571164.validation_scores"))))
y       = collect(reinterpret(Float32, read(joinpath(@__DIR__, "calibration_test", "validation_labels"))))
weights = collect(reinterpret(Float32, read(joinpath(@__DIR__, "calibration_test", "validation_weights"))))

loss_before = MemoryConstrainedTreeBoosting.compute_mean_logloss(y, scores, weights)

println("Loss before:\t$loss_before")


loss_before = MemoryConstrainedTreeBoosting.compute_mean_logloss(y, scores, weights)
println("Loss before scaling: $loss_before")
score_transformer = MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)
loss_after = MemoryConstrainedTreeBoosting.compute_mean_logloss(y, scores, weights; score_transformer = score_transformer)
println("Loss after scaling: $loss_after")

high_filter = scores .>= -10 # ~0.001
println(length(scores[high_filter]))

loss_before = MemoryConstrainedTreeBoosting.compute_mean_logloss(y[high_filter], scores[high_filter], weights[high_filter])
println("High scores loss before scaling: $loss_before")
score_transformer = MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y[high_filter], scores[high_filter], weights[high_filter])
loss_after = MemoryConstrainedTreeBoosting.compute_mean_logloss(y[high_filter], scores[high_filter], weights[high_filter]; score_transformer = score_transformer)
println("High scores loss after scaling: $loss_after")


low_filter = scores .< -10 # ~0.001
println(length(scores[low_filter]))

loss_before = MemoryConstrainedTreeBoosting.compute_mean_logloss(y[low_filter], scores[low_filter], weights[low_filter])
println("Low scores loss before scaling: $loss_before")
score_transformer = MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y[low_filter], scores[low_filter], weights[low_filter])
loss_after = MemoryConstrainedTreeBoosting.compute_mean_logloss(y[low_filter], scores[low_filter], weights[low_filter]; score_transformer = score_transformer)
println("Low scores loss after scaling: $loss_after")


# @time MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)
# @time MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)
# @time MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)
# @time MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)
# @time MemoryConstrainedTreeBoosting.compute_platt_scaling_score_transformer(y, scores, weights)


# loss_before = MemoryConstrainedTreeBoosting.compute_mean_logloss(y, scores, weights; score_transformer = score_transformer)
# println("Loss before min/max prediction: $loss_before")
# min_max_score_transformer = MemoryConstrainedTreeBoosting.compute_min_max_score_transformer(y, scores, weights; prior_score_transformer = score_transformer)
# loss_after = MemoryConstrainedTreeBoosting.compute_mean_logloss(y, scores, weights; score_transformer = min_max_score_transformer)
# println("Loss after min/max prediction: $loss_after")
#
# @time MemoryConstrainedTreeBoosting.compute_min_max_score_transformer(y, scores, weights; prior_score_transformer = score_transformer)
# @time MemoryConstrainedTreeBoosting.compute_min_max_score_transformer(y, scores, weights; prior_score_transformer = score_transformer)
# @time MemoryConstrainedTreeBoosting.compute_min_max_score_transformer(y, scores, weights; prior_score_transformer = score_transformer)
