push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MemoryConstrainedTreeBoosting

Random.seed!(123456)

σ = MemoryConstrainedTreeBoosting.σ

X, y, weights = begin
  a, b = 0.5f0, -1f0

  X = Array{Float32}(undef, (21,1))
  X[:,1] = -10:10

  y       = map(x -> σ(a*x + b),       @view X[:,1])
  weights = map(_ -> 10*rand(Float32), @view X[:,1])

  (X, y, weights)
end

bin_splits = prepare_bin_splits(X)
# println(bin_splits)

X_binned = apply_bins(X, bin_splits)

# unbinned = map(i -> MemoryConstrainedTreeBoosting.unbin(X_binned, bin_splits, 1, i), 1:length(y))
# println(X)
# println(unbinned)

max_delta_score = 10f0
min_data_to_regress_in_leaf = 5

leaf =
  MemoryConstrainedTreeBoosting.Leaf(
      0f0, 0f0, -Inf32, Inf32, 1, collect(1:length(y)), sum(weights)
    )

scores = zeros(MemoryConstrainedTreeBoosting.Score, length(y))
scratch_memory = MemoryConstrainedTreeBoosting.ScratchMemory(
    y; min_data_to_regress_in_leaf = min_data_to_regress_in_leaf
  )

new_leaf = MemoryConstrainedTreeBoosting.regress_in_leaves(
  leaf, X_binned, y, bin_splits, scores, weights,
  scratch_memory, 1, max_delta_score, min_data_to_regress_in_leaf
)

println("Initial tree:")
MemoryConstrainedTreeBoosting.print_tree(leaf)
println("Final tree (off by 0.25 because bin_splits):")
MemoryConstrainedTreeBoosting.print_tree(new_leaf)

MemoryConstrainedTreeBoosting.scale_leaf_Δscores!(new_leaf, 0.5)
println("Scaled by 0.5:")
MemoryConstrainedTreeBoosting.print_tree(new_leaf)

final_scores = predict_on_binned(X_binned, [new_leaf, new_leaf], bin_splits, output_raw_scores = true)


println("y:")
println(y)
println("predicted:")
println(σ.(final_scores))

abs_diff = sum(abs.(y - σ.(final_scores)))
println("abs diff: $abs_diff")

@assert abs_diff < 0.01
