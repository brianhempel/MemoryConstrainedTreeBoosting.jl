module MagicTreeBoosting

export train, train_one_epoch, apply_bin_splits, predict

import Random


default_config = (
  bin_count     = 255,
  epoch_count   = 10,
  max_leaves    = 32,
  learning_rate = 0.1,
)

function get_config_field(config, key)
  if !isa(config, Tuple{}) && haskey(config, key)
    config[key]
  else
    default_config[key]
  end
end

data_count(X)    = size(X,1)
feature_count(X) = size(X,2)

const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0

const BinSplits = Vector{T} where T <: AbstractFloat

abstract type Tree{T <: AbstractFloat} end

mutable struct SplitCandidate{T}
  expected_Δloss :: T
  feature_i      :: Int64
  split_i        :: UInt8
end

mutable struct HistBin{T}
  Σ∇loss  :: T
  Σ∇∇loss :: T

  HistBin() = new{Float64}(0.0, 0.0)
end

mutable struct Node{T} <: Tree{T}
  feature_i :: Int64
  split_i   :: UInt8
  left      :: Tree{T}
  right     :: Tree{T}
end

mutable struct Leaf{T} <: Tree{T}
  weight :: T
  is :: Union{Vector{Int64},Nothing}                        # Transient. Needed during tree growing.
  maybe_split_candidate :: Union{SplitCandidate{T},Nothing} # Transient. Needed during tree growing.
end


function print_tree(tree :: Tree{T}, level) where T
  indentation = repeat("    ", level)
  if isa(tree, Node)
    println(indentation * "Node: feature_i $(tree.feature_i)\tsplit_i $(tree.split_i)")
    print_tree(tree.left,  level + 1)
    print_tree(tree.right, level + 1)
  else
    println(indentation * "Leaf: weight $(tree.weight)\tis_count $(length(tree.is))")
  end
end


function tree_leaves(tree :: Tree{T}) :: Vector{Leaf{T}} where T
  if isa(tree, Leaf)
    [tree]
  else
    vcat(tree_leaves(tree.left), tree_leaves(tree.right))
  end
end


# Mutates tree, but returns the entire tree since old may be the root.
function replace_leaf(tree, old, replacement)
  if tree === old
    replacement
  elseif isa(tree, Node)
    tree.left  = replace_leaf(tree.left,  old, replacement)
    tree.right = replace_leaf(tree.right, old, replacement)
    tree
  else
    tree
  end
end


# Mutates and returns tree.
function scale_leaf_weights(tree, learning_rate)
  for leaf in tree_leaves(tree)
    leaf.weight *= learning_rate
  end
  tree
end


# Returns vector of untransformed scores (linear, pre-sigmoid).
function apply_tree(X_binned :: Array{UInt8,2}, tree :: Tree{T}) :: Vector{T} where T
  scores = zeros(T, data_count(X_binned))
  apply_tree!(X_binned, tree, scores)
end


# Mutates scores.
function apply_tree!(X_binned :: Array{UInt8,2}, tree :: Tree{T}, scores :: Vector{T}) :: Vector{T} where T
  for i in 1:data_count(X_binned)
    node = tree
    while !isa(node, Leaf)
      if X_binned[i, node.feature_i] <= node.split_i
        node = node.left
      else
        node = node.right
      end
    end
    scores[i] += node.weight
  end

  scores
end


# Returns vector of untransformed scores (linear, pre-sigmoid). Does not mutate starting_scores.
function apply_trees(X_binned :: Array{UInt8,2}, trees :: Vector{Tree{T}}, starting_scores = nothing) :: Vector{T} where T
  scores = zeros(T, data_count(X_binned))

  if starting_scores != nothing
    scores += starting_scores
  end

  for tree in trees
    apply_tree!(X_binned, tree, scores)
  end

  scores
end


# Returns vector of predictions ŷ (post-sigmoid).
function predict(X_binned :: Array{UInt8,2}, trees :: Vector{Tree{T}}, starting_scores = nothing) :: Vector{T} where T
  scores = apply_trees(X_binned, trees, starting_scores)
  σ.(scores)
end


# Aim for a roughly equal number of data points in each bin.
function prepare_bin_splits(X, bin_count) :: Vector{BinSplits}
  if bin_count < 2 || bin_count > 255
    error("prepare_bin_splits: bin_count must be between 2 and 255")
  end
  ideal_sample_count = bin_count * 1_000
  is = collect(Iterators.take(Random.shuffle(1:data_count(X)), ideal_sample_count))

  sample_count = length(is)
  split_count = bin_count - 1

  bin_splits =
    map(1:feature_count(X)) do j
      sorted_feature_values = sort(X[is, j])

      splits = zeros(eltype(sorted_feature_values), split_count)

      for split_i in 1:split_count
        split_sample_i = Int64(floor(sample_count / bin_count * split_i))
        value_below_split = sorted_feature_values[split_sample_i]
        value_above_split = sorted_feature_values[min(split_sample_i + 1, sample_count)]
        splits[split_i] = (value_below_split + value_above_split) / 2
      end

      splits
    end

  bin_splits
end


function apply_bins(X, bin_splits) :: Array{UInt8,2}
  X_binned = zeros(UInt8, size(X))

  for j in 1:feature_count(X)
    splits_for_feature = bin_splits[j]
    bin_count = length(splits_for_feature) + 1
    for i in 1:data_count(X)
      value   = X[i,j]
      split_i = findfirst(split_value -> split_value > value, splits_for_feature)
      bin_i   = split_i == nothing ? bin_count : split_i

      X_binned[i,j] = UInt8(bin_i) # Store as 1-255 to match Julia indexing. We leave 0 unused but saves us from having to remember to convert.
    end
  end

  X_binned
end


function build_one_tree(X_binned :: Array{UInt8,2}, y, ŷ; config...) # y = labels, ŷ = predictions so far
  tree = Leaf(optimal_weight(y, ŷ), collect(1:length(y)), nothing)

  tree_changed = true

  while tree_changed
    # print_tree(tree, 0)
    # println()
    # println()
    (tree_changed, tree) = perhaps_split_tree(tree, X_binned, y, ŷ, config...)
  end

  tree
end


# Trains for epoch_count rounds and teturns (bin_splits, prior_and_new_trees).
function train(X, y; bin_splits=nothing, prior_trees=Tree{Float64}[], config...)
  if bin_splits == nothing
    bin_splits = prepare_bin_splits(X, get_config_field(config, :bin_count))
  end
  # println(bin_splits)

  X_binned = apply_bins(X, bin_splits)
  # println(X_binned)

  scores = apply_trees(X_binned, prior_trees) # Linear scores, before sigmoid transform.

  trees = []

  for epoch_i in 1:get_config_field(config, :epoch_count)
    (scores, tree) = train_one_epoch(X_binned, y, scores; config...)

    ŷ = σ.(scores)
    epoch_loss = sum(logloss.(y, ŷ)) / length(y)
    # println(ŷ)
    println("Epoch $epoch_i training loss: $epoch_loss")
    push!(trees, tree)
  end

  (bin_splits, vcat(prior_trees, trees))
end

# Returns (new_scores, tree)
function train_one_epoch(X_binned, y, scores; config...)
  ŷ = σ.(scores)
  # println(ŷ)
  tree = build_one_tree(X_binned, y, ŷ)
  tree = scale_leaf_weights(tree, get_config_field(config, :learning_rate))
  new_scores = scores .+ apply_tree(X_binned, tree)
  (new_scores, tree)
end


σ(x) = 1.0 / (1.0 + exp(-x))

# Copied from Flux.jl.
logloss(y, ŷ) = -y*log(ŷ + ε) - (1 - y)*log(1 - ŷ + ε)

# Derivatives with respect to margin (i.e. pre-sigmoid values, but you still provide the post-sigmoid probability).
#
# It's a bit of math but it works out nicely. (Verified against XGBoost.)
∇logloss(y, ŷ) = ŷ - y
∇∇logloss(ŷ)   = ŷ * (1 - ŷ) # Interestingly, not dependent on y. XGBoost adds an ε term


# Assuming binary classification with log loss.
function optimal_weight(y, ŷ)
  Σ∇loss  = sum(∇logloss.(y, ŷ))
  Σ∇∇loss = sum(∇∇logloss.(ŷ))

  # And the minima is at:
  -Σ∇loss / (Σ∇∇loss + ε)
end


# -0.5 * (Σ∇loss)² / (Σ∇∇loss)
function leaf_expected_Δloss(Σ∇loss, Σ∇∇loss)
  -0.5 * Σ∇loss * Σ∇loss / (Σ∇∇loss + ε)
end


# Mutates tree, but also returns the tree in case tree was a lone leaf.
#
# Returns (bool, tree) where bool is true if any split was made, otherwise false.
function perhaps_split_tree(tree, X_binned :: Array{UInt8,2}, y, ŷ, config...)
  leaves = tree_leaves(tree)

  if length(leaves) >= get_config_field(config, :max_leaves)
    return (false, tree)
  end

  # Ensure split_candidate on all leaves
  for leaf in leaves
    if leaf.maybe_split_candidate == nothing
      # Find best feature and split
      # Expected Δlogloss at leaf = -0.5 * (Σ ∇loss)² / (Σ ∇∇loss)

      best_expected_Δloss = 0.0
      best_feature_i      = 1
      best_split_i        = UInt8(1)

      for feature_i in 1:feature_count(X_binned)
        max_bins = maximum(@view X_binned[:,feature_i])

        hist_bins = map(bin_i -> HistBin(), 1:max_bins)

        this_leaf_Σ∇loss  = 0.0
        this_leaf_Σ∇∇loss = 0.0

        for i in leaf.is
          bin_i    = X_binned[i, feature_i]
          hist_bin = hist_bins[bin_i]

          ∇loss  = ∇logloss(y[i], ŷ[i])
          ∇∇loss = ∇∇logloss(ŷ[i])

          hist_bin.Σ∇loss  += ∇loss
          hist_bin.Σ∇∇loss += ∇∇loss

          this_leaf_Σ∇loss  += ∇loss
          this_leaf_Σ∇∇loss += ∇∇loss
        end

        this_leaf_expected_Δloss = leaf_expected_Δloss(this_leaf_Σ∇loss, this_leaf_Σ∇∇loss)

        left_Σ∇loss  = 0.0
        left_Σ∇∇loss = 0.0

        for bin_i in UInt8(1):UInt8(max_bins-1)
          hist_bin = hist_bins[bin_i]

          left_Σ∇loss  += hist_bin.Σ∇loss
          left_Σ∇∇loss += hist_bin.Σ∇∇loss

          right_Σ∇loss  = this_leaf_Σ∇loss  - left_Σ∇loss
          right_Σ∇∇loss = this_leaf_Σ∇∇loss - left_Σ∇∇loss

          expected_Δloss =
            -this_leaf_expected_Δloss +
            leaf_expected_Δloss(left_Σ∇loss,  left_Σ∇∇loss) +
            leaf_expected_Δloss(right_Σ∇loss, right_Σ∇∇loss)

          if expected_Δloss < best_expected_Δloss
            best_expected_Δloss = expected_Δloss
            best_feature_i      = feature_i
            best_split_i        = bin_i
          end
        end # for bin_i in 1:(max_bins-1)
      end # for feature_i in 1:feature_count(X)

      leaf.maybe_split_candidate = SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)

    end # if leaf.maybe_split_candidate == nothing
  end # for leaf in leaves

  # Expand best split_candiate (if any)

  (expected_Δloss, leaf_i_to_split) = findmin(map(leaf -> leaf.maybe_split_candidate.expected_Δloss, leaves))

  if expected_Δloss < 0.0
    # We have a usable split!

    leaf_to_split = leaves[leaf_i_to_split]

    feature_i = leaf_to_split.maybe_split_candidate.feature_i
    split_i   = leaf_to_split.maybe_split_candidate.split_i

    # A bit of recalculation below but should be fine.

    left_is  = filter(i -> X_binned[i,feature_i] <= split_i, leaf_to_split.is)
    right_is = filter(i -> X_binned[i,feature_i] >  split_i, leaf_to_split.is)

    left_ys  = y[left_is]
    left_ŷs  = ŷ[left_is]
    right_ys = y[right_is]
    right_ŷs = ŷ[right_is]

    left_weight  = optimal_weight(left_ys,  left_ŷs)
    right_weight = optimal_weight(right_ys, right_ŷs)

    left_leaf  = Leaf(left_weight,  left_is,  nothing)
    right_leaf = Leaf(right_weight, right_is, nothing)

    new_node = Node(feature_i, split_i, left_leaf, right_leaf)

    tree = replace_leaf(tree, leaf_to_split, new_node)

    (true, tree)
  else
    (false, tree)
  end
end

end # module MagicTreeBoosting