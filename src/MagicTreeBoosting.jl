module MagicTreeBoosting

export train, train_on_binned, prepare_bin_splits, apply_bins, predict, predict_on_binned, save, load

import Random

import BSON


default_config = (
  bin_count               = 255,
  iteration_count         = 10,
  min_data_weight_in_leaf = 10.0,
  l2_regularization       = 1.0,
  max_leaves              = 32,
  max_depth               = 6,
  max_delta_score         = 1.0e10, # Before shrinkage.
  learning_rate           = 0.1,
  feature_fraction        = 1.0, # Per tree.
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

const Score      = Float64
const Loss       = Float64
const Prediction = Float64
const DataWeight = Float64

const BinSplits = Vector{T} where T <: AbstractFloat

abstract type Tree end

mutable struct SplitCandidate
  expected_Δloss :: Loss
  feature_i      :: Int64
  split_i        :: UInt8
end

mutable struct HistBin
  Σ∇loss      :: Loss
  Σ∇∇loss     :: Loss
  data_weight :: DataWeight

  HistBin() = new(0.0, 0.0, 0.0)
end

mutable struct Node <: Tree
  feature_i :: Int64
  split_i   :: UInt8
  left      :: Tree
  right     :: Tree
end

mutable struct Leaf <: Tree
  Δscore :: Score # Called "weight" in the literature
  is :: Union{Vector{Int64},Nothing}                     # Transient. Needed during tree growing.
  maybe_split_candidate :: Union{SplitCandidate,Nothing} # Transient. Needed during tree growing.
end


# Returns path
function save(path, bin_splits, trees)
  trees = map(strip_tree_training_info, trees)
  # model = Model(bin_splits, trees)
  BSON.@save path bin_splits trees
  path
end


# Returns (bin_splits, trees)
function load(path)
  BSON.@load path bin_splits trees
  FeatureType = typeof(bin_splits[1][1])
  (Vector{BinSplits{FeatureType}}(bin_splits), Vector{Tree}(trees))
end


function print_tree(tree :: Tree, level)
  indentation = repeat("    ", level)
  if isa(tree, Node)
    println(indentation * "Node: feature_i $(tree.feature_i)\tsplit_i $(tree.split_i)")
    print_tree(tree.left,  level + 1)
    print_tree(tree.right, level + 1)
  else
    println(indentation * "Leaf: Δscore $(tree.Δscore)\tis_count $(length(tree.is))")
  end
end


# Returns list of feature_i => appearance_count pairs, most common first.
function feature_importance_by_appearance_count(trees :: Vector{Tree})
  feature_i_to_count = Dict{Int64,Int64}()

  for tree in trees
    for split_node in tree_split_nodes(tree)
      feature_i = split_node.feature_i
      feature_i_to_count[feature_i] = 1 + get(feature_i_to_count, feature_i, 0)
    end
  end

  sort(collect(feature_i_to_count), by=(kv -> -kv[2]))
end


# Returns list of feature_i => absolute_Δscore pairs, most important first.
function feature_importance_by_absolute_delta_score(trees :: Vector{Tree})
  feature_i_to_absolute_Δscore = Dict{Int64,Float64}()

  for tree in trees
    for split_node in tree_split_nodes(tree)
      feature_i       = split_node.feature_i
      leaves          = tree_leaves(split_node)
      absolute_Δscore = sum(map(leaf -> abs(leaf.Δscore), leaves))

      feature_i_to_absolute_Δscore[feature_i] = absolute_Δscore + get(feature_i_to_absolute_Δscore, feature_i, 0.0)
    end
  end

  sort(collect(feature_i_to_absolute_Δscore), by=(kv -> -kv[2]))
end


function tree_split_nodes(tree :: Tree) :: Vector{Node}
  if isa(tree, Leaf)
    []
  else
    vcat([tree], tree_split_nodes(tree.left), tree_split_nodes(tree.right))
  end
end

function tree_leaves(tree :: Tree) :: Vector{Leaf}
  if isa(tree, Leaf)
    [tree]
  else
    vcat(tree_leaves(tree.left), tree_leaves(tree.right))
  end
end


function leaf_depth(tree :: Tree, leaf :: Leaf) :: Union{Int64,Nothing}
  if tree === leaf
    1
  elseif isa(tree, Leaf)
    nothing
  else
    left_depth = leaf_depth(tree.left, leaf)
    if left_depth != nothing
      left_depth + 1
    else
      right_depth = leaf_depth(tree.right, leaf)
      right_depth == nothing ? nothing : right_depth + 1
    end
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

# Non-mutating. Returns a new tree with training info removed from the leaves.
function strip_tree_training_info(tree :: Tree) :: Tree
  if isa(tree, Node)
    left  = strip_tree_training_info(tree.left)
    right = strip_tree_training_info(tree.right)
    Node(tree.feature_i, tree.split_i, left, right)
  else
    Leaf(tree.Δscore, nothing, nothing)
  end
end

# Mutates and returns tree.
function scale_leaf_Δscores(tree, learning_rate)
  for leaf in tree_leaves(tree)
    leaf.Δscore *= learning_rate
  end
  tree
end


# Returns vector of untransformed scores (linear, pre-sigmoid).
function apply_tree(X_binned :: Array{UInt8,2}, tree :: Tree) :: Vector{Score}
  scores = zeros(Score, data_count(X_binned))
  apply_tree!(X_binned, tree, scores)
end


# Mutates scores.
function apply_tree!(X_binned :: Array{UInt8,2}, tree :: Tree, scores :: Vector) :: Vector
  for i in 1:data_count(X_binned)
    node = tree
    while !isa(node, Leaf)
      if X_binned[i, node.feature_i] <= node.split_i
        node = node.left
      else
        node = node.right
      end
    end
    scores[i] += node.Δscore
  end

  scores
end


# Returns vector of untransformed scores (linear, pre-sigmoid). Does not mutate starting_scores.
function apply_trees(X_binned :: Array{UInt8,2}, trees :: Vector{Tree}, starting_scores = nothing) :: Vector{Score}

  thread_scores = map(_ -> zeros(Score, data_count(X_binned)), 1:Threads.nthreads())

  Threads.@threads for tree in trees
    apply_tree!(X_binned, tree, thread_scores[Threads.threadid()])
  end

  scores = sum(thread_scores)

  if starting_scores != nothing
    scores += starting_scores
  end

  scores
end


# Returns vector of predictions ŷ (post-sigmoid).
function predict(X, bin_splits, trees; starting_scores = nothing) :: Vector{Prediction}
  X_binned = apply_bins(X, bin_splits)

  predict_on_binned(X_binned, trees, starting_scores = starting_scores)
end

# Returns vector of predictions ŷ (post-sigmoid).
function predict_on_binned(X_binned :: Array{UInt8,2}, trees :: Vector{Tree}; starting_scores = nothing) :: Vector{Prediction}
  scores = apply_trees(X_binned, trees, starting_scores)
  σ.(scores)
end


# Aim for a roughly equal number of data points in each bin.
function prepare_bin_splits(X :: Array{FeatureType,2}, bin_count) :: Vector{BinSplits{FeatureType}} where FeatureType <: AbstractFloat
  if bin_count < 2 || bin_count > 255
    error("prepare_bin_splits: bin_count must be between 2 and 255")
  end
  ideal_sample_count = bin_count * 1_000
  is = sort(collect(Iterators.take(Random.shuffle(1:data_count(X)), ideal_sample_count)))

  sample_count = length(is)
  split_count = bin_count - 1

  bin_splits = Vector{BinSplits{FeatureType}}(undef, feature_count(X))

  Threads.@threads for j in 1:feature_count(X)
    sorted_feature_values = sort(X[is, j])

    splits = zeros(eltype(sorted_feature_values), split_count)

    for split_i in 1:split_count
      split_sample_i = Int64(floor(sample_count / bin_count * split_i))
      value_below_split = sorted_feature_values[split_sample_i]
      value_above_split = sorted_feature_values[min(split_sample_i + 1, sample_count)]
      splits[split_i] = (value_below_split + value_above_split) / 2f0 # Avoid coericing Float32 to Float64
    end

    bin_splits[j] = splits
  end

  bin_splits
end


function apply_bins(X, bin_splits) :: Array{UInt8,2}
  X_binned = zeros(UInt8, size(X))

  Threads.@threads for j in 1:feature_count(X)
    splits_for_feature = bin_splits[j]
    bin_count = length(splits_for_feature) + 1
    for i in 1:data_count(X)
      value   = X[i,j]

      jump_step = div(bin_count - 1, 2)
      split_i   = 1

      # Binary-ish jumping

      # invariant: split_i > 1 implies split_i split <= value
      while jump_step > 0
        while jump_step > 0 && splits_for_feature[split_i + jump_step] > value
          jump_step = div(jump_step, 2)
        end
        split_i += jump_step
        jump_step = div(jump_step, 2)
      end

      bin_i = bin_count
      for k in split_i:length(splits_for_feature)
        if splits_for_feature[k] > value
          bin_i = k
          break
        end
      end

      # split_i = findfirst(split_value -> split_value > value, @view splits_for_feature[split_i:length(splits_for_feature)])
      # bin_i   = split_i == nothing ? bin_count : split_i

      X_binned[i,j] = UInt8(bin_i) # Store as 1-255 to match Julia indexing. We leave 0 unused but saves us from having to remember to convert.
    end
  end

  X_binned
end


function build_one_tree(X_binned :: Array{UInt8,2}, y, ŷ; config...) # y = labels, ŷ = predictions so far
  tree = Leaf(optimal_Δscore(y, ŷ, get_config_field(config, :l2_regularization), get_config_field(config, :max_delta_score)), collect(1:length(y)), nothing)

  features_to_use_count = Int64(ceil(get_config_field(config, :feature_fraction) * feature_count(X_binned)))

  # I suspect the cache benefits for sorting the indexes are trivial but it feels cleaner.
  feature_is = sort(Random.shuffle(1:feature_count(X_binned))[1:features_to_use_count])

  tree_changed = true

  while tree_changed
    # print_tree(tree, 0)
    # println()
    # println()
    (tree_changed, tree) = perhaps_split_tree(tree, X_binned, y, ŷ, feature_is; config...)
  end

  tree
end


# Trains for iteration_count rounds and returns (bin_splits, prior_and_new_trees).
function train(X :: Array{FeatureType,2}, y; bin_splits=nothing, prior_trees=Tree[], config...) :: Tuple{Vector{BinSplits{FeatureType}}, Vector{Tree}} where FeatureType <: AbstractFloat
  if bin_splits == nothing
    print("Preparing bin splits...")
    bin_splits = prepare_bin_splits(X, get_config_field(config, :bin_count))
    println("done.")
  end
  # println(bin_splits)

  print("Binning input data...")
  X_binned = apply_bins(X, bin_splits)
  println("done.")
  # println(X_binned)

  trees = train_on_binned(X_binned, y; prior_trees = prior_trees, config...)

  (bin_splits, trees)
end

function train_on_binned(X_binned :: Array{UInt8,2}, y; prior_trees=Tree[], config...) :: Vector{Tree}
  scores = apply_trees(X_binned, prior_trees) # Linear scores, before sigmoid transform.

  trees = []

  for iteration_i in 1:get_config_field(config, :iteration_count)
    (scores, tree) = train_one_iteration(X_binned, y, scores; config...)

    ŷ = σ.(scores)
    iteration_loss = sum(logloss.(y, ŷ)) / length(y)
    # println(ŷ)
    println("Iteration $iteration_i training loss: $iteration_loss")

    print_tree(tree, 0)
    println()

    push!(trees, strip_tree_training_info(tree)) # For long boosting sessions, saves memory if we strip off the list of indices
  end

  vcat(prior_trees, trees)
end

# Returns (new_scores, tree)
function train_one_iteration(X_binned, y, scores; config...)
  ŷ = σ.(scores)
  # println(ŷ)
  tree = build_one_tree(X_binned, y, ŷ; config...)
  tree = scale_leaf_Δscores(tree, get_config_field(config, :learning_rate))
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
function optimal_Δscore(y, ŷ, l2_regularization, max_delta_score)
  Σ∇loss  = sum(∇logloss.(y, ŷ))
  Σ∇∇loss = sum(∇∇logloss.(ŷ))

  # And the loss minima is at:
  clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)
end


# -0.5 * (Σ∇loss)² / (Σ∇∇loss) in XGBoost paper; but can't simplify so much if clamping the score.
function leaf_expected_Δloss(Σ∇loss, Σ∇∇loss, l2_regularization, max_delta_score)
  Δscore = clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)

  Σ∇loss * Δscore + 0.5 * Σ∇∇loss * Δscore * Δscore
end


# Mutates tree, but also returns the tree in case tree was a lone leaf.
#
# Returns (bool, tree) where bool is true if any split was made, otherwise false.
function perhaps_split_tree(tree, X_binned :: Array{UInt8,2}, y, ŷ, feature_is; config...)
  leaves = tree_leaves(tree)

  if length(leaves) >= get_config_field(config, :max_leaves)
    return (false, tree)
  end

  min_data_weight_in_leaf = get_config_field(config, :min_data_weight_in_leaf)
  l2_regularization       = get_config_field(config, :l2_regularization)
  max_delta_score         = get_config_field(config, :max_delta_score)

  dont_split = SplitCandidate(0.0, 0, 0)

  # Ensure split_candidate on all leaves
  for leaf in leaves
    if leaf.maybe_split_candidate == nothing && leaf_depth(tree, leaf) >= get_config_field(config, :max_depth)
      leaf.maybe_split_candidate = dont_split
    elseif leaf.maybe_split_candidate == nothing
      # Find best feature and split
      # Expected Δlogloss at leaf = -0.5 * (Σ ∇loss)² / (Σ ∇∇loss)

      # best_expected_Δloss, best_feature_i, best_split_i
      thread_bests = map(_ -> (0.0, 0, UInt8(0)), 1:Threads.nthreads())

      Threads.@threads for feature_i in feature_is
        best_expected_Δloss, best_feature_i, best_split_i = thread_bests[Threads.threadid()]

        max_bins = maximum(@view X_binned[:,feature_i])

        hist_bins = map(bin_i -> HistBin(), 1:max_bins)

        this_leaf_Σ∇loss      = 0.0
        this_leaf_Σ∇∇loss     = 0.0
        this_leaf_data_weight = 0.0

        for i in leaf.is
          bin_i    = X_binned[i, feature_i]
          hist_bin = hist_bins[bin_i]

          data_point_weight = 1.0

          ∇loss  = ∇logloss(y[i], ŷ[i])
          ∇∇loss = ∇∇logloss(ŷ[i])

          hist_bin.Σ∇loss      += ∇loss
          hist_bin.Σ∇∇loss     += ∇∇loss
          hist_bin.data_weight += data_point_weight

          this_leaf_Σ∇loss      += ∇loss
          this_leaf_Σ∇∇loss     += ∇∇loss
          this_leaf_data_weight += data_point_weight
        end

        this_leaf_expected_Δloss = leaf_expected_Δloss(this_leaf_Σ∇loss, this_leaf_Σ∇∇loss, l2_regularization, max_delta_score)

        left_Σ∇loss      = 0.0
        left_Σ∇∇loss     = 0.0
        left_data_weight = 0.0

        for bin_i in UInt8(1):UInt8(max_bins-1)
          hist_bin = hist_bins[bin_i]

          left_Σ∇loss      += hist_bin.Σ∇loss
          left_Σ∇∇loss     += hist_bin.Σ∇∇loss
          left_data_weight += hist_bin.data_weight

          if left_data_weight < min_data_weight_in_leaf
            continue
          end

          right_Σ∇loss      = this_leaf_Σ∇loss  - left_Σ∇loss
          right_Σ∇∇loss     = this_leaf_Σ∇∇loss - left_Σ∇∇loss
          right_data_weight = this_leaf_data_weight - left_data_weight

          if right_data_weight < min_data_weight_in_leaf
            break
          end

          expected_Δloss =
            -this_leaf_expected_Δloss +
            leaf_expected_Δloss(left_Σ∇loss,  left_Σ∇∇loss,  l2_regularization, max_delta_score) +
            leaf_expected_Δloss(right_Σ∇loss, right_Σ∇∇loss, l2_regularization, max_delta_score)

          if expected_Δloss < best_expected_Δloss
            best_expected_Δloss = expected_Δloss
            best_feature_i      = feature_i
            best_split_i        = bin_i

            thread_bests[Threads.threadid()] = (best_expected_Δloss, best_feature_i, best_split_i)
          end
        end # for bin_i in 1:(max_bins-1)
      end # for feature_i in feature_is

      best_expected_Δloss, best_feature_i, best_split_i = minimum(thread_bests)

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

    left_Δscore  = optimal_Δscore(left_ys,  left_ŷs,  l2_regularization, max_delta_score)
    right_Δscore = optimal_Δscore(right_ys, right_ŷs, l2_regularization, max_delta_score)

    left_leaf  = Leaf(left_Δscore,  left_is,  nothing)
    right_leaf = Leaf(right_Δscore, right_is, nothing)

    new_node = Node(feature_i, split_i, left_leaf, right_leaf)

    tree = replace_leaf(tree, leaf_to_split, new_node)

    (true, tree)
  else
    (false, tree)
  end
end

end # module MagicTreeBoosting