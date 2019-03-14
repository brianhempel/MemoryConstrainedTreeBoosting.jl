module MemoryConstrainedTreeBoosting

export train, train_on_binned, prepare_bin_splits, bin_and_compress, finalize_loading, apply_bins, predict, predict_on_binned, save, load

import Random
import Statistics

import BSON
import TranscodingStreams, CodecZstd


default_config = (
  weights                 = nothing, # weights for the data
  bin_count               = 255,
  iteration_count         = 10,
  min_data_weight_in_leaf = 10.0,
  l2_regularization       = 1.0,
  max_leaves              = 32,
  max_depth               = 6,
  max_delta_score         = 1.0e10, # Before shrinkage.
  learning_rate           = 0.1,
  feature_fraction        = 1.0, # Per tree.
  bagging_temperature     = 1.0, # Same as Catboost's Bayesian bagging. 0.0 doesn't change the weights. 1.0 samples from the exponential distribution to scale each datapoint's weight.
  random_strength         = 0.0, # Inspired by Catboost. Relative standard deviation of noise added to the expected loss improvement of each split. Affects split randomization. Decays throughout training. If using, start with a value of 0.01 - 0.2.
  feature_i_to_name       = nothing,
  iteration_callback      = trees -> (),
)


function get_config_field(config, key)
  if !isa(config, Tuple{}) && haskey(config, key)
    config[key]
  else
    default_config[key]
  end
end

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0

const Score      = Float32
const Loss       = Float32
const Prediction = Float32
const DataWeight = Float32

const BinSplits = Vector{T} where T <: AbstractFloat

mutable struct CompressedData
  data_count          :: Int64
  compressed_features :: Vector{Vector{UInt8}}
end

# FeatureBeingCompressed is used while data is still being prepared. Replaced by CompressedData (via finalize_loading) before training.
mutable struct FeatureBeingCompressed
  last_bin_i          :: UInt8
  buffer              :: IOBuffer
  compression_stream  :: CodecZstd.ZstdCompressorStream

  FeatureBeingCompressed(buffer, compression_stream) = new(0x00, buffer, compression_stream)
end

mutable struct DataBeingCompressed
  data_count                :: Int64
  features_being_compressed :: Vector{FeatureBeingCompressed}
end

const Data                  = Union{Array{UInt8,2},CompressedData}
const DataOrDataBeingLoaded = Union{Data,DataBeingCompressed}
# const Data = Array{UInt8,2}

data_count(X :: Array{<:Number,2}) = size(X,1)
data_count(X :: CompressedData)    = X.data_count

feature_count(X :: Array{<:Number,2}) = size(X,2)
feature_count(X :: CompressedData)    = length(X.compressed_features)

get_feature(X_binned :: Array{UInt8,2}, feature_i) = @view X_binned[:, feature_i]
get_feature(X_binned :: CompressedData, feature_i) = begin
  out = TranscodingStreams.transcode(CodecZstd.ZstdDecompressor, X_binned.compressed_features[feature_i]) # Decompress the whole feature and return it.
  @inbounds for i in 2:data_count(X_binned)
    out[i] += out[i-1] # Undo the delta encoding.
  end
  out
end


abstract type Tree end

mutable struct SplitCandidate
  expected_Δloss :: Loss
  feature_i      :: Int64
  split_i        :: UInt8
end

const dont_split = SplitCandidate(0.0f0, 0, 0x00)

mutable struct Histogram
  Σ∇losses     :: Vector{Loss}
  Σ∇∇losses    :: Vector{Loss}
  data_weights :: Vector{DataWeight}

  Histogram(bin_count) = new(zeros(Loss, bin_count), zeros(Loss, bin_count), zeros(DataWeight, bin_count))
end

mutable struct Node <: Tree
  feature_i           :: Int64
  split_i             :: UInt8
  left                :: Tree
  right               :: Tree
  features_histograms :: Vector{Histogram} # Transient. Used to speed up tree calculation.
end

mutable struct Leaf <: Tree
  Δscore                :: Score # Called "weight" in the literature
  is                    :: Union{Vector{Int64},Nothing}  # Transient. Needed during tree growing.
  maybe_split_candidate :: Union{SplitCandidate,Nothing} # Transient. Needed during tree growing.
  features_histograms   :: Vector{Histogram}             # Transient. Used to speed up tree calculation.

  Leaf(Δscore, is = nothing, maybe_split_candidate = nothing, features_histograms = []) = new(Δscore, is, maybe_split_candidate, features_histograms)
end


# Returns path
function save(path, bin_splits, trees)
  trees = map(strip_tree_training_info, trees)
  BSON.@save path bin_splits trees
  path
end


# Returns (bin_splits, trees)
function load(path)
  BSON.@load path bin_splits trees
  FeatureType = typeof(bin_splits[1][1])
  (Vector{BinSplits{FeatureType}}(bin_splits), Vector{Tree}(trees))
end


function print_tree(tree, level = 0; feature_i_to_name = nothing)
  indentation = repeat("    ", level)
  if isa(tree, Node)
    feature_name =
      if feature_i_to_name != nothing
        feature_i_to_name(tree.feature_i)
      else
        "feature $(tree.feature_i)"
      end
    println(indentation * "$feature_name\tsplit at $(tree.split_i)")
    print_tree(tree.left,  level + 1, feature_i_to_name = feature_i_to_name)
    print_tree(tree.right, level + 1, feature_i_to_name = feature_i_to_name)
  else
    println(indentation * "Δscore $(tree.Δscore)\t$(length(tree.is)) datapoints")
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
  feature_i_to_absolute_Δscore = Dict{Int64,Score}()

  for tree in trees
    for split_node in tree_split_nodes(tree)
      feature_i       = split_node.feature_i
      leaves          = tree_leaves(split_node)
      absolute_Δscore = sum(map(leaf -> abs(leaf.Δscore), leaves))

      feature_i_to_absolute_Δscore[feature_i] = absolute_Δscore + get(feature_i_to_absolute_Δscore, feature_i, 0.0f0)
    end
  end

  sort(collect(feature_i_to_absolute_Δscore), by=(kv -> -kv[2]))
end


function tree_split_nodes(tree) :: Vector{Node}
  if isa(tree, Leaf)
    []
  else
    vcat([tree], tree_split_nodes(tree.left), tree_split_nodes(tree.right))
  end
end

function tree_leaves(tree) :: Vector{Leaf}
  if isa(tree, Leaf)
    [tree]
  else
    vcat(tree_leaves(tree.left), tree_leaves(tree.right))
  end
end

function parent_node(tree, target) :: Union{Node,Nothing}
  if isa(tree, Leaf)
    nothing
  elseif tree.left === target
    tree
  elseif tree.right === target
    tree
  else
    parent_in_left = parent_node(tree.left, target)
    if parent_in_left != nothing
      parent_in_left
    else
      parent_node(tree.right, target)
    end
  end
end


function leaf_depth(tree, leaf) :: Union{Int64,Nothing}
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
function strip_tree_training_info(tree) :: Tree
  if isa(tree, Node)
    left  = strip_tree_training_info(tree.left)
    right = strip_tree_training_info(tree.right)
    Node(tree.feature_i, tree.split_i, left, right, [])
  else
    Leaf(tree.Δscore, nothing, nothing, [])
  end
end

# Mutates and returns tree.
function scale_leaf_Δscores(tree, learning_rate) :: Tree
  for leaf in tree_leaves(tree)
    leaf.Δscore *= learning_rate
  end
  tree
end


# Returns vector of untransformed scores (linear, pre-sigmoid).
function apply_tree(X_binned :: Data, tree :: Tree) :: Vector{Score}
  scores = zeros(Score, data_count(X_binned))
  apply_tree!(X_binned, tree, scores)
end

# Mutates scores.
function apply_tree!(X_binned :: Array{UInt8,2}, tree :: Tree, scores :: Vector{Score}) :: Vector{Score}
  Threads.@threads for i in 1:data_count(X_binned)
  # for i in 1:data_count(X_binned)
    @inbounds begin
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
  end

  scores
end
# Mutates scores.
function apply_tree!(X_binned :: CompressedData, tree :: Tree, scores :: Vector{Score}) :: Vector{Score}
  feature_is = unique(map(node -> node.feature_i, tree_split_nodes(tree)))
  features_decompressed = Vector{Union{Nothing,AbstractArray}}(nothing, feature_count(X_binned))

  Threads.@threads for feature_i in feature_is
    features_decompressed[feature_i] = get_feature(X_binned, feature_i)
  end

  # No threading yet.
  Threads.@threads for i in 1:data_count(X_binned)
    @inbounds begin
      node = tree
      while !isa(node, Leaf)
        feature_i = node.feature_i
        if features_decompressed[feature_i][i] <= node.split_i
          node = node.left
        else
          node = node.right
        end
      end
      scores[i] += node.Δscore
    end
  end

  scores
end


# Returns vector of untransformed scores (linear, pre-sigmoid). Does not mutate starting_scores.
function apply_trees(X_binned :: Data, trees :: Vector{<:Tree}, starting_scores = nothing) :: Vector{Score}

  # thread_scores = map(_ -> zeros(Score, data_count(X_binned)), 1:Threads.nthreads())
  scores = zeros(Score, data_count(X_binned))

  # Threads.@threads for tree in trees
  for tree in trees
    # apply_tree!(X_binned, tree, thread_scores[Threads.threadid()])
    apply_tree!(X_binned, tree, scores)
  end

  # scores = sum(thread_scores)

  if starting_scores != nothing
    scores += starting_scores
  end

  scores
end


# Returns vector of predictions ŷ (post-sigmoid).
function predict(X, bin_splits, trees; starting_scores = nothing, output_raw_scores = false) :: Vector{Prediction}
  X_binned = apply_bins(X, bin_splits)

  predict_on_binned(X_binned, trees, starting_scores = starting_scores, output_raw_scores = output_raw_scores)
end

# Returns vector of predictions ŷ (post-sigmoid).
function predict_on_binned(X_binned :: Data, trees :: Vector{<:Tree}; starting_scores = nothing, output_raw_scores = false) :: Vector{Prediction}
  scores = apply_trees(X_binned, trees, starting_scores)
  if output_raw_scores
    scores
  else
    σ.(scores)
  end
end


# Aim for a roughly equal number of data points in each bin.
# Does not support weights.
function prepare_bin_splits(X :: Array{FeatureType,2}, bin_count = 255) :: Vector{BinSplits{FeatureType}} where FeatureType <: AbstractFloat
  if bin_count < 2 || bin_count > 255
    error("prepare_bin_splits: bin_count must be between 2 and 255")
  end
  ideal_sample_count = bin_count * 1_000
  is = sort(collect(Iterators.take(Random.shuffle(1:data_count(X)), ideal_sample_count)))

  sample_count = length(is)
  split_count = bin_count - 1

  bin_splits = Vector{BinSplits{FeatureType}}(undef, feature_count(X))

  Threads.@threads for j in 1:feature_count(X)
  # for j in 1:feature_count(X)
    sorted_feature_values = sort(X[is, j])

    splits = zeros(eltype(sorted_feature_values), split_count)

    for split_i in 1:split_count
      split_sample_i = max(1, Int64(floor(sample_count / bin_count * split_i)))
      value_below_split = sorted_feature_values[split_sample_i]
      value_above_split = sorted_feature_values[min(split_sample_i + 1, sample_count)]
      splits[split_i] = (value_below_split + value_above_split) / 2f0 # Avoid coercing Float32 to Float64
    end

    bin_splits[j] = splits
  end

  bin_splits
end


function apply_bins(X, bin_splits) :: Array{UInt8,2}
  X_binned = zeros(UInt8, size(X))

  Threads.@threads for j in 1:feature_count(X)
  # for j in 1:feature_count(X)
    splits_for_feature = bin_splits[j]
    bin_count = length(splits_for_feature) + 1
    @inbounds for i in 1:data_count(X)
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

function bin_and_compress(X, bin_splits; prior_data = nothing) :: DataBeingCompressed
  X_binned = apply_bins(X, bin_splits)

  if prior_data == nothing
    features_being_compressed =
      map(1:feature_count(X)) do feature_i
        buffer             = IOBuffer()
        compression_stream = CodecZstd.ZstdCompressorStream(buffer)
        FeatureBeingCompressed(buffer, compression_stream)
      end

    data_being_compressed = DataBeingCompressed(0, features_being_compressed)
  else
    features_being_compressed = prior_data.features_being_compressed
    data_being_compressed     = prior_data
  end

  Threads.@threads for feature_i in 1:feature_count(X)
    # for feature_i in 1:feature_count(X)
    feature_being_compressed = features_being_compressed[feature_i]
    last_bin_i               = feature_being_compressed.last_bin_i
    chunk_to_compress        = Vector{UInt8}(undef, data_count(X))
    @inbounds for i in 1:data_count(X)
      chunk_to_compress[i] = X_binned[i, feature_i] - last_bin_i
      last_bin_i           = X_binned[i, feature_i]
    end
    feature_being_compressed.last_bin_i = last_bin_i
    write(feature_being_compressed.compression_stream, chunk_to_compress)
    flush(feature_being_compressed.compression_stream)
  end

  data_being_compressed.data_count += data_count(X)

  data_being_compressed
end

# If training on compressed features, readout and close the compression streams.
#
# This should be done by the client. If we did it automatically, the client's GC will retain the DataBeingCompressed structs. (Which...might not be a problem depending on how the internal buffers work.)
function finalize_loading(data_being_compressed :: DataBeingCompressed) :: CompressedData
  compressed_features = map(data_being_compressed.features_being_compressed) do feature_being_compressed
    write(feature_being_compressed.compression_stream, TranscodingStreams.TOKEN_END)
    flush(feature_being_compressed.compression_stream)
    compressed_feature = take!(feature_being_compressed.buffer)[:]
    close(feature_being_compressed.compression_stream)
    compressed_feature
  end

  CompressedData(data_being_compressed.data_count, compressed_features)
end

function finalize_loading(X_binned :: Data)
  X_binned
end

function compression_ratios(compressed_data :: CompressedData)
  map(compressed_data.compressed_features) do compressed_feature
    compressed_data.data_count / length(compressed_feature)
  end
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

function train_on_binned(X_binned :: Data, y; prior_trees=Tree[], config...) :: Vector{Tree}
  scores = apply_trees(X_binned, prior_trees) # Linear scores, before sigmoid transform.

  trees = copy(prior_trees)

  weights = get_config_field(config, :weights)
  if weights == nothing
    weights = fill(DataWeight(1.0), length(y))
  end

  for iteration_i in 1:get_config_field(config, :iteration_count)
    # @time begin
    begin
      (scores, tree) = train_one_iteration(X_binned, y, weights, scores, length(trees); config...)

      ŷ = σ.(scores)
      iteration_loss = sum(logloss.(y, ŷ) .* weights) / sum(weights)
      # println(ŷ)
      # println("Iteration $iteration_i training loss: $iteration_loss")

      # print_tree(tree; feature_i_to_name = get_config_field(config, :feature_i_to_name))
      # println()

      push!(trees, strip_tree_training_info(tree)) # For long boosting sessions, should save memory if we strip off the list of indices

      get_config_field(config, :iteration_callback)(trees)
    end
  end

  trees
end


# Returns (new_scores, tree)
function train_one_iteration(X_binned, y :: Vector{Prediction}, weights :: Vector{DataWeight}, scores :: Vector{Score}, prior_tree_count; config...) :: Tuple{Vector{Score}, Tree}
  ŷ = σ.(scores)
  # println(ŷ)

  learning_rate       = Score(get_config_field(config, :learning_rate))
  bagging_temperature = DataWeight(get_config_field(config, :bagging_temperature))
  random_strength     = Loss(get_config_field(config, :random_strength))

  weights =
    # Adapted from Catboost
    if bagging_temperature > 0.0f0
      map(weights) do weight
        r = -log(rand(Float32) + ε)
        if bagging_temperature != 1.0f0
          weight * r^bagging_temperature
        else
          weight * r
        end
      end
    else
      weights
    end

  split_expected_Δloss_noise_std_dev = begin
    if random_strength == 0.0f0
      0.0f0
    else
      l2_regularization = Loss(get_config_field(config, :l2_regularization))
      max_delta_score   = Score(get_config_field(config, :max_delta_score))

      # Not how Catboost does it, but this seems the most reasonable baseline given how we calculate improvements.

      Δloss_if_each_datapoint_got_its_own_leaf = 0.0f0
      @inbounds for i in 1:data_count(X_binned)
        ∇loss  = ∇logloss(y[i], ŷ[i]) * weights[i]
        ∇∇loss = ∇∇logloss(ŷ[i])      * weights[i]

        Δloss_if_each_datapoint_got_its_own_leaf += leaf_expected_Δloss(∇loss, ∇∇loss, l2_regularization / data_count(X_binned), max_delta_score)
      end

      # # This decay schedule is from Catboost, https://github.com/catboost/catboost/blob/master/catboost/cuda/methods/random_score_helper.h#L18-L22
      # decay_multiplier = begin
      #   model_size     = prior_tree_count * learning_rate
      #   log_data_count = log(data_count(X_binned))
      #   model_left     = exp(log_data_count - model_size)
      #
      #   model_left / (1.0 + model_left)
      # end

      # Catboost's decay schedule doesn't make sense. We start overfitting at a model size of 5-10, not at a model size of log(sample_count).
      decay_multiplier = begin
        model_size = prior_tree_count * learning_rate
        0.01f0^(model_size / 5.0f0)
      end

      Δloss_if_each_datapoint_got_its_own_leaf * decay_multiplier * random_strength
    end
  end

  tree = build_one_tree(X_binned, y, ŷ, weights, split_expected_Δloss_noise_std_dev; config...)
  tree = scale_leaf_Δscores(tree, learning_rate)
  new_scores = copy(scores)
  apply_tree!(X_binned, tree, new_scores)
  (new_scores, tree)
end


function build_one_tree(X_binned :: Data, y, ŷ, weights, split_expected_Δloss_noise_std_dev; config...) # y = labels, ŷ = predictions so far
  tree = Leaf(optimal_Δscore(y, ŷ, weights, Loss(get_config_field(config, :l2_regularization)), Score(get_config_field(config, :max_delta_score))), collect(1:length(y)), nothing, [])

  features_to_use_count = Int64(ceil(get_config_field(config, :feature_fraction) * feature_count(X_binned)))

  # I suspect the cache benefits for sorting the indexes are trivial but it feels cleaner.
  feature_is = sort(Random.shuffle(1:feature_count(X_binned))[1:features_to_use_count])

  tree_changed = true

  while tree_changed
    # print_tree(tree)
    # println()
    # println()
    (tree_changed, tree) = perhaps_split_tree(tree, X_binned, y, ŷ, weights, feature_is, split_expected_Δloss_noise_std_dev; config...)
  end

  tree
end


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

# Copied from Flux.jl.
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

# Derivatives with respect to margin (i.e. pre-sigmoid values, but you still provide the post-sigmoid probability).
#
# It's a bit of math but it works out nicely. (Verified against XGBoost.)
∇logloss(y, ŷ) = ŷ - y
∇∇logloss(ŷ)   = ŷ * (1.0f0 - ŷ) # Interestingly, not dependent on y. XGBoost adds an ε term


# Assuming binary classification with log loss.
function optimal_Δscore(y :: AbstractArray{Prediction}, ŷ :: AbstractArray{Prediction}, weights :: AbstractArray{DataWeight}, l2_regularization :: Loss, max_delta_score :: Score)
  Σ∇loss  = sum(∇logloss.(y, ŷ) .* weights)
  Σ∇∇loss = sum(∇∇logloss.(ŷ)   .* weights)

  # And the loss minima is at:
  clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)
end


# -0.5 * (Σ∇loss)² / (Σ∇∇loss) in XGBoost paper; but can't simplify so much if clamping the score.
function leaf_expected_Δloss(Σ∇loss :: Loss, Σ∇∇loss :: Loss, l2_regularization :: Loss, max_delta_score :: Score)
  Δscore = clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)

  Σ∇loss * Δscore + 0.5f0 * Σ∇∇loss * Δscore * Δscore
end


# Mutates tree, but also returns the tree in case tree was a lone leaf.
#
# Returns (bool, tree) where bool is true if any split was made, otherwise false.
function perhaps_split_tree(tree, X_binned :: Data, y, ŷ, weights, feature_is, split_expected_Δloss_noise_std_dev; config...)
  leaves = sort(tree_leaves(tree), by = (leaf -> length(leaf.is))) # Process smallest leaves first, should speed up histogram computation.

  max_bins = Int64(typemax(UInt8))

  if length(leaves) >= get_config_field(config, :max_leaves)
    return (false, tree)
  end

  min_data_weight_in_leaf = DataWeight(get_config_field(config, :min_data_weight_in_leaf))
  l2_regularization       = Loss(get_config_field(config, :l2_regularization))
  max_delta_score         = Score(get_config_field(config, :max_delta_score))

  # Ensure split_candidate on all leaves
  for leaf in leaves
    if leaf.maybe_split_candidate == nothing && leaf_depth(tree, leaf) >= get_config_field(config, :max_depth)
      leaf.maybe_split_candidate = dont_split
    elseif leaf.maybe_split_candidate == nothing
      # Find best feature and best split
      # Expected Δlogloss at leaf = -0.5 * (Σ ∇loss)² / (Σ ∇∇loss)

      if isempty(leaf.features_histograms)
        leaf.features_histograms = map(_ -> Histogram(max_bins), 1:feature_count(X_binned))
      end

      Threads.@threads for feature_i in feature_is
      # for feature_i in feature_is
        calculate_feature_histogram!(X_binned, y, ŷ, weights, feature_i, tree, leaf)
      end

      leaf.maybe_split_candidate = find_best_split(leaf.features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score, split_expected_Δloss_noise_std_dev)
    end # if leaf.maybe_split_candidate == nothing
  end # for leaf in leaves

  # Expand best split_candiate (if any)

  (expected_Δloss, leaf_i_to_split) = findmin(map(leaf -> leaf.maybe_split_candidate.expected_Δloss, leaves))

  expected_Δloss :: Loss

  if expected_Δloss < 0.0f0
    # We have a usable split!

    leaf_to_split = leaves[leaf_i_to_split]

    feature_i = leaf_to_split.maybe_split_candidate.feature_i
    split_i   = leaf_to_split.maybe_split_candidate.split_i

    # A bit of recalculation below but should be fine.

    feature_binned = get_feature(X_binned, feature_i)

    # Can't seem to get it faster than this version.
    left_is  = filter(i -> feature_binned[i] <= split_i, leaf_to_split.is)
    right_is = filter(i -> feature_binned[i] >  split_i, leaf_to_split.is)

    left_ys       = @view y[left_is]
    left_ŷs       = @view ŷ[left_is]
    right_ys      = @view y[right_is]
    right_ŷs      = @view ŷ[right_is]
    left_weights  = @view weights[left_is]
    right_weights = @view weights[right_is]

    left_Δscore  = optimal_Δscore(left_ys,  left_ŷs,  left_weights,  l2_regularization, max_delta_score)
    right_Δscore = optimal_Δscore(right_ys, right_ŷs, right_weights, l2_regularization, max_delta_score)

    left_leaf  = Leaf(left_Δscore,  left_is,  nothing, [])
    right_leaf = Leaf(right_Δscore, right_is, nothing, [])

    new_node = Node(feature_i, split_i, left_leaf, right_leaf, leaf_to_split.features_histograms)

    tree = replace_leaf(tree, leaf_to_split, new_node)

    (true, tree)
  else
    (false, tree)
  end
end

# Calculates and sets leaf.features_histograms[feature_i]
function calculate_feature_histogram!(X_binned :: Data, y, ŷ, weights, feature_i, tree, leaf)

  max_bins = Int64(typemax(UInt8))

  histogram = leaf.features_histograms[feature_i]

  # If the parent cached its histogram, and the other sibling has already done its calculation, then we can calculate our histogram by simple subtraction.
  parent = parent_node(tree, leaf)
  if parent != nothing
    sibling = (parent.left === leaf ? parent.right : parent.left)
  end

  if parent != nothing && !isempty(parent.features_histograms) && !isempty(sibling.features_histograms)
    # Expediated histogram calculation.
    parent_histogram  = parent.features_histograms[feature_i]
    sibling_histogram = sibling.features_histograms[feature_i]

    histogram.Σ∇losses     .= parent_histogram.Σ∇losses     .- sibling_histogram.Σ∇losses
    histogram.Σ∇∇losses    .= parent_histogram.Σ∇∇losses    .- sibling_histogram.Σ∇∇losses
    histogram.data_weights .= parent_histogram.data_weights .- sibling_histogram.data_weights
  else
    # Can't expediate hist_bin calculation.
    feature_binned = get_feature(X_binned, feature_i)

    Σ∇losses     = histogram.Σ∇losses
    Σ∇∇losses    = histogram.Σ∇∇losses
    data_weights = histogram.data_weights

    leaf_is = leaf.is

    @inbounds for ii in 1:3:(length(leaf.is)-2)
      i1 = leaf_is[ii]
      i2 = leaf_is[ii+1]
      i3 = leaf_is[ii+2]
      # i4 = leaf.is[ii+3]
    # for i in leaf.is # this version is almost twice as slow. Doesn't make sense.
      bin_i1 = feature_binned[i1]
      bin_i2 = feature_binned[i2]
      bin_i3 = feature_binned[i3]
      # bin_i4 = feature_binned[i4]

      data_point_weight1 = weights[i1]
      data_point_weight2 = weights[i2]
      data_point_weight3 = weights[i3]
      # data_point_weight4 = weights[i4]

      ∇loss1  = ∇logloss(y[i1], ŷ[i1]) * data_point_weight1
      ∇loss2  = ∇logloss(y[i2], ŷ[i2]) * data_point_weight2
      ∇loss3  = ∇logloss(y[i3], ŷ[i3]) * data_point_weight3
      # ∇loss4  = ∇logloss(y[i4], ŷ[i4]) * data_point_weight4
      ∇∇loss1 = ∇∇logloss(ŷ[i1])       * data_point_weight1
      ∇∇loss2 = ∇∇logloss(ŷ[i2])       * data_point_weight2
      ∇∇loss3 = ∇∇logloss(ŷ[i3])       * data_point_weight3
      # ∇∇loss4 = ∇∇logloss(ŷ[i4])       * data_point_weight4

      Σ∇losses[bin_i1]     += ∇loss1
      Σ∇losses[bin_i2]     += ∇loss2
      Σ∇losses[bin_i3]     += ∇loss3
      # Σ∇losses[bin_i4]     += ∇loss4
      Σ∇∇losses[bin_i1]    += ∇∇loss1
      Σ∇∇losses[bin_i2]    += ∇∇loss2
      Σ∇∇losses[bin_i3]    += ∇∇loss3
      # Σ∇∇losses[bin_i4]    += ∇∇loss4
      data_weights[bin_i1] += data_point_weight1
      data_weights[bin_i2] += data_point_weight2
      data_weights[bin_i3] += data_point_weight3
      # data_weights[bin_i4] += data_point_weight4
    end

    @inbounds for ii in ((1:3:(length(leaf.is)-2)).stop + 3):length(leaf.is)
      i = leaf_is[ii]
      bin_i = feature_binned[i]

      data_point_weight = weights[i]

      ∇loss  = ∇logloss(y[i], ŷ[i]) * data_point_weight
      ∇∇loss = ∇∇logloss(ŷ[i])      * data_point_weight

      Σ∇losses[bin_i]     += ∇loss
      Σ∇∇losses[bin_i]    += ∇∇loss
      data_weights[bin_i] += data_point_weight
    end
  end

  ()
end

# Returns SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)
function find_best_split(features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score, split_expected_Δloss_noise_std_dev)
  # Shouldn't be a slowdown if using fewer bins: the loop already aborts when there's not enough data on the other side of a split.
  max_bins = Int64(typemax(UInt8))

  best_expected_Δloss, best_feature_i, best_split_i = (Loss(0.0), 0, UInt8(0))

  # This should be fast enough that threading won't help, because it's only O(max_bins*feature_count).

  for feature_i in feature_is
    histogram = features_histograms[feature_i]

    Σ∇losses     = histogram.Σ∇losses
    Σ∇∇losses    = histogram.Σ∇∇losses
    data_weights = histogram.data_weights

    this_leaf_Σ∇loss      = sum(Σ∇losses)
    this_leaf_Σ∇∇loss     = sum(Σ∇∇losses)
    this_leaf_data_weight = sum(data_weights)

    this_leaf_expected_Δloss = leaf_expected_Δloss(this_leaf_Σ∇loss, this_leaf_Σ∇∇loss, l2_regularization, max_delta_score)

    left_Σ∇loss      = 0.0f0
    left_Σ∇∇loss     = 0.0f0
    left_data_weight = 0.0f0

    @inbounds for bin_i in UInt8(1):UInt8(max_bins-1)
      left_Σ∇loss      += Σ∇losses[bin_i]
      left_Σ∇∇loss     += Σ∇∇losses[bin_i]
      left_data_weight += data_weights[bin_i]

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

      expected_Δloss += randn(Loss) * split_expected_Δloss_noise_std_dev

      if expected_Δloss < best_expected_Δloss
        best_expected_Δloss = expected_Δloss
        best_feature_i      = feature_i
        best_split_i        = bin_i
      end
    end # for bin_i in 1:(max_bins-1)
  end # for feature_i in feature_is

  SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)
end

end # module MemoryConstrainedTreeBoosting