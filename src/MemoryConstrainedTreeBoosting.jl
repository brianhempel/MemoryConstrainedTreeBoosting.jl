module MemoryConstrainedTreeBoosting

export train, train_on_binned, prepare_bin_splits, apply_bins, predict, predict_on_binned, save, load, load_unbinned_predictor, make_callback_to_track_validation_loss

import Random

import BSON
import SIMD

using MPI


# f should be a function that take an indices_range and returns a tuple of reduction values
#
# parallel_iterate will unzip those tuples into a tuple of arrays of reduction values and return that.
function parallel_iterate(f, count)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * count, Threads.nthreads()) + 1
    stop  = div( thread_i    * count, Threads.nthreads())
    thread_results[thread_i] = f(start:stop)
  end

  if isa(thread_results[1], Tuple)
    # Mangling so you get a tuple of arrays.
    Tuple(collect.(zip(thread_results...)))
  else
    thread_results
  end
end

# f should be a function that take an element of chunks and returns nothing
function parallel_iterate_work_stealing(f, chunks)
  chunk_ii = Threads.Atomic{Int64}(1)

  Threads.@threads for thread_i in 1:Threads.nthreads()
    while (my_chunk_ii = Threads.atomic_add!(chunk_ii, 1)) <= length(chunks)
      f(chunks[my_chunk_ii])
    end
  end

  ()
end


function parallel_map!(f, out, in)
  @assert length(out) == length(in)
  Threads.@threads for i in 1:length(in)
    @inbounds out[i] = f(in[i])
  end
  out
end

# Provide pre-allocated trues and falses arrays.
#
# Fast, at the cost of up to 2x max memory usage compared to growing the arrays dynamically.
#
# Returns views into the pre-allocated trues and falses arrays.
function partition!(f, trues, falses, in)
  true_i  = 1
  false_i = 1

  @inbounds for i in 1:length(in)
    elem = in[i]
    if f(elem)
      trues[true_i] = elem
      true_i += 1
    else
      falses[false_i] = elem
      false_i += 1
    end
  end

  ( view(trues,  1:(true_i-1))
  , view(falses, 1:(false_i-1))
  )
end

# Provide pre-allocated trues and falses arrays for scratch memory.
#
# Fast, at the cost of 2x max memory usage compared to growing the arrays dynamically non-parallel.
#
# Returned arrays are views into the out array.
# The original array is essentially sorted by the
# truthiness of its values and we return views into
# its first part (trues) and second part (falses).
#
# Both out and in can be the same array.
#
# Each thread partitions one chunk of the array.
# Then, based on the size of the partition chunks,
# each thread copies its chunk into an appropriate
# position in the out array.
function parallel_partition!(f, out :: AbstractArray{T}, trues :: AbstractArray{T}, falses :: AbstractArray{T}, in :: AbstractArray{T}) where T
  @assert length(out) == length(in)
  @assert length(trues) == length(falses)
  @assert length(out) <= length(trues)

  # Partition in chunks.
  thread_trues, thread_falses = parallel_iterate(length(in)) do thread_range
    @views partition!(f, trues[thread_range], falses[thread_range], in[thread_range])
  end

  # Compute chunk locations in final list.
  trues_end_indicies  = cumsum(map(length, thread_trues))
  falses_end_indicies = cumsum(map(length, thread_falses))
  trues_count  = last(trues_end_indicies)
  falses_count = last(falses_end_indicies)

  # @assert trues_count + falses_count == length(in)

  trues  = view(out, 1:trues_count)
  falses = view(out, (trues_count+1):length(out))

  # @assert Threads.nthreads() == length(thread_trues)
  # @assert Threads.nthreads() == length(thread_falses)

  _parallel_partition!(trues, falses, trues_end_indicies, falses_end_indicies, thread_trues, thread_falses)

  # @assert trues  == filter(f, in)
  # @assert falses == filter(x -> !f(x), in)

  (trues, falses)
end

function _parallel_partition!(trues, falses, trues_end_indicies, falses_end_indicies, thread_trues, thread_falses)
  Threads.@threads for thread_i in 1:Threads.nthreads()
    trues_start_index  = 1 + (thread_i >= 2 ? trues_end_indicies[thread_i-1]  : 0)
    falses_start_index = 1 + (thread_i >= 2 ? falses_end_indicies[thread_i-1] : 0)
    trues_range  = trues_start_index:trues_end_indicies[thread_i]
    falses_range = falses_start_index:falses_end_indicies[thread_i]
    trues[trues_range]   = thread_trues[thread_i]
    falses[falses_range] = thread_falses[thread_i]
  end
  ()
end


default_config = (
  weights                            = nothing, # weights for the data
  bin_count                          = 255,
  iteration_count                    = 100,
  min_data_weight_in_leaf            = 10.0,
  l2_regularization                  = 1.0,
  max_leaves                         = 32,
  max_depth                          = 6,
  max_delta_score                    = 1.0e10, # Before shrinkage.
  learning_rate                      = 0.03,
  feature_fraction                   = 1.0, # Per tree.
  exclude_features                   = [], # Indices. Applied before feature_fraction
  bagging_temperature                = 1.0, # Same as Catboost's Bayesian bagging. 0.0 doesn't change the weights. 1.0 samples from the exponential distribution to scale each datapoint's weight.
  feature_i_to_name                  = nothing,
  iteration_callback                 = nothing, # Optional. Callback is given trees. If you want to override the default early stopping validation callback.
  validation_X                       = nothing,
  validation_y                       = nothing,
  validation_weights                 = nothing,
  max_iterations_without_improvement = typemax(Int64)
)


function get_config_field(config, key)
  if !isa(config, Tuple{}) && haskey(config, key)
    config[key]
  else
    default_config[key]
  end
end

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0

const Score      = Float64
const Loss       = Float64
const Prediction = Float64
const DataWeight = Float64

const BinSplits = Vector{T} where T <: AbstractFloat

const max_bins = Int64(typemax(UInt8))

const Data = Array{UInt8,2}

data_count(X :: Array{<:Number,2}) = size(X,1)

feature_count(X :: Array{<:Number,2}) = size(X,2)


abstract type Tree end

mutable struct SplitCandidate # When running on multiple machines, these should always be the global stats.
  expected_Δloss    :: Loss
  feature_i         :: Int64
  split_i           :: UInt8
  left_Δscore       :: Score
  left_data_weight  :: DataWeight
  right_Δscore      :: Score
  right_data_weight :: DataWeight
end

const dont_split = SplitCandidate(0f0, 0, 0x00, 0f0, 0f0, 0f0, 0f0)

# struct LossInfo
#   ∇loss  :: Loss
#   ∇∇loss :: Loss
#   weight :: DataWeight
#   dummy  :: Loss

#   LossInfo(∇loss, ∇∇loss, weight = 1f0) = new(∇loss, ∇∇loss, weight, 0f0)
# end
# zeros(LossInfo, n) = reinterpret(LossInfo, zeros(Float64, n*4))

# mutable struct Histogram
#   Σ∇losses     :: Vector{Loss}
#   Σ∇∇losses    :: Vector{Loss}
#   data_weights :: Vector{DataWeight}

#   Histogram(bin_count) = new(zeros(Loss, bin_count), zeros(Loss, bin_count), zeros(DataWeight, bin_count))
# end
const Histogram = Vector{Loss}


mutable struct Node <: Tree
  feature_i           :: Int64
  split_i             :: UInt8
  left                :: Tree
  right               :: Tree
  features_histograms :: Vector{Union{Histogram,Nothing}} # Transient. Used to speed up tree calculation.
end

mutable struct Leaf <: Tree
  Δscore                           :: Score                            # Called "weight" in the literature
  is                                                                   # Transient. Needed during tree growing.
  max_data_count_on_single_machine :: Union{Int64,Nothing}             # Transient. Needed during tree growing across multiple machines, so each machine splits leaves in the same order.
  maybe_data_weight                :: Union{DataWeight,Nothing}        # Transient. Needed during tree growing.
  maybe_split_candidate            :: Union{SplitCandidate,Nothing}    # Transient. Needed during tree growing.
  features_histograms              :: Vector{Union{Histogram,Nothing}} # Transient. Used to speed up tree calculation.

  Leaf(Δscore, is = nothing, max_data_count_on_single_machine = nothing, maybe_data_weight = nothing, maybe_split_candidate = nothing, features_histograms = []) = new(Δscore, is, max_data_count_on_single_machine, maybe_data_weight, maybe_split_candidate, features_histograms)
end

# For use in a list; right_i and left_i are indices into the list. feature_i == -1 for leaves.
# Would love to generate a Julia-native function, but can't seem to dodge world conflicts. (And invokelatest is slow.)
struct FastNode
  feature_i :: Int64
  split_i   :: Int64
  left_i    :: Int64
  right_i   :: Int64
  Δscore    :: Score
end

function tree_to_dict(node :: Node) :: Dict{Symbol,Any}
  Dict(
    :type      => "Node",
    :feature_i => node.feature_i,
    :split_i   => node.split_i,
    :left      => tree_to_dict(node.left),
    :right     => tree_to_dict(node.right)
  )
end

function tree_to_dict(leaf :: Leaf) :: Dict{Symbol,Any}
  Dict(
    :type        => "Leaf",
    :delta_score => leaf.Δscore
  )
end

function dict_to_tree(dict :: Dict{Symbol,Any}) :: Tree
  if dict[:type] == "Node"
    Node(dict[:feature_i], dict[:split_i], dict_to_tree(dict[:left]), dict_to_tree(dict[:right]), [])
  elseif dict[:type] == "Leaf"
    Leaf(dict[:delta_score])
  else
    error("Bad tree type $(dict[:type])!")
  end
end

# Returns path
# Simple use of BSON.@save sometimes croaks on load with "UndefVarError: MemoryConstrainedTreeBoosting not defined"
# So we avoid our custom Tree things
function save(path, bin_splits, trees)
  trees = map(strip_tree_training_info, trees)

  # BSON.@save path bin_splits trees
  # Default serialization is fine for bin_splits.
  BSON.bson(path, bin_splits = bin_splits, trees = map(tree_to_dict, trees))

  path
end


# Returns (bin_splits, trees)
# Simple use of BSON.@load sometimes croaks with "UndefVarError: MemoryConstrainedTreeBoosting not defined"
function load(path)
  # BSON.@load path bin_splits trees
  # println(BSON.parse(path))
  dict       = BSON.load(path)
  bin_splits = dict[:bin_splits] # Default deserialization is fine for bin_splits.
  trees      = map(dict_to_tree, dict[:trees])

  FeatureType = typeof(bin_splits[1][1])
  (Vector{BinSplits{FeatureType}}(bin_splits), Vector{Tree}(trees))
end

# I tried to use Julia's JIT, but it was horreeeeendeously slow to compile the trees
# to native code (sometimes on the order of minutes). Of course, it was blazing fast_nodes
# once that was done, but...
function load_unbinned_predictor(path)
  bin_splits, trees = load(path)

  fast_trees = map(tree_to_fast_nodes, trees)

  predict(X) = begin
    thread_scores = parallel_iterate(length(fast_trees)) do thread_range
      scores = zeros(Score, data_count(X))

      for tree_i in thread_range
        fast_nodes = fast_trees[tree_i]

        @inbounds for i in 1:data_count(X)
          node_i = 1
          @inbounds while true
            node = fast_nodes[node_i]
            if node.feature_i > 0
              if X[i, node.feature_i] < bin_splits[node.feature_i][node.split_i]
                node_i = node.left_i
              else
                node_i = node.right_i
              end
            else
              scores[i] += node.Δscore
              break
            end
          end
        end
      end

      scores
    end

    σ.(sum(thread_scores))
  end

  predict
end

function print_tree(tree, level = 0; feature_i_to_name = nothing, bin_splits = nothing)
  indentation = repeat("    ", level)
  if isa(tree, Node)
    feature_name = isnothing(feature_i_to_name) ? "feature $(tree.feature_i)" : feature_i_to_name(tree.feature_i)
    split_str = isnothing(bin_splits) ? "$(tree.split_i)" : "$(bin_splits[tree.feature_i][tree.split_i])"
    println(indentation * "$feature_name\tsplit at $split_str")
    print_tree(tree.left,  level + 1, feature_i_to_name = feature_i_to_name)
    print_tree(tree.right, level + 1, feature_i_to_name = feature_i_to_name)
  else
    println(indentation * "Δscore $(tree.Δscore)")
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
    if isnothing(parent_in_left)
      parent_node(tree.right, target)
    else
      parent_in_left
    end
  end
end

function sibling_node(tree, target) :: Union{Tree,Nothing}
  parent = parent_node(tree, target)
  if !isnothing(parent)
    parent.left === target ? parent.right : parent.left
  else
    nothing
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
function replace_leaf!(tree, old, replacement)
  if tree === old
    replacement
  elseif isa(tree, Node)
    tree.left  = replace_leaf!(tree.left,  old, replacement)
    tree.right = replace_leaf!(tree.right, old, replacement)
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
    Leaf(tree.Δscore, nothing, nothing, nothing, nothing, [])
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
function apply_tree(X_binned, tree :: Tree) :: Vector{Score}
  scores = zeros(Score, data_count(X_binned))
  apply_tree!(X_binned, tree, scores)
end

# Mutates scores.
function apply_tree!(X_binned, tree :: Tree, scores :: Vector{Score}) :: Vector{Score}
  # Would love to generate a function and use it, but can't seem to dodge world conflicts and invokelatest was slow.
  fast_nodes = tree_to_fast_nodes(tree)

  _apply_tree!(X_binned, fast_nodes, scores)
end

# Pre-order traversal, so index of subtree root is always the given node_i.
function tree_to_fast_nodes(node :: Node, node_i = 1) :: Vector{FastNode}
  left_i = node_i + 1
  left_nodes  = tree_to_fast_nodes(node.left, left_i)
  right_i = left_i + length(left_nodes)
  right_nodes = tree_to_fast_nodes(node.right, right_i)

  node = FastNode(node.feature_i, node.split_i, left_i, right_i, 0f0)
  vcat([node], left_nodes, right_nodes)
end

function tree_to_fast_nodes(leaf :: Leaf, node_i = 1) :: Vector{FastNode}
  [FastNode(-1, -1, -1, -1, leaf.Δscore)]
end

# Mutates scores.
function _apply_tree!(X_binned, fast_nodes :: Vector{FastNode}, scores :: Vector{Score}) :: Vector{Score}
  Threads.@threads for i in 1:data_count(X_binned)
    node_i = 1
    @inbounds while true
      node = fast_nodes[node_i]
      if node.feature_i > 0
        if X_binned[i, node.feature_i] <= node.split_i
          node_i = node.left_i
        else
          node_i = node.right_i
        end
      else
        scores[i] += node.Δscore
        break
      end
    end
  end
  scores
end

# Returns vector of untransformed scores (linear, pre-sigmoid). Does not mutate starting_scores.
function apply_trees(X_binned, trees :: Vector{<:Tree}; starting_scores = nothing) :: Vector{Score}

  # thread_scores = map(_ -> zeros(Score, data_count(X_binned)), 1:Threads.nthreads())
  scores = zeros(Score, data_count(X_binned))

  # Threads.@threads for tree in trees
  for tree in trees
    # apply_tree!(X_binned, tree, thread_scores[Threads.threadid()])
    apply_tree!(X_binned, tree, scores)
  end

  # scores = sum(thread_scores)

  if !isnothing(starting_scores)
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
function predict_on_binned(X_binned, trees :: Vector{<:Tree}; starting_scores = nothing, output_raw_scores = false) :: Vector{Prediction}
  scores = apply_trees(X_binned, trees; starting_scores = starting_scores)
  if output_raw_scores
    scores
  else
    parallel_map!(σ, scores, scores)
  end
end


# Aim for a roughly equal number of data points in each bin.
# Does not support weights.
function prepare_bin_splits(X :: Array{FeatureType,2}; bin_count = 255) :: Vector{BinSplits{FeatureType}} where FeatureType <: AbstractFloat
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
    sorted_feature_values = sort(@view X[is, j])

    splits = zeros(eltype(sorted_feature_values), split_count)

    for split_i in 1:split_count
      split_sample_i = max(1, Int64(floor(sample_count / bin_count * split_i)))
      value_below_split = sorted_feature_values[split_sample_i]
      value_above_split = sorted_feature_values[min(split_sample_i + 1, sample_count)]
      splits[split_i] = (value_below_split + value_above_split) / 2f0 # Avoid coercing Float64 to Float64
    end

    bin_splits[j] = splits
  end

  bin_splits
end


function apply_bins(X, bin_splits) :: Data
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

struct EarlyStop <: Exception
end

# Note the returned closure is stateful (need to remake the iteration callback for new runs).
function make_callback_to_track_validation_loss(validation_X_binned, validation_y; validation_weights = nothing, max_iterations_without_improvement = typemax(Int64), mpi_comm = nothing)
  validation_scores              = nothing
  best_loss                      = Loss(Inf)
  iterations_without_improvement = 0

  # Returns validation_loss in case you want to call this from your own callback.
  #
  # On early stop, mutates trees to remove the last few unhelpful trees.
  iteration_callback(trees) = begin
    new_tree = last(trees)

    if isnothing(validation_scores)
      validation_scores = predict_on_binned(validation_X_binned, trees, output_raw_scores = true)
    else
      apply_tree!(validation_X_binned, new_tree, validation_scores)
    end
    validation_loss = compute_mean_logloss(validation_y, validation_scores; weights = validation_weights, mpi_comm = mpi_comm)

    if validation_loss < best_loss
      best_loss                      = validation_loss
      iterations_without_improvement = 0
    else
      iterations_without_improvement += 1
      if iterations_without_improvement >= max_iterations_without_improvement
        resize!(trees, length(trees) - max_iterations_without_improvement)
        throw(EarlyStop())
      end
    end
    mpi_print(mpi_comm, "\rBest validation loss: $best_loss    ")

    validation_loss
  end

  iteration_callback
end

# Trains for iteration_count rounds and returns (bin_splits, prior_and_new_trees).
function train(X :: Array{FeatureType,2}, y; bin_splits=nothing, prior_trees=Tree[], config...) :: Tuple{Vector{BinSplits{FeatureType}}, Vector{Tree}} where FeatureType <: AbstractFloat
  if isnothing(bin_splits)
    print("Preparing bin splits...")
    bin_splits = prepare_bin_splits(X; bin_count = get_config_field(config, :bin_count))
    println("done.")
  end
  # println(bin_splits)

  print("Binning input data...")
  X_binned = apply_bins(X, bin_splits)
  println("done.")
  # println(X_binned)

  iteration_callback =
    if !isnothing(get_config_field(config, :validation_X)) && !isnothing(get_config_field(config, :validation_y))
      if isnothing(get_config_field(config, :iteration_callback))
        print("Binning validation data...")
        validation_X_binned = apply_bins(get_config_field(config, :validation_X), bin_splits)
        println("done.")

        make_callback_to_track_validation_loss(
            validation_X_binned,
            get_config_field(config, :validation_y);
            validation_weights = get_config_field(config, :validation_weights),
            max_iterations_without_improvement = get_config_field(config, :max_iterations_without_improvement)
          )
      else
        println("Warning: both validation_X and iteration_callback provided. validation_X ignored.")
        get_config_field(config, :iteration_callback)
      end
    elseif !isnothing(get_config_field(config, :validation_X)) || !isnothing(get_config_field(config, :validation_y))
      println("Warning: Must provide both validation_X and validation_y!")
      get_config_field(config, :iteration_callback)
    else
      get_config_field(config, :iteration_callback)
    end

  trees = train_on_binned(X_binned, y; prior_trees = prior_trees, config..., iteration_callback = iteration_callback)

  (bin_splits, trees)
end

index_type(array) = length(array) < typemax(UInt32) ? UInt32 : Int64

mutable struct ScratchHistograms
  next_free_i :: Threads.Atomic{Int64}
  histograms  :: Vector{Vector{Loss}}
  consolidated_for_mpi_communication :: Union{Vector{Loss},Nothing}

  ScratchHistograms(X_binned; config...) = begin
    raw_features_count = feature_count(X_binned) - length(unique(get_config_field(config, :exclude_features)))
    features_to_use_count = Int(ceil(get_config_field(config, :feature_fraction) * raw_features_count))
    histogram_count = features_to_use_count * min(get_config_field(config, :max_leaves), 2^get_config_field(config, :max_depth) - 1)
    histogram_size  = 4*max_bins + Int(64/sizeof(Loss)) # Ensure no two features share a cache line...allocate a little extra.
    histograms = map(_ -> resize!(Vector{Loss}(undef, histogram_size), 4*max_bins), 1:histogram_count)

    new(Threads.Atomic{Int64}(1), histograms, nothing)
  end
end

function reset_scratch_histograms(scratch_histograms :: ScratchHistograms)
  Threads.atomic_xchg!(scratch_histograms.next_free_i, 1)
  scratch_histograms
end

function next_free_histogram(scratch_histograms :: ScratchHistograms)
  free_i = Threads.atomic_add!(scratch_histograms.next_free_i, 1)
  histogram =
    if free_i <= length(scratch_histograms.histograms)
      scratch_histograms.histograms[free_i]
    else
      println("WARNING: Math off: should never need to allocate new histograms")
      histogram_size  = 4*max_bins + Int(64/sizeof(Loss))
      resize!(Vector{Loss}(undef, histogram_size), 4*max_bins)
    end

  fill!(histogram, Loss(0))
end

# Reusable memory to avoid allocations between trees.
mutable struct ScratchMemory
  ∇losses_∇∇losses_weights :: Vector{Loss}
  ∇losses_∇∇losses_weights_consolidated :: Vector{Loss}
  # ∇losses  :: Vector{Loss}
  # ∇∇losses :: Vector{Loss}
  # weights  :: Union{Nothing,Vector{DataWeight}}
  is       :: Union{Vector{UInt32},Vector{Int64}}
  trues    :: Union{Vector{UInt32},Vector{Int64}} # During parallel partition
  falses   :: Union{Vector{UInt32},Vector{Int64}} # During parallel partition
  scratch_histograms :: ScratchHistograms

  ScratchMemory(X_binned, y; mpi_comm, config...) = begin
    histogram_count = min(get_config_field(config, :max_leaves), 2^get_config_field(config, :max_depth) - 1)
    histogram_size  = 4*max_bins + Int(64/sizeof(Loss)) # Ensure no two features share a cache line...allocate a little extra.

    new(
      Vector{Loss}(undef, data_count(X_binned)*4),
      Vector{Loss}(undef, data_count(X_binned)*(isnothing(mpi_comm) ? 2 : 4)), # when not distributed, leaf_is will always be less than half the total
      # Vector{Loss}(undef, length(y)),
      # get_config_field(config, :bagging_temperature) > 0 ? Vector{DataWeight}(undef, length(y)) : nothing,
      Vector{index_type(y)}(undef, data_count(X_binned)),
      Vector{index_type(y)}(undef, data_count(X_binned)),
      Vector{index_type(y)}(undef, data_count(X_binned)),
      ScratchHistograms(X_binned; config...)
    )
  end
end

function train_on_binned(X_binned, y; prior_trees=Tree[], mpi_comm = nothing, config...) :: Vector{Tree}
  weights = get_config_field(config, :weights)
  if isnothing(weights)
    weights = ones(DataWeight, length(y))
  end

  if isempty(prior_trees)
    initial_score = begin
      probability = compute_mean_probability(y, weights, mpi_comm = mpi_comm)
      log(probability / (1-probability)) # inverse sigmoid
    end

    # print("$initial_score\n")

    prior_trees = Tree[Leaf(initial_score)]
  end

  scores = apply_trees(X_binned, prior_trees) # Linear scores, before sigmoid transform.

  trees = copy(prior_trees)

  scratch_memory = ScratchMemory(X_binned, y; mpi_comm = mpi_comm, config...)

  try
    for iteration_i in 1:get_config_field(config, :iteration_count)
      duration = @elapsed begin
        tree = train_one_iteration(X_binned, y, weights, scores; scratch_memory = scratch_memory, mpi_comm = mpi_comm, config...)
        tree = strip_tree_training_info(tree) # For long boosting sessions, should save memory if we strip off the list of indices
        apply_tree!(X_binned, tree, scores)

        # iteration_loss = compute_mean_logloss(y, scores, weights)
        # println(ŷ)
        # println("Iteration $iteration_i training loss: $iteration_loss")

        # print_tree(tree; feature_i_to_name = get_config_field(config, :feature_i_to_name))
        # println()

        push!(trees, tree)

        if !isnothing(get_config_field(config, :iteration_callback))
          get_config_field(config, :iteration_callback)(trees)
        end
      end
      mpi_print(mpi_comm, "$duration sec/tree     \n")
    end
  catch expection
    mpi_print(mpi_comm, "\n")
    if isa(expection, EarlyStop)
    else
      rethrow()
    end
  end

  # scores         = nothing
  # weights        = nothing
  # scratch_memory = nothing
  # GC.gc(true)

  trees
end


# Returns new tree
function train_one_iteration(X_binned, y :: Vector{Prediction}, weights :: Vector{DataWeight}, scores :: Vector{Score}; scratch_memory = nothing, mpi_comm = nothing, config...) :: Tree
  if isnothing(scratch_memory)
    scratch_memory = ScratchMemory(X_binned, y; mpi_comm = mpi_comm, config...)
  end
  learning_rate       = Score(get_config_field(config, :learning_rate))
  bagging_temperature = DataWeight(get_config_field(config, :bagging_temperature))


  ∇losses_∇∇losses_weights = scratch_memory.∇losses_∇∇losses_weights
  ∇losses_∇∇losses_weights_scratch = scratch_memory.∇losses_∇∇losses_weights_consolidated
  # ∇losses  = isnothing(scratch_memory) ? Vector{Loss}(undef, length(y))          : scratch_memory.∇losses
  # ∇∇losses = isnothing(scratch_memory) ? Vector{Loss}(undef, length(y))          : scratch_memory.∇∇losses
  is       = scratch_memory.is
  trues    = scratch_memory.trues
  falses   = scratch_memory.falses
  scratch_histograms = scratch_memory.scratch_histograms
  reset_scratch_histograms(scratch_histograms)

  compute_weights!(weights, bagging_temperature, ∇losses_∇∇losses_weights)
  # Needs to be a separate method otherwise type inference croaks.
  compute_∇losses_∇∇losses!(y, scores, ∇losses_∇∇losses_weights)

  tree = build_one_tree(X_binned, ∇losses_∇∇losses_weights, ∇losses_∇∇losses_weights_scratch, is, trues, falses, scratch_histograms; mpi_comm = mpi_comm, config...)
  tree = scale_leaf_Δscores(tree, learning_rate)
  tree
end

# Mutates out
function bagged_weights!(weights, bagging_temperature, out)
  seed = rand(Int)

  # logs and powers are slow; sample from pre-computed distribution for non-extreme values.

  weight_lookup = map(r -> (-log(r))^bagging_temperature, 0.005f0:0.01f0:0.995f0)
  # Temperatures between 0 and 1 shift the expected effective total weight
  # downward slightly (up to ~11.5%) which effectively changes the
  # min_data_in_leaf and L2 regularization. Compensate.
  temperature_compensation = length(weight_lookup) / sum(weight_lookup)
  weight_lookup .*= temperature_compensation

  parallel_iterate(length(weights)) do thread_range
    rng = Random.MersenneTwister(abs(seed + thread_range.start * 1234))
    @inbounds for i in thread_range
      bin_i = rand(rng, 1:100)
      count =
        if bin_i in 2:99
          weight_lookup[bin_i]
        else
          r = rand(rng, DataWeight) * 0.01f0 + (bin_i == 100 ? 0.99f0 : ε)
          (-log(r))^bagging_temperature * temperature_compensation
        end

      out[i] = weights[i] * count
    end
    ()
  end

  ()
end

function build_one_tree(X_binned, ∇losses_∇∇losses_weights, ∇losses_∇∇losses_weights_scratch, is, trues, falses, scratch_histograms; mpi_comm = nothing, config...)
  # Use a range rather than a list for the root. Saves us having to initialize is.
  all_is = UnitRange{index_type(1:data_count(X_binned))}(1:data_count(X_binned))
  max_data_count_on_single_machine = mpi_max(mpi_comm, length(all_is))
  tree =
    Leaf(
      sum_optimal_Δscore(llw_∇losses(∇losses_∇∇losses_weights), llw_∇∇losses(∇losses_∇∇losses_weights), Loss(get_config_field(config, :l2_regularization)), Score(get_config_field(config, :max_delta_score))),
      all_is,
      max_data_count_on_single_machine,
      nothing, # maybe_data_weight
      nothing, # maybe_split_candidate
      []       # feature_histograms
    )

  raw_features_count = feature_count(X_binned) - length(unique(get_config_field(config, :exclude_features)))
  features_to_use_count = UInt32(ceil(get_config_field(config, :feature_fraction) * raw_features_count))

  # This still allocates :/
  # I suspect the cache benefits for sorting the indexes are trivial but it feels cleaner.
  feature_is = mpi_compute_on_one_and_share(mpi_comm) do
    sort(
      Random.shuffle(
        setdiff(UInt32(1):UInt32(feature_count(X_binned)), get_config_field(config, :exclude_features))
      )[1:features_to_use_count]
    )
  end

  tree_changed = true

  while tree_changed
    # print_tree(tree)
    # println()
    # println()
    (tree_changed, tree) = perhaps_split_tree(tree, X_binned, ∇losses_∇∇losses_weights, ∇losses_∇∇losses_weights_scratch, is, feature_is, trues, falses, scratch_histograms; mpi_comm = mpi_comm, config...)
  end

  tree
end


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

# Copied from Flux.jl.
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

# Derivatives with respect to margin (i.e. pre-sigmoid values, but you still provide the post-sigmoid probability).
#
# It's a bit of math but it works out nicely. (Verified against XGBoost.)
@inline ∇logloss(y, ŷ) = ŷ - y
@inline ∇∇logloss(ŷ)   = ŷ * (1.0f0 - ŷ) # Interestingly, not dependent on y. XGBoost adds an ε term

# Only print on the root process
function mpi_print(comm, str)
  if isnothing(comm) || MPI.Comm_rank(comm) == 0
    print(str)
  end
end

function mpi_sum_histograms!(comm, feature_is_to_compute, leaf_features_histograms, scratch_histograms)
  isnothing(comm) && return ()
  length(feature_is_to_compute) == 0 && return ()

  # # The slow but simple way, lots of communication.
  # for feature_i in feature_is_to_compute
  #   MPI.Allreduce!(leaf_features_histograms[feature_i], +, comm)
  # end

  # Smash all the histograms into one long array before sending off.
  # Reduces communication calls.

  hist_size = length(leaf_features_histograms[feature_is_to_compute[1]])

  if isnothing(scratch_histograms.consolidated_for_mpi_communication)
    scratch_histograms.consolidated_for_mpi_communication = Vector{Loss}(undef, hist_size * length(feature_is_to_compute))
  end
  buf = scratch_histograms.consolidated_for_mpi_communication

  # We thread all the other places like this, so...
  Threads.@threads for feature_ii in 1:length(feature_is_to_compute)
    feature_i = feature_is_to_compute[feature_ii]
    buf[(feature_ii-1)*hist_size + 1 : feature_ii*hist_size] = leaf_features_histograms[feature_i]
  end

  MPI.Allreduce!(buf, +, comm)

  Threads.@threads for feature_ii in 1:length(feature_is_to_compute)
    feature_i = feature_is_to_compute[feature_ii]
    leaf_features_histograms[feature_i][:] = @view buf[(feature_ii-1)*hist_size + 1 : feature_ii*hist_size]
  end

  ()
end

function mpi_max(comm, x)
  isnothing(comm) ? x : MPI.Allreduce(x, max, comm)
end

function mpi_sum(comm, xs...)
  isnothing(comm) ? xs : MPI.Allreduce([xs...], +, comm)
end

function mpi_mean(comm, my_Σ, my_weight)
  # thank you to time 5:06 of https://www.youtube.com/watch?v=pV-8YqfOxQE
  Σ, weight = mpi_sum(comm, my_Σ, my_weight)
  Σ / weight
end

function mpi_compute_on_one_and_share(f, comm)
  isnothing(comm) && return f()
  out = MPI.Comm_rank(comm) == 0 ? f() : nothing
  out = MPI.bcast(out, 0, comm)
  out
end

function compute_mean_probability(y, weights; mpi_comm = nothing)
  thread_Σlabels, thread_Σweight = parallel_iterate(length(y)) do thread_range
    Σlabel  = 0.0
    Σweight = 0.0
    @inbounds for i in thread_range
      Σlabel  += Float64(y[i] * weights[i])
      Σweight += Float64(weights[i])
    end
    (Float64(Σlabel), Float64(Σweight))
  end
  mpi_mean(mpi_comm, sum(thread_Σlabels), sum(thread_Σweight))
end

function compute_mean_logloss(y, scores; weights = nothing, mpi_comm = nothing)
  Σlosses, Σweights =  _compute_mean_logloss(y, scores, weights)
  mpi_mean(mpi_comm, Σlosses, Σweights)
end

function _compute_mean_logloss(y, scores, weights :: Nothing)
  thread_Σlosses = parallel_iterate(length(y)) do thread_range
    Σloss   = 0.0
    @inbounds for i in thread_range
      ŷ_i      = σ(scores[i])
      Σloss   += Float64(logloss(y[i], ŷ_i))
    end
    Float64(Σloss)
  end
  sum(thread_Σlosses), Float64(length(y))
end

function _compute_mean_logloss(y, scores, weights :: Vector{DataWeight})
  # Broadcast version, which performs allocations:
  # ŷ = σ.(scores)
  # mean_logloss = sum(logloss.(y, ŷ) .* weights) / sum(weights)
  thread_Σlosses, thread_Σweights = parallel_iterate(length(y)) do thread_range
    Σloss   = 0.0
    Σweight = 0.0
    @inbounds for i in thread_range
      ŷ_i      = σ(scores[i])
      Σloss   += Float64(logloss(y[i], ŷ_i) * weights[i])
      Σweight += Float64(weights[i])
    end
    (Float64(Σloss), Float64(Σweight))
  end
  sum(thread_Σlosses), sum(thread_Σweights)
end

# @inline llw_base_i(i) = 1+(i-1)*4
@inline llw_base_i(i) = -3 + 4*i
llw_∇losses(∇losses_∇∇losses_weights)  = @view ∇losses_∇∇losses_weights[1:4:length(∇losses_∇∇losses_weights)]
llw_∇∇losses(∇losses_∇∇losses_weights) = @view ∇losses_∇∇losses_weights[2:4:length(∇losses_∇∇losses_weights)]
llw_weights(∇losses_∇∇losses_weights)  = @view ∇losses_∇∇losses_weights[3:4:length(∇losses_∇∇losses_weights)]

function compute_weights!(weights, bagging_temperature, ∇losses_∇∇losses_weights)
  out = llw_weights(∇losses_∇∇losses_weights)
  # Adapted from Catboost
  if bagging_temperature > 0
    bagged_weights!(weights, bagging_temperature, out)
  else
    parallel_iterate(length(weights)) do thread_range
      out[thread_range] = weights[thread_range]
    end
  end
  ()
end

# Stores results in ∇losses, ∇∇losses of ∇losses_∇∇losses_weights
function compute_∇losses_∇∇losses!(y, scores, ∇losses_∇∇losses_weights)
  llw = ∇losses_∇∇losses_weights
  parallel_iterate(length(y)) do thread_range
    llw_i = llw_base_i(thread_range.start)
    @inbounds for i in thread_range
      ŷ_i          = σ(scores[i])
      llw[llw_i]   = ∇logloss(y[i], ŷ_i) * llw[llw_i+2]
      llw[llw_i+1] = ∇∇logloss(ŷ_i)      * llw[llw_i+2]
      llw_i += 4
    end
    # @inbounds begin
      # ŷ_i = σ(scores[i])
      # ∇losses[i]  = (ŷ_i - y[i])        * weights[i] # ∇logloss  = ŷ - y
      # ∇∇losses[i] = ŷ_i * (1.0f0 - ŷ_i) * weights[i] # ∇∇logloss = ŷ * (1.0f0 - ŷ)
    # end
    # out[thread_range] = weights[thread_range]
  end

  # Threads.@threads for i in 1:length(y)
  #   @inbounds begin
  #     ŷ_i = σ(scores[i])
  #     ∇losses[i]  = (ŷ_i - y[i])        * weights[i] # ∇logloss  = ŷ - y
  #     ∇∇losses[i] = ŷ_i * (1.0f0 - ŷ_i) * weights[i] # ∇∇logloss = ŷ * (1.0f0 - ŷ)
  #   end
  # end
  ()
end

# Assuming binary classification with log loss.
function sum_optimal_Δscore(∇losses :: AbstractArray{Loss}, ∇∇losses :: AbstractArray{Loss}, l2_regularization :: Loss, max_delta_score :: Score; mpi_comm = nothing)
  Σ∇loss, Σ∇∇loss = sum_∇loss_∇∇loss(∇losses, ∇∇losses)
  Σ∇loss, Σ∇∇loss = mpi_sum(mpi_comm, Σ∇loss, Σ∇∇loss)

  optimal_Δscore(Σ∇loss, Σ∇∇loss, l2_regularization, max_delta_score)
end

function optimal_Δscore(Σ∇loss :: Loss, Σ∇∇loss :: Loss, l2_regularization :: Loss, max_delta_score :: Score)
  # And the loss minima is at:
  clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)
end

function sum_∇loss_∇∇loss(∇losses, ∇∇losses)
  thread_Σ∇losses, thread_Σ∇∇losses = parallel_iterate(length(∇losses)) do thread_range
    Σ∇loss  = 0.0
    Σ∇∇loss = 0.0
    @inbounds for i in thread_range
      Σ∇loss  += Float64(∇losses[i])
      Σ∇∇loss += Float64(∇∇losses[i])
    end
    (Float64(Σ∇loss), Float64(Σ∇∇loss))
  end

  (sum(thread_Σ∇losses), sum(thread_Σ∇∇losses))
end

# -0.5 * (Σ∇loss)² / (Σ∇∇loss) in XGBoost paper; but can't simplify so much if clamping the score.
function leaf_expected_Δloss(Σ∇loss :: Loss, Σ∇∇loss :: Loss, l2_regularization :: Loss, max_delta_score :: Score)
  Δscore = optimal_Δscore(Σ∇loss, Σ∇∇loss, l2_regularization, max_delta_score)

  Σ∇loss * Δscore + 0.5f0 * Σ∇∇loss * Δscore * Δscore
end

# Mutates tree, but also returns the tree in case tree was a lone leaf.
#
# Returns (bool, tree) where bool is true if any split was made, otherwise false.
function perhaps_split_tree(tree, X_binned, ∇losses_∇∇losses_weights, ∇losses_∇∇losses_weights_scratch, is, feature_is, trues, falses, scratch_histograms; mpi_comm = nothing, config...)

  # This still allocates :/
  leaves = sort(tree_leaves(tree), by = (leaf -> leaf.max_data_count_on_single_machine)) # Process smallest leaves first, should speed up histogram computation.

  if length(leaves) >= get_config_field(config, :max_leaves)
    return (false, tree)
  end

  min_data_weight_in_leaf = DataWeight(get_config_field(config, :min_data_weight_in_leaf))
  l2_regularization       = Loss(get_config_field(config, :l2_regularization))
  max_delta_score         = Score(get_config_field(config, :max_delta_score))

  # Ensure split_candidate on all leaves
  for leaf in leaves
    if isnothing(leaf.maybe_split_candidate) && leaf_depth(tree, leaf) >= get_config_field(config, :max_depth)
      leaf.maybe_split_candidate = dont_split

    elseif isnothing(leaf.maybe_split_candidate)

      leaf_too_small_to_split = !isnothing(leaf.maybe_data_weight) && leaf.maybe_data_weight < min_data_weight_in_leaf*DataWeight(2)
      maybe_sibling = sibling_node(tree, leaf)
      sibling_splittable_and_needs_histograms =
        isa(maybe_sibling, Leaf) && isnothing(maybe_sibling.maybe_split_candidate) &&
        (isnothing(maybe_sibling.maybe_data_weight) || maybe_sibling.maybe_data_weight >= min_data_weight_in_leaf*DataWeight(2))

      if leaf_too_small_to_split && !sibling_splittable_and_needs_histograms
        leaf.maybe_split_candidate = dont_split
      else
        # Find best feature and best split
        # Expected Δlogloss at leaf = -0.5 * (Σ ∇loss)² / (Σ ∇∇loss)

        if isempty(leaf.features_histograms)
          leaf.features_histograms = map(_ -> nothing, 1:feature_count(X_binned))
        end

        # Don't really need threads on this one but it doesn't hurt.
        Threads.@threads for feature_i in feature_is
          perhaps_calculate_feature_histogram_from_parent_and_sibling!(feature_i, tree, leaf)
        end

        feature_is_to_compute = filter(feature_i -> isnothing(leaf.features_histograms[feature_i]), feature_is)

        if !isempty(feature_is_to_compute)
          Threads.@threads for feature_i in feature_is_to_compute
            leaf.features_histograms[feature_i] = next_free_histogram(scratch_histograms)
          end

          # println(@code_llvm compute_histograms!(X_binned, ∇losses_∇∇losses_weights, feature_is_to_compute, leaf.features_histograms, leaf.is))
          # println()

          # # consolidate ∇losses,∇∇losses,weights when leaf_is is small?
          ∇losses_∇∇losses_weights_consolidated =
            if isa(leaf.is, UnitRange)
              ∇losses_∇∇losses_weights
            else
              consolidate_∇losses_∇∇losses_weights!(∇losses_∇∇losses_weights, leaf.is, ∇losses_∇∇losses_weights_scratch)
            end

          compute_histograms!(X_binned, ∇losses_∇∇losses_weights_consolidated, feature_is_to_compute, leaf.features_histograms, leaf.is)

          # We only calculated the histogram for our local data. Sum with the rest of the data, so histograms are global.
          mpi_sum_histograms!(mpi_comm, feature_is_to_compute, leaf.features_histograms, scratch_histograms)
        end

        leaf.maybe_split_candidate = find_best_split(leaf.features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score)
      end
    end # if leaf.maybe_split_candidate == nothing
  end # for leaf in leaves

  # Expand best split_candiate (if any)

  (expected_Δloss, leaf_i_to_split) = findmin(map(leaf -> leaf.maybe_split_candidate.expected_Δloss, leaves))

  expected_Δloss :: Loss

  if expected_Δloss < 0.0f0
    # We have a usable split!

    leaf_to_split   = leaves[leaf_i_to_split]
    split_candidate = leaf_to_split.maybe_split_candidate

    feature_i = split_candidate.feature_i
    split_i   = split_candidate.split_i

    feature_binned = @view X_binned[:, feature_i]

    # If root node, switch from unit range to our scratch memory
    scratch_is = isa(leaf_to_split.is, UnitRange) ? is : leaf_to_split.is

    left_is, right_is = parallel_partition!(i -> feature_binned[i] <= split_i, scratch_is, trues, falses, leaf_to_split.is)

    left_leaf  = Leaf(split_candidate.left_Δscore,  left_is,  mpi_max(mpi_comm, length(left_is)),  split_candidate.left_data_weight,  nothing, [])
    right_leaf = Leaf(split_candidate.right_Δscore, right_is, mpi_max(mpi_comm, length(right_is)), split_candidate.right_data_weight, nothing, [])

    # left_leaf, right_leaf = make_split_leaves(feature_binned, ∇losses_∇∇losses_weights, leaf_to_split.maybe_data_weight, split_i, scratch_is, trues, falses, leaf_to_split.is, l2_regularization, max_delta_score)

    new_node = Node(feature_i, split_i, left_leaf, right_leaf, leaf_to_split.features_histograms)

    tree = replace_leaf!(tree, leaf_to_split, new_node)

    (true, tree)
  else
    (false, tree)
  end
end

# Returns the parent_histogram mutated into the second sibling
function convert_parent_to_second_sibling_histogram!(parent_histogram, sibling_histogram)
  parent_histogram .-= sibling_histogram
  # @inbounds @simd for i in 1:length(parent_histogram)
  #   parent_histogram[i] -= sibling_histogram[i]
  # end
  parent_histogram
end

# If the parent cached its histogram, and the other sibling has already done its calculation, then we can calculate our histogram by simple subtraction.
#
# Possibly mutates leaf.feature_histograms[feature_i]
function perhaps_calculate_feature_histogram_from_parent_and_sibling!(feature_i, tree, leaf)
  if !isnothing(leaf.features_histograms[feature_i])
    return ()
  end

  parent = parent_node(tree, leaf)
  if !isnothing(parent)
    sibling = sibling_node(tree, leaf)

    if !isempty(parent.features_histograms) && !isempty(sibling.features_histograms)
      parent_histogram  = parent.features_histograms[feature_i]
      sibling_histogram = sibling.features_histograms[feature_i]

      if !isnothing(parent_histogram) && !isnothing(sibling_histogram)
        leaf.features_histograms[feature_i] = convert_parent_to_second_sibling_histogram!(parent_histogram, sibling_histogram)
        parent.features_histograms[feature_i] = nothing
      end
    end
  end

  ()
end


# Get all our cache misses out of the way in one go.
# Stores result in out (and returns out)
function consolidate_∇losses_∇∇losses_weights!(∇losses_∇∇losses_weights, leaf_is, out)

  # Make sure we weren't too fancy with allocating a minimal amount of scratch memory
  @assert length(leaf_is) <= length(out)

  parallel_iterate(length(leaf_is)) do thread_range
    stride = 4
    leaf_ii_stop = thread_range.stop - stride + 1
    consolidate_∇losses_∇∇losses_weights_unrolled!(∇losses_∇∇losses_weights, leaf_is, thread_range.start, leaf_ii_stop, out)

    first_ii_unprocessed = last(thread_range.start:stride:leaf_ii_stop) < thread_range.start ? thread_range.start : last(thread_range.start:stride:leaf_ii_stop) + stride

    # The last couple points...
    llw_i_out = llw_base_i(first_ii_unprocessed)
    @inbounds for leaf_ii in first_ii_unprocessed:thread_range.stop
      llw_i = llw_base_i(leaf_is[leaf_ii])
      SIMD.vstorea(
        SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_i),
        out,
        llw_i_out
      )
      llw_i_out += 4
    end
  end
  out
end

function consolidate_∇losses_∇∇losses_weights_unrolled!(∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, out)
  llw_i_out = llw_base_i(leaf_ii_start)
  @inbounds for leaf_ii in (leaf_ii_start:4:leaf_ii_stop)
    llw_i1 = llw_base_i(leaf_is[leaf_ii])
    llw_i2 = llw_base_i(leaf_is[leaf_ii+1])
    llw_i3 = llw_base_i(leaf_is[leaf_ii+2])
    llw_i4 = llw_base_i(leaf_is[leaf_ii+3])
    loss_info1 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_i1)
    loss_info2 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_i2)
    loss_info3 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_i3)
    loss_info4 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_i4)
    SIMD.vstorea(loss_info1, out, llw_i_out)
    SIMD.vstorea(loss_info2, out, llw_i_out+4)
    SIMD.vstorea(loss_info3, out, llw_i_out+8)
    SIMD.vstorea(loss_info4, out, llw_i_out+12)
    llw_i_out += 16
  end
end

# # Mutates the histogram parts: Σ∇losses, Σ∇∇losses, data_weights
#  Like 2.1 2.3 GB/s, 50s total
function build_histogram_unrolled!(X_binned, feature_i, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram)

  # println(InteractiveUtils.@code_llvm _build_histogram_unrolled!(X_binned, size(X_binned,1)*(feature_i-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, Σ∇losses, Σ∇∇losses, data_weights))
  # throw(:crash)

  _build_histogram_unrolled!(X_binned, size(X_binned,1)*(feature_i-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram)

  first_ii_unprocessed =
    leaf_ii_start + 2*length(leaf_ii_start:2:(leaf_ii_stop-1))

  # The last couple points...
  @inbounds for ii in first_ii_unprocessed:leaf_ii_stop
    i = leaf_is[ii]
    bin_i = llw_base_i(X_binned[i, feature_i])
    llw_i = llw_base_i(ii)
    histogram[bin_i]   += ∇losses_∇∇losses_weights[llw_i]
    histogram[bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
    histogram[bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
  end
end

function _build_histogram_unrolled!(X_binned, feature_start_i, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram)
  # Memory per datapoint = 4 (leaf_is) + 1  (featue_binned) + 4  (∇losses) + 4  (∇∇losses) +  4 (weights) =  17 bytes if dense
  # Memory per datapoint = 4 (leaf_is) + 64 (featue_binned) + 64 (∇losses) + 64 (∇∇losses) + 64 (weights) = 260 bytes if sparse (different cache lines)
  #
  # Conservatively assuming 1MB shared cache, that's 60,000 datapoints in cache if dense, or 4,000 if sparse

  @inbounds for ii in leaf_ii_start:2:(leaf_ii_stop-1)
    i1 = leaf_is[ii]
    i2 = leaf_is[ii+1]
    # i3 = leaf_is[ii+2]
    # i4 = leaf_is[ii+3]

    bin_i1 = llw_base_i(X_binned[feature_start_i + i1])
    bin_i2 = llw_base_i(X_binned[feature_start_i + i2])

    # llw_i1 = llw_base_i(i1)
    # llw_i2 = llw_base_i(i2)
    # # bin_i3 = feature_binned[i3]
    # # bin_i4 = feature_binned[i4]

    # There's still a minor discrepency between the ∇losses_∇∇losses_weights and the ∇losses,∇∇losses,weights versions but it's not here.
    bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram, bin_i1)
    loss_info_i = llw_base_i(ii)
    loss_info = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, loss_info_i)
    SIMD.vstorea(bin + loss_info, histogram, bin_i1)

    bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram, bin_i2)
    loss_info = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, loss_info_i+4)
    SIMD.vstorea(bin + loss_info, histogram, bin_i2)

    # histogram[bin_i1]     += ∇losses_∇∇losses_weights[llw_i1]
    # histogram[bin_i2]     += ∇losses_∇∇losses_weights[llw_i2]
    # # Σ∇losses[bin_i3]     += ∇losses[i3]
    # # Σ∇losses[bin_i4]     += ∇losses[i4]
    # histogram[bin_i1+1]    += ∇losses_∇∇losses_weights[llw_i1+1]
    # histogram[bin_i2+1]    += ∇losses_∇∇losses_weights[llw_i2+1]
    # # Σ∇∇losses[bin_i3]    += ∇∇losses[i3]
    # # Σ∇∇losses[bin_i4]    += ∇∇losses[i4]
    # histogram[bin_i1+2] += ∇losses_∇∇losses_weights[llw_i1+2]
    # histogram[bin_i2+2] += ∇losses_∇∇losses_weights[llw_i2+2]
    # # data_weights[bin_i3] += weights[i3]
    # # data_weights[bin_i4] += weights[i4]
  end

  ()
end

# Mutates the histogram parts: Σ∇losses, Σ∇∇losses, data_weights
# Like 2.4 GB/s 45s total with stride of 1
# Like 2.6 GBs 43s total with a stride of 2
function build_2histograms_unrolled!(X_binned, feature_i1, feature_i2, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram1, histogram2)

  # println(InteractiveUtils.@code_llvm _build_histogram_unrolled!(X_binned, size(X_binned,1)*(feature_i-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, Σ∇losses, Σ∇∇losses, data_weights))
  # throw(:crash)

  _build_2histograms_unrolled!(X_binned, size(X_binned,1)*(feature_i1-1), size(X_binned,1)*(feature_i2-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram1, histogram2)

  first_ii_unprocessed =
    leaf_ii_start + 2*length(leaf_ii_start:2:(leaf_ii_stop-1))

  # The last couple points...
  @inbounds for ii in first_ii_unprocessed:leaf_ii_stop
    i = leaf_is[ii]
    feat1_bin_i = llw_base_i(X_binned[i, feature_i1])
    feat2_bin_i = llw_base_i(X_binned[i, feature_i2])
    llw_i = llw_base_i(ii)
    histogram1[feat1_bin_i]   += ∇losses_∇∇losses_weights[llw_i]
    histogram1[feat1_bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
    histogram1[feat1_bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
    histogram2[feat2_bin_i]   += ∇losses_∇∇losses_weights[llw_i]
    histogram2[feat2_bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
    histogram2[feat2_bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
  end
end

function _build_2histograms_unrolled!(X_binned, feature_start_i1, feature_start_i2, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram1, histogram2)
  # Memory per datapoint = 4 (leaf_is) + 1  (featue_binned) + 4  (∇losses) + 4  (∇∇losses) +  4 (weights) =  17 bytes if dense
  # Memory per datapoint = 4 (leaf_is) + 64 (featue_binned) + 64 (∇losses) + 64 (∇∇losses) + 64 (weights) = 260 bytes if sparse (different cache lines)
  #
  # Conservatively assuming 1MB shared cache, that's 60,000 datapoints in cache if dense, or 4,000 if sparse

  @inbounds for ii in leaf_ii_start:2:(leaf_ii_stop-1)
    i1 = leaf_is[ii]
    i2 = leaf_is[ii+1]
    # i3 = leaf_is[ii+2]
    # i4 = leaf_is[ii+3]


    # llw_i1 = llw_base_i(i1)
    # llw_i2 = llw_base_i(i2)
    # # bin_i3 = feature_binned[i3]
    # # bin_i4 = feature_binned[i4]
    # feat3_bin_i = llw_base_i(X_binned[feature_start_i3 + i])

    loss_info_i = llw_base_i(ii)
    loss_info1 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, loss_info_i)
    loss_info2 = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, loss_info_i+4)

    feat1_bin_i1 = llw_base_i(X_binned[feature_start_i1 + i1])
    feat2_bin_i1 = llw_base_i(X_binned[feature_start_i2 + i1])

    # There's still a minor discrepency between the ∇losses_∇∇losses_weights and the ∇losses,∇∇losses,weights versions but it's not here.
    feat1_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram1, feat1_bin_i1)
    feat2_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram2, feat2_bin_i1)

    # feat3_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram3, feat3_bin_i)
    SIMD.vstorea(feat1_bin + loss_info1, histogram1, feat1_bin_i1)
    SIMD.vstorea(feat2_bin + loss_info1, histogram2, feat2_bin_i1)
    # SIMD.vstorea(feat3_bin + loss_info, histogram3, feat3_bin_i)

    feat1_bin_i2 = llw_base_i(X_binned[feature_start_i1 + i2])
    feat2_bin_i2 = llw_base_i(X_binned[feature_start_i2 + i2])

    feat1_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram1, feat1_bin_i2)
    feat2_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram2, feat2_bin_i2)
    SIMD.vstorea(feat1_bin + loss_info2, histogram1, feat1_bin_i2)
    SIMD.vstorea(feat2_bin + loss_info2, histogram2, feat2_bin_i2)
  end

  ()
end


function build_3histograms_unrolled!(X_binned, feature_i1, feature_i2, feature_i3, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, hists)

  # println(InteractiveUtils.@code_llvm _build_2histograms_unrolled!(X_binned, size(X_binned,1)*(feature_i1-1), size(X_binned,1)*(feature_i2-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram1, histogram2))
  # println(InteractiveUtils.@code_native _build_2histograms_unrolled!(X_binned, size(X_binned,1)*(feature_i1-1), size(X_binned,1)*(feature_i2-1), ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, histogram1, histogram2))
  # throw(:crash)

  _build_3histograms_unrolled!(
    X_binned,
    size(X_binned,1)*(feature_i1-1),
    size(X_binned,1)*(feature_i2-1),
    size(X_binned,1)*(feature_i3-1),
    # size(X_binned,1)*(feature_i4-1),
    # feature_i1, feature_i2, feature_i3,
    ∇losses_∇∇losses_weights,
    leaf_is, leaf_ii_start, leaf_ii_stop,
    hists
    # features_histograms
    # (features_histograms[feature_i1], features_histograms[feature_i2], features_histograms[feature_i3])
    # (features_histograms[feature_i1], features_histograms[feature_i2], features_histograms[feature_i3], features_histograms[feature_i4])
  )

  # first_ii_unprocessed =
  #   leaf_ii_start + 2*length(leaf_ii_start:2:(leaf_ii_stop-1))

  # # The last couple points...
  # @inbounds for ii in first_ii_unprocessed:leaf_ii_stop
  #   i = leaf_is[ii]
  #   feat1_bin_i = llw_base_i(X_binned[i, feature_i1])
  #   feat2_bin_i = llw_base_i(X_binned[i, feature_i2])
  #   feat3_bin_i = llw_base_i(X_binned[i, feature_i3])
  #   llw_i = llw_base_i(ii)
  #   histogram1[feat1_bin_i]   += ∇losses_∇∇losses_weights[llw_i]
  #   histogram1[feat1_bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
  #   histogram1[feat1_bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
  #   histogram2[feat2_bin_i]   += ∇losses_∇∇losses_weights[llw_i]
  #   histogram2[feat2_bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
  #   histogram2[feat2_bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
  #   histogram3[feat3_bin_i]   += ∇losses_∇∇losses_weights[llw_i]
  #   histogram3[feat3_bin_i+1] += ∇losses_∇∇losses_weights[llw_i+1]
  #   histogram3[feat3_bin_i+2] += ∇losses_∇∇losses_weights[llw_i+2]
  # end
end

function _build_3histograms_unrolled!(X_binned, feature_start_i1, feature_start_i2, feature_start_i3, ∇losses_∇∇losses_weights, leaf_is, leaf_ii_start, leaf_ii_stop, hists)
  # Memory per datapoint = 4 (leaf_is) + 1  (featue_binned) + 4  (∇losses) + 4  (∇∇losses) +  4 (weights) =  17 bytes if dense
  # Memory per datapoint = 4 (leaf_is) + 64 (featue_binned) + 64 (∇losses) + 64 (∇∇losses) + 64 (weights) = 260 bytes if sparse (different cache lines)
  #
  # Conservatively assuming 1MB shared cache, that's 60,000 datapoints in cache if dense, or 4,000 if sparse

  histogram1 = hists.hist1
  histogram2 = hists.hist2
  histogram3 = hists.hist3
  # histogram4 = hists.hist4

  @inbounds for ii in leaf_ii_start:leaf_ii_stop
    i1 = leaf_is[ii]
    # i2 = leaf_is[ii+1]
    # i3 = leaf_is[ii+2]
    # i4 = leaf_is[ii+3]


    # llw_i1 = llw_base_i(i1)
    # llw_i2 = llw_base_i(i2)
    # # bin_i3 = feature_binned[i3]
    # # bin_i4 = feature_binned[i4]
    # feat3_bin_i = llw_base_i(X_binned[feature_start_i3 + i])


    # feat1_bin_i1 = llw_base_i(X_binned[i1, feature_i1])
    # feat2_bin_i1 = llw_base_i(X_binned[i1, feature_i2])
    # feat3_bin_i1 = llw_base_i(X_binned[i1, feature_i3])
    feat1_bin_i1 = llw_base_i(X_binned[feature_start_i1 + i1])
    feat2_bin_i1 = llw_base_i(X_binned[feature_start_i2 + i1])
    feat3_bin_i1 = llw_base_i(X_binned[feature_start_i3 + i1])
    # feat4_bin_i1 = llw_base_i(X_binned[feature_start_i4 + i1])

    loss_info = SIMD.vloada(SIMD.Vec{4,Float64}, ∇losses_∇∇losses_weights, llw_base_i(ii))
    # There's still a minor discrepency between the ∇losses_∇∇losses_weights and the ∇losses,∇∇losses,weights versions but it's not here.
    # feat1_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram1, feat1_bin_i1)
    # feat2_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram2, feat2_bin_i1)
    # feat3_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram3, feat3_bin_i1)
    # feat4_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram4, feat4_bin_i1)

    # feat3_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram3, feat3_bin_i)
    SIMD.vstorea(SIMD.vloada(SIMD.Vec{4,Float64}, histogram1, feat1_bin_i1) + loss_info, histogram1, feat1_bin_i1)
    SIMD.vstorea(SIMD.vloada(SIMD.Vec{4,Float64}, histogram2, feat2_bin_i1) + loss_info, histogram2, feat2_bin_i1)
    SIMD.vstorea(SIMD.vloada(SIMD.Vec{4,Float64}, histogram3, feat3_bin_i1) + loss_info, histogram3, feat3_bin_i1)
    # SIMD.vstorea(SIMD.vloada(SIMD.Vec{4,Float64}, histogram4, feat4_bin_i1) + loss_info, histogram4, feat4_bin_i1)
    # SIMD.vstorea(feat4_bin + loss_info, features_histograms[4], feat4_bin_i1)
    # SIMD.vstorea(feat3_bin + loss_info, histogram3, feat3_bin_i)

    # feat1_bin_i2 = llw_base_i(X_binned[feature_start_i1 + i2])
    # feat2_bin_i2 = llw_base_i(X_binned[feature_start_i2 + i2])

    # feat1_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram1, feat1_bin_i2)
    # feat2_bin = SIMD.vloada(SIMD.Vec{4,Float64}, histogram2, feat2_bin_i2)
    # SIMD.vstorea(feat1_bin + loss_info2, histogram1, feat1_bin_i2)
    # SIMD.vstorea(feat2_bin + loss_info2, histogram2, feat2_bin_i2)
  end

  ()
end



# Before:

# $ sudo perf stat -B -a -d sleep 300

#  Performance counter stats for 'system wide':

#       9,599,528.20 msec cpu-clock                 #   31.998 CPUs utilized
#          3,021,958      context-switches          #    0.315 K/sec
#             42,504      cpu-migrations            #    0.004 K/sec
#         18,966,535      page-faults               #    0.002 M/sec
# 15,578,104,049,383      cycles                    #    1.623 GHz                      (40.00%)
# 14,510,053,924,294      stalled-cycles-frontend   #   93.14% frontend cycles idle     (50.00%)
# 10,431,986,116,034      stalled-cycles-backend    #   66.97% backend cycles idle      (50.00%)
# 12,889,788,964,692      instructions              #    0.83  insn per cycle
#                                                   #    1.13  stalled cycles per insn  (60.00%)
#    611,519,387,938      branches                  #   63.703 M/sec                    (60.00%)
#      3,122,439,194      branch-misses             #    0.51% of all branches          (60.00%)
#  6,463,883,611,164      L1-dcache-loads           #  673.354 M/sec                    (34.33%)
#    474,313,448,229      L1-dcache-load-misses     #    7.34% of all L1-dcache hits    (32.79%)
#    105,581,747,978      LLC-loads                 #   10.999 M/sec                    (20.00%)
#     29,421,821,984      LLC-load-misses           #   27.87% of all LL-cache hits     (30.00%)

#      300.004860572 seconds time elapsed



# Profile.jl: 48.9s (1,600,000 pts x 4000 features)
# ProfileHRRR.jl: 51.3s (734,269 pts x 9288 features)


# After:

# $ sudo perf stat -B -a -d sleep 300
#
#  Performance counter stats for 'system wide':
#
#       9,601,957.53 msec cpu-clock                 #   31.998 CPUs utilized
#          2,512,022      context-switches          #    0.262 K/sec
#             12,059      cpu-migrations            #    0.001 K/sec
#        133,044,287      page-faults               #    0.014 M/sec
# 11,970,602,809,273      cycles                    #    1.247 GHz                      (40.00%)
# 15,583,206,900,105      stalled-cycles-frontend   #  130.18% frontend cycles idle     (50.00%)
# 12,580,741,772,405      stalled-cycles-backend    #  105.10% backend cycles idle      (50.00%)
# 12,498,886,008,654      instructions              #    1.04  insn per cycle
#                                                   #    1.25  stalled cycles per insn  (60.00%)
#    670,570,842,271      branches                  #   69.837 M/sec                    (60.00%)
#      5,016,108,242      branch-misses             #    0.75% of all branches          (60.00%)
#  6,087,732,201,293      L1-dcache-loads           #  634.009 M/sec                    (33.82%)
#    311,332,643,953      L1-dcache-load-misses     #    5.11% of all L1-dcache hits    (32.97%)
#     76,864,890,325      LLC-loads                 #    8.005 M/sec                    (20.00%)
#     41,446,103,665      LLC-load-misses           #   53.92% of all LL-cache hits     (30.00%)
#
#      300.077780575 seconds time elapsed

# Hmmm, not using all threads all the time...try work stealing?

# After work stealing:

# $ sudo perf stat -B -a -d sleep 300
#
#  Performance counter stats for 'system wide':
#
#       9,600,901.95 msec cpu-clock                 #   31.998 CPUs utilized
#          2,770,768      context-switches          #    0.289 K/sec
#              8,814      cpu-migrations            #    0.001 K/sec
#         17,590,617      page-faults               #    0.002 M/sec
# 17,040,666,272,115      cycles                    #    1.775 GHz                      (38.44%)
# 14,920,973,369,568      stalled-cycles-frontend   #   87.56% frontend cycles idle     (48.44%)
#  9,007,863,366,220      stalled-cycles-backend    #   52.86% backend cycles idle      (50.00%)
# 16,201,076,017,964      instructions              #    0.95  insn per cycle
#                                                   #    0.92  stalled cycles per insn  (60.00%)
#    769,258,226,952      branches                  #   80.124 M/sec                    (60.00%)
#      5,054,113,485      branch-misses             #    0.66% of all branches          (60.00%)
#  8,178,126,773,570      L1-dcache-loads           #  851.808 M/sec                    (31.61%)
#    444,386,110,419      L1-dcache-load-misses     #    5.43% of all L1-dcache hits    (33.47%)
#     99,783,974,790      LLC-loads                 #   10.393 M/sec                    (20.00%)
#     46,342,683,050      LLC-load-misses           #   46.44% of all LL-cache hits     (28.44%)
#
#      300.050504689 seconds time elapsed
#
# Still need to get GC to run less, I think. It's single-threaded.
#

# what's the limiter?
# per feature-datapoint we must do the following if processing one feature at a time:
# 1. 1 load to get the feature (there may be a fancy way to consolidate this when the data is dense)
# 2. 1 load to get the loss info
# 3. 1 load to get the bin's loss info (dependent on step 1)
# 4. 1 store to store the bin's new loss info

# So that's 3 L1 loads per feature-datapoint
# ~2 passes over the data
# * 170,000,000,000 feature-datapoints
# * 3 loads/each
# / 2 loads per cycle (Sandy Bridge)
# / 3,100,000,000 cycles per second
# / 16 cores
# = 10 seconds

# Sandy Bridge has only one memory write port, so that is also a possible limiter.
# ~2 passes over the data
# * 170,000,000,000 feature-datapoints
# * 1 write/each
# / 1 write per cycle (Sandy Bridge)
# / 3,100,000,000 cycles per second
# / 16 cores
# = 7 seconds

# Per Agner Fog "Memory operations of 128 bits or less have a throughput of two reads or one read and one
# write per clock cycle. It is not possible to do two reads and one write per clock cycle
# because there are only two address calculation units (port 2 and 3). "
# So:
# ~2 passes over the data
# * 170,000,000,000 feature-datapoints
# * (3 reads + 1 write)/each
# / 2 mem ops per cycle (Sandy Bridge)
# / 3,100,000,000 cycles per second
# / 16 cores
# = 14 seconds

# ~2 passes over the data
# * 170GB
# / 50GB/s DRAM bandwidth
# = 7 seconds

# Agner Fog: "Each consecutive 128 bytes, or two cache lines, in the data cache is divided into 8 banks of
# 16 bytes each. It is not possible to do two memory reads in the same clock cycle if the two
# memory addresses have the same bank number, i.e. if bit 4 - 6 in the two addresses are
# the same.
# In addition, there is a false dependence between memory addresses with the same set and
# offset, i.e. with a distance that is a multiple of 4 Kbytes."


mutable struct Hists
  hist1 :: Vector{Loss}
  hist2 :: Vector{Loss}
  hist3 :: Vector{Loss}
  # hist4 :: Vector{Loss}
end

function make_a_chunk_of_histograms(X_binned, ∇losses_∇∇losses_weights, leaf_is, chunk_feature_ii_start, chunk_feature_ii_stop, is_chunk_size, hists, feature_is_to_compute, features_histograms)
  for ii in 1:is_chunk_size:length(leaf_is) # Currently: 32 features/chunk * 16 threads = reloaded 36x = 5.5GB of leaf_is,∇losses,∇∇losses,weights loading
    # for feature_ii in 1:length(chunk_feature_is_to_compute) # Currently: 8704 points/feature = histograms reloaded 1100x = 61GB of histogram loading; 448 pts/feature = histograms reloaded 22000x = 1200GB of histogram loading BUT all that should be from L3; ideally each histogram is loaded into L3 only once
    # for feature_ii in 1:2:(length(chunk_feature_is_to_compute)-1) # Currently: 8704 points/feature = histograms reloaded 1100x = 61GB of histogram loading; 448 pts/feature = histograms reloaded 22000x = 1200GB of histogram loading BUT all that should be from L3; ideally each histogram is loaded into L3 only once
    feature_ii = chunk_feature_ii_start
    while feature_ii <= chunk_feature_ii_stop

      # Always 171GB of X_binned loading (unavoidable)
      if feature_ii + 2 <= chunk_feature_ii_stop
        feature_i1 = feature_is_to_compute[feature_ii]
        feature_i2 = feature_is_to_compute[feature_ii+1]
        feature_i3 = feature_is_to_compute[feature_ii+2]
        # feature_i4 = feature_is_to_compute[feature_ii+3]
        hists.hist1 = features_histograms[feature_i1]
        hists.hist2 = features_histograms[feature_i2]
        hists.hist3 = features_histograms[feature_i3]
        # hists.hist4 = features_histograms[feature_i4]
        build_3histograms_unrolled!(X_binned, feature_i1, feature_i2, feature_i3, ∇losses_∇∇losses_weights, leaf_is, ii, min(ii+is_chunk_size-1, length(leaf_is)), hists)
        feature_ii += 3
      elseif feature_ii + 1 <= chunk_feature_ii_stop
        feature_i1 = feature_is_to_compute[feature_ii]
        feature_i2 = feature_is_to_compute[feature_ii+1]
        build_2histograms_unrolled!(X_binned, feature_i1, feature_i2, ∇losses_∇∇losses_weights, leaf_is, ii, min(ii+is_chunk_size-1, length(leaf_is)), features_histograms[feature_i1], features_histograms[feature_i2])
        feature_ii += 2
      else
        feature_i = feature_is_to_compute[feature_ii]
        build_histogram_unrolled!(X_binned, feature_i, ∇losses_∇∇losses_weights, leaf_is, ii, min(ii+is_chunk_size-1, length(leaf_is)), features_histograms[feature_i])
        feature_ii += 1
      end
    end

  end
end

function compute_histograms!(X_binned, ∇losses_∇∇losses_weights, feature_is_to_compute, features_histograms, leaf_is)

  # println((length(leaf_is), length(feature_is_to_compute)))

  # For SREF, up to 27.7mb of leaf_is, and always 57.6mb of ∇losses,∇∇losses,weights
  # For HRRR (9897675 datapoints with 18577 features each), up to 40mb leaf_is (usually much less); always 113mb ∇losses,∇∇losses,weights; always 54mb of feature histograms; 171GB of X_binned

  # Cache-optimal chunk sizes for root and others, chosen by search.
  # is_chunk_size = 8704
  # Should also choose chunk size by whether we are using all the features
  # (Prefetcher may perform better when we are)
  # Below is chosen at feature_fraction of 0.6
  is_chunk_size = isa(leaf_is, UnitRange) ? 20736 : 320
  # is_chunk_size = isa(leaf_is, UnitRange) ? 64*64*64 : 64*64*64
    # if isa(leaf_is, UnitRange)
    #   8704
    # else
    #   data_fraction = length(leaf_is) / data_count(X_binned)
    #   pts_per_∇losses_cache_line = ceil(16 * data_fraction)
    #   cache_lines = 85*1024 / 64 # Target 85k, earlier experiments pointed to 448 optimal chunk size
    #   Int64(round(pts_per_∇losses_cache_line * cache_lines / 4)) # That 85k needs to be shared over 3 arrays: ∇losses_∇∇losses_weights
    # end

  features_chunk_size = 12


  features_chunk_size = clamp(Int64(floor(length(feature_is_to_compute) / Threads.nthreads())), 1, features_chunk_size)

  # For 256kb L2, ~12,000 ≈ 192kb resident ∇losses_∇∇losses_weights, leaf_is + 12kb X_binned + 3k Σ∇losses Σ∇∇losses data_weights per feature
  # L3 more difficult to compute b/c lots of X_binned and leaf_is flowing through it

  # start_time = time_ns()

  chunks = 1:features_chunk_size:length(feature_is_to_compute)

  # We'll re-allocate these in threads to be sure they are on different cache lines.
  histss = Union{Hists,Nothing}[]
  for i in 1:Threads.nthreads()
    push!(histss, nothing)
  end

  # parallel_iterate(length(feature_is_to_compute)) do thread_range
  parallel_iterate_work_stealing(chunks) do chunk_feature_ii_start
    if isnothing(histss[Threads.threadid()])
      histss[Threads.threadid()] = Hists([],[],[])
    end
    hists = histss[Threads.threadid()]

    chunk_feature_ii_stop = min(chunk_feature_ii_start + features_chunk_size - 1, length(feature_is_to_compute))

    make_a_chunk_of_histograms(X_binned, ∇losses_∇∇losses_weights, leaf_is, chunk_feature_ii_start, chunk_feature_ii_stop, is_chunk_size, hists, feature_is_to_compute, features_histograms)

  end

  # duration = (time_ns() - start_time) / 1e9

  # bytes_dispactched = length(leaf_is)*length(feature_is_to_compute)

  # cache_line_start = leaf_is[1]
  # cache_lines_est = 1
  # for i in leaf_is
  #   if i >= cache_line_start+64
  #     cache_line_start = i
  #     cache_lines_est += 1
  #   end
  # end

  # est_memory_access = cache_lines_est*64*length(feature_is_to_compute)

  # max_memory_accessed = min(data_count(X_binned)*length(feature_is_to_compute), 64*bytes_dispactched)

  # println("$(bytes_dispactched/duration/1024/1024/1024)-$(max_memory_accessed/duration/1024/1024/1024) GB/s, likely $(est_memory_access/duration/1024/1024/1024) GB/s\t$(Float64(duration))s")

end

# Returns SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)
function find_best_split(features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score)

  thread_best_split_candidates =
    parallel_iterate(length(feature_is)) do thread_range
      # best_expected_Δloss, best_feature_i, best_split_i = (Loss(0.0), 0, UInt8(0))

      best_split_candidate    = deepcopy(dont_split)
      scratch_split_candidate = deepcopy(dont_split)

      for feature_i in view(feature_is, thread_range)
        histogram = features_histograms[feature_i]

        if isnothing(histogram)
          continue
        end

        scratch_split_candidate.feature_i = feature_i
        best_split_for_feature!(scratch_split_candidate, histogram, min_data_weight_in_leaf, l2_regularization, max_delta_score)

        if scratch_split_candidate.expected_Δloss < best_split_candidate.expected_Δloss
          best_split_candidate, scratch_split_candidate = scratch_split_candidate, best_split_candidate
        end
      end # for feature_i in feature_is

      best_split_candidate
    end

  (_, best_thread_i) = findmin(map(candidate -> candidate.expected_Δloss, thread_best_split_candidates))

  thread_best_split_candidates[best_thread_i]
end

# returns Σ∇losses, Σ∇∇losses, Σweights
function sum_histogram(histogram)
  Σ∇losses_Σ∇∇losses_Σweights_Σdummy = SIMD.Vec{4,Float64}(0)
  @inbounds for i in 1:4:length(histogram)
    Σ∇losses_Σ∇∇losses_Σweights_Σdummy += SIMD.vloada(SIMD.Vec{4,Float64}, histogram, i)
  end

  Σ∇losses, Σ∇∇losses, Σweights, Σdummy = NTuple{4,Float64}(Σ∇losses_Σ∇∇losses_Σweights_Σdummy)

  (Σ∇losses, Σ∇∇losses, Σweights)
end

# Resets and mutates scratch_split_candidate
function best_split_for_feature!(scratch_split_candidate, histogram, min_data_weight_in_leaf, l2_regularization, max_delta_score)
  best_expected_Δloss = Loss(0.0)

  scratch_split_candidate.expected_Δloss    = Loss(0.0)
  scratch_split_candidate.split_i           = UInt8(0)
  scratch_split_candidate.left_Δscore       = Score(0.0)
  scratch_split_candidate.left_data_weight  = DataWeight(0.0)
  scratch_split_candidate.right_Δscore      = Score(0.0)
  scratch_split_candidate.right_data_weight = DataWeight(0.0)

  # Σ∇losses     = llw_∇losses(histogram)
  # Σ∇∇losses    = llw_∇∇losses(histogram)
  # data_weights = llw_weights(histogram)

  this_leaf_Σ∇loss, this_leaf_Σ∇∇loss, this_leaf_data_weight = sum_histogram(histogram)

  # this_leaf_Σ∇loss      = sum(Σ∇losses)
  # this_leaf_Σ∇∇loss     = sum(Σ∇∇losses)
  # this_leaf_data_weight = sum(data_weights)

  this_leaf_expected_Δloss = leaf_expected_Δloss(this_leaf_Σ∇loss, this_leaf_Σ∇∇loss, l2_regularization, max_delta_score)

  left_Σ∇loss      = Loss(0.0)
  left_Σ∇∇loss     = Loss(0.0)
  left_data_weight = Loss(0.0)

  @inbounds for bin_i in UInt8(1):UInt8(max_bins-1)
    llw_i = llw_base_i(bin_i)
    left_Σ∇loss      += histogram[llw_i]
    left_Σ∇∇loss     += histogram[llw_i+1]
    left_data_weight += histogram[llw_i+2]

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

      scratch_split_candidate.expected_Δloss    = best_expected_Δloss
      scratch_split_candidate.split_i           = bin_i
      scratch_split_candidate.left_Δscore       = optimal_Δscore(left_Σ∇loss, left_Σ∇∇loss, l2_regularization, max_delta_score)
      scratch_split_candidate.left_data_weight  = left_data_weight
      scratch_split_candidate.right_Δscore      = optimal_Δscore(right_Σ∇loss, right_Σ∇∇loss, l2_regularization, max_delta_score)
      scratch_split_candidate.right_data_weight = right_data_weight
    end
  end # for bin_i in 1:(max_bins-1)

  ()
end

end # module MemoryConstrainedTreeBoosting
