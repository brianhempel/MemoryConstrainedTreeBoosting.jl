module MemoryConstrainedTreeBoosting

export train, train_on_binned, prepare_bin_splits, apply_bins, predict, predict_on_binned, save, load, load_unbinned_predictor, make_callback_to_track_validation_loss

import Random

import BSON


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

  # Mangling so you get a tuple of arrays.
  Tuple(collect.(zip(thread_results...)))
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

struct Const
  x
end
Base.getindex(c::Const, i::Int) = c.x


default_config = (
  weights                            = nothing, # weights for the data
  bin_count                          = 255,
  iteration_count                    = 10,
  min_data_weight_in_leaf            = 10.0,
  l2_regularization                  = 1.0,
  max_leaves                         = 32,
  max_depth                          = 6,
  max_delta_score                    = 1.0e10, # Before shrinkage.
  learning_rate                      = 0.1,
  feature_fraction                   = 1.0, # Per tree.
  bagging_temperature                = 1.0, # Same as Catboost's Bayesian bagging. 0.0 doesn't change the weights. 1.0 samples from the exponential distribution to scale each datapoint's weight.
  feature_i_to_name                  = nothing,
  iteration_callback                 = nothing, # Callback is given trees. If you want to override the default early stopping validation callback.
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
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0

const Score      = Float32
const Loss       = Float32
const Prediction = Float32
const DataWeight = Float32

const BinSplits = Vector{T} where T <: AbstractFloat

const max_bins = Int64(typemax(UInt8))

const Data = Array{UInt8,2}

data_count(X :: Array{<:Number,2}) = size(X,1)

feature_count(X :: Array{<:Number,2}) = size(X,2)

get_feature(X_binned :: Data, feature_i) = @view X_binned[:, feature_i]


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
  features_histograms :: Vector{Union{Histogram,Nothing}} # Transient. Used to speed up tree calculation.
end

mutable struct Leaf <: Tree
  Δscore                :: Score # Called "weight" in the literature
  is                                                        # Transient. Needed during tree growing.
  maybe_split_candidate :: Union{SplitCandidate,Nothing}    # Transient. Needed during tree growing.
  features_histograms   :: Vector{Union{Histogram,Nothing}} # Transient. Used to speed up tree calculation.

  Leaf(Δscore, is = nothing, maybe_split_candidate = nothing, features_histograms = []) = new(Δscore, is, maybe_split_candidate, features_histograms)
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

# Returns a JIT-able function that takes in data and returns predictions.
function load_unbinned_predictor(path)
  bin_splits, trees = load(path)

  tree_funcs = map(tree -> tree_to_function(bin_splits, tree), trees)

  predict(X) = begin
    thread_scores = map(_ -> zeros(Score, data_count(X)), 1:Threads.nthreads())

    Threads.@threads for tree_func in tree_funcs
      tree_func(X, thread_scores[Threads.threadid()])
    end

    σ.(sum(thread_scores))
  end
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
function apply_tree!(X_binned :: Data, tree :: Tree, scores :: Vector{Score}) :: Vector{Score}
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
function _apply_tree!(X_binned :: Data, fast_nodes :: Vector{FastNode}, scores :: Vector{Score}) :: Vector{Score}
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

# Returns a function tree_func(X, scores) that runs the tree on data X and mutates scores
#
# If bin_splits == nothing, assumes data is already binned
function tree_to_function(bin_splits, tree)
  eval(quote
    (X, scores) -> begin
      for i in 1:data_count(X)
        $(tree_to_exp(bin_splits, tree))
      end
    end
  end)
end

# If bin_splits == nothing, then assumes the data is already binned
function tree_to_exp(bin_splits, node :: Node)
  threshold = isnothing(bin_splits) ? node.split_i : bin_splits[node.feature_i][node.split_i]
  quote
    if X[i, $(node.feature_i)] <= $(threshold)
      $(tree_to_exp(bin_splits, node.left))
    else
      $(tree_to_exp(bin_splits, node.right))
    end
  end
end

function tree_to_exp(bin_splits, leaf :: Leaf)
  quote
    scores[i] += $(leaf.Δscore)
  end
end


# Returns vector of untransformed scores (linear, pre-sigmoid). Does not mutate starting_scores.
function apply_trees(X_binned :: Data, trees :: Vector{<:Tree}; starting_scores = nothing) :: Vector{Score}

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
  scores = apply_trees(X_binned, trees; starting_scores = starting_scores)
  if output_raw_scores
    scores
  else
    parallel_map!(σ, scores, scores)
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
    sorted_feature_values = sort(@view X[is, j])

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

function make_callback_to_track_validation_loss(validation_X_binned, validation_y; validation_weights = nothing, max_iterations_without_improvement = typemax(Int64))
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
    validation_loss = compute_mean_logloss(validation_y, validation_scores, validation_weights)

    print("\rValidation loss: $validation_loss    ")
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

    validation_loss
  end

  iteration_callback
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

# Reusable memory to avoid allocations between trees.
mutable struct ScratchMemory
  ∇losses  :: Vector{Loss}
  ∇∇losses :: Vector{Loss}
  weights  :: Union{Nothing,Vector{DataWeight}}
  is       :: Union{Vector{UInt32},Vector{Int64}}
  trues    :: Union{Vector{UInt32},Vector{Int64}} # During parallel partition
  falses   :: Union{Vector{UInt32},Vector{Int64}} # During parallel partition

  ScratchMemory(y, config) =
    new(
      Vector{Loss}(undef, length(y)),
      Vector{Loss}(undef, length(y)),
      get_config_field(config, :bagging_temperature) > 0 ? Vector{DataWeight}(undef, length(y)) : nothing,
      Vector{index_type(y)}(undef, length(y)),
      Vector{index_type(y)}(undef, length(y)),
      Vector{index_type(y)}(undef, length(y)),
    )
end

function train_on_binned(X_binned :: Data, y; prior_trees=Tree[], config...) :: Vector{Tree}
  weights = get_config_field(config, :weights)
  if weights == nothing
    weights = ones(DataWeight, length(y))
  end

  if isempty(prior_trees)
    initial_score = begin
      probability = compute_mean_probability(y, weights)
      log(probability / (1-probability)) # inverse sigmoid
    end

    prior_trees = Tree[Leaf(initial_score)]
  end

  scores = apply_trees(X_binned, prior_trees) # Linear scores, before sigmoid transform.

  trees = copy(prior_trees)

  scratch_memory = ScratchMemory(y, config)

  try
    for iteration_i in 1:get_config_field(config, :iteration_count)
      duration = @elapsed begin
        tree = train_one_iteration(X_binned, y, weights, scores; scratch_memory = scratch_memory, config...)
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
      print("$duration sec/tree     ")
    end
  catch expection
    println()
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
function train_one_iteration(X_binned, y :: Vector{Prediction}, weights :: Vector{DataWeight}, scores :: Vector{Score}; scratch_memory = nothing, config...) :: Tree
  learning_rate       = Score(get_config_field(config, :learning_rate))
  bagging_temperature = DataWeight(get_config_field(config, :bagging_temperature))

  weights =
    # Adapted from Catboost
    if bagging_temperature > 0
      bagged_weights(weights, bagging_temperature, isnothing(scratch_memory) ? nothing : scratch_memory.weights)
    else
      weights
    end

  ∇losses  = isnothing(scratch_memory) ? Vector{Loss}(undef, length(y))          : scratch_memory.∇losses
  ∇∇losses = isnothing(scratch_memory) ? Vector{Loss}(undef, length(y))          : scratch_memory.∇∇losses
  is       = isnothing(scratch_memory) ? Vector{index_type(y)}(undef, length(y)) : scratch_memory.is
  trues    = isnothing(scratch_memory) ? Vector{index_type(y)}(undef, length(y)) : scratch_memory.trues
  falses   = isnothing(scratch_memory) ? Vector{index_type(y)}(undef, length(y)) : scratch_memory.falses

  # Needs to be a separate method otherwise type inference croaks.
  compute_∇losses_∇∇losses!(y, scores, ∇losses, ∇∇losses, weights)

  tree = build_one_tree(X_binned, ∇losses, ∇∇losses, weights, is, trues, falses; config...)
  tree = scale_leaf_Δscores(tree, learning_rate)
  tree
end

function bagged_weights(weights, bagging_temperature, weights_scratch = nothing)
  out = isnothing(weights_scratch) ? Vector{DataWeight}(undef, length(weights)) : weights_scratch
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

  out
end

function build_one_tree(X_binned :: Data, ∇losses, ∇∇losses, weights, is, trues, falses; config...)
  # Use a range rather than a list for the root. Saves us having to initialize is.
  all_is = UnitRange{index_type(weights)}(1:length(weights))
  tree =
    Leaf(
      optimal_Δscore(∇losses, ∇∇losses, Loss(get_config_field(config, :l2_regularization)), Score(get_config_field(config, :max_delta_score))),
      all_is,
      nothing, # maybe_split_candidate
      []       # feature_histograms
    )

  features_to_use_count = Int64(ceil(get_config_field(config, :feature_fraction) * feature_count(X_binned)))

  # I suspect the cache benefits for sorting the indexes are trivial but it feels cleaner.
  feature_is = sort(Random.shuffle(1:feature_count(X_binned))[1:features_to_use_count])

  tree_changed = true

  while tree_changed
    # print_tree(tree)
    # println()
    # println()
    (tree_changed, tree) = perhaps_split_tree(tree, X_binned, ∇losses, ∇∇losses, weights, is, feature_is, trues, falses; config...)
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


function compute_mean_probability(y, weights)
  thread_Σlabels, thread_Σweight = parallel_iterate(length(y)) do thread_range
    Σlabel  = zero(Prediction)
    Σweight = zero(DataWeight)
    @inbounds for i in thread_range
      Σlabel  += y[i] * weights[i]
      Σweight += weights[i]
    end
    (Σlabel, Σweight)
  end
  sum(thread_Σlabels) / sum(thread_Σweight)
end

function compute_mean_logloss(y, scores, weights = nothing)
  weights = isnothing(weights) ? Const(one(DataWeight)) : weights
  _compute_mean_logloss(y, scores, weights)
end

function _compute_mean_logloss(y, scores, weights)
  # Broadcast version, which performs allocations:
  # ŷ = σ.(scores)
  # mean_logloss = sum(logloss.(y, ŷ) .* weights) / sum(weights)
  thread_Σlosses, thread_Σweights = parallel_iterate(length(y)) do thread_range
    Σloss   = zero(Loss)
    Σweight = zero(DataWeight)
    @inbounds for i in thread_range
      ŷ_i      = σ(scores[i])
      Σloss   += logloss(y[i], ŷ_i) * weights[i]
      Σweight += weights[i]
    end
    (Σloss, Σweight)
  end
  sum(thread_Σlosses) / sum(thread_Σweights)
end

# Stores results in ∇losses, ∇∇losses
function compute_∇losses_∇∇losses!(y, scores, ∇losses, ∇∇losses, weights)
  Threads.@threads for i in 1:length(y)
    @inbounds begin
      ŷ_i = σ(scores[i])
      ∇losses[i]  = (ŷ_i - y[i])        * weights[i] # ∇logloss  = ŷ - y
      ∇∇losses[i] = ŷ_i * (1.0f0 - ŷ_i) * weights[i] # ∇∇logloss = ŷ * (1.0f0 - ŷ)
    end
  end
  ()
end

# Assuming binary classification with log loss.
function optimal_Δscore(∇losses :: AbstractArray{Loss}, ∇∇losses :: AbstractArray{Loss}, l2_regularization :: Loss, max_delta_score :: Score)
  Σ∇loss, Σ∇∇loss = sum_∇loss_∇∇loss(∇losses, ∇∇losses)

  # And the loss minima is at:
  clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)
end

function sum_∇loss_∇∇loss(∇losses, ∇∇losses)
  thread_Σ∇losses, thread_Σ∇∇losses = parallel_iterate(length(∇losses)) do thread_range
    Σ∇loss  = zero(Loss)
    Σ∇∇loss = zero(Loss)
    @inbounds for i in thread_range
      Σ∇loss  += ∇losses[i]
      Σ∇∇loss += ∇∇losses[i]
    end
    (Σ∇loss, Σ∇∇loss)
  end

  (sum(thread_Σ∇losses), sum(thread_Σ∇∇losses))
end

# -0.5 * (Σ∇loss)² / (Σ∇∇loss) in XGBoost paper; but can't simplify so much if clamping the score.
function leaf_expected_Δloss(Σ∇loss :: Loss, Σ∇∇loss :: Loss, l2_regularization :: Loss, max_delta_score :: Score)
  Δscore = clamp(-Σ∇loss / (Σ∇∇loss + l2_regularization + ε), -max_delta_score, max_delta_score)

  Σ∇loss * Δscore + 0.5f0 * Σ∇∇loss * Δscore * Δscore
end

function range_chunks(n, chunk_size)
  chunks = UnitRange{Int64}[]
  for start in 1:chunk_size:n
    push!(chunks, start:min(start + chunk_size - 1, n))
  end
  chunks
end

# Mutates tree, but also returns the tree in case tree was a lone leaf.
#
# Returns (bool, tree) where bool is true if any split was made, otherwise false.
function perhaps_split_tree(tree, X_binned :: Data, ∇losses, ∇∇losses, weights, is, feature_is, trues, falses; config...)
  leaves = sort(tree_leaves(tree), by = (leaf -> length(leaf.is))) # Process smallest leaves first, should speed up histogram computation.

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
        leaf.features_histograms = map(_ -> nothing, 1:feature_count(X_binned))
      end

      # Don't really need threads on this one but it doesn't hurt.
      Threads.@threads for feature_i in feature_is
        perhaps_calculate_feature_histogram_from_parent_and_sibling!(feature_i, tree, leaf)
      end

      feature_is_to_compute = filter(feature_i -> isnothing(leaf.features_histograms[feature_i]), feature_is)

      per_thread_cache_lines    = zeros(Int64, (8, Threads.nthreads()))
      thread_sync_points        = view(per_thread_cache_lines, 5, :)

      # Threads.@threads for feature_i in feature_is_to_compute
      parallel_iterate(length(feature_is_to_compute)) do thread_range
        for feature_i in @view feature_is_to_compute[thread_range]
          leaf.features_histograms[feature_i] =
            calculate_feature_histogram(X_binned, ∇losses, ∇∇losses, weights, feature_i, leaf.is, thread_sync_points)
        end
        thread_sync_points[Threads.threadid()] = typemax(Int64)
      end

      leaf.maybe_split_candidate = find_best_split(leaf.features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score)
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

    feature_binned = get_feature(X_binned, feature_i)

    # If root node, switch from unit range to our scratch memory/
    scratch_is = isa(leaf_to_split.is, UnitRange) ? is : leaf_to_split.is

    left_leaf, right_leaf = make_split_leaves(feature_binned, ∇losses, ∇∇losses, split_i, scratch_is, trues, falses, leaf_to_split.is, l2_regularization, max_delta_score)

    new_node = Node(feature_i, split_i, left_leaf, right_leaf, leaf_to_split.features_histograms)

    tree = replace_leaf!(tree, leaf_to_split, new_node)

    (true, tree)
  else
    (false, tree)
  end
end

function make_split_leaves(feature_binned, ∇losses, ∇∇losses, split_i, scratch_is, trues, falses, leaf_is, l2_regularization, max_delta_score)
  left_is, right_is = parallel_partition!(i -> feature_binned[i] <= split_i, scratch_is, trues, falses, leaf_is)

  left_∇losses   = @view ∇losses[left_is]
  left_∇∇losses  = @view ∇∇losses[left_is]
  right_∇losses  = @view ∇losses[right_is]
  right_∇∇losses = @view ∇∇losses[right_is]

  left_Δscore  = optimal_Δscore(left_∇losses,  left_∇∇losses,  l2_regularization, max_delta_score)
  right_Δscore = optimal_Δscore(right_∇losses, right_∇∇losses, l2_regularization, max_delta_score)

  left_leaf  = Leaf(left_Δscore,  left_is,  nothing, [])
  right_leaf = Leaf(right_Δscore, right_is, nothing, [])

  (left_leaf, right_leaf)
end

# If the parent cached its histogram, and the other sibling has already done its calculation, then we can calculate our histogram by simple subtraction.
#
# Possibly mutates leaf.feature_histograms[feature_i]
function perhaps_calculate_feature_histogram_from_parent_and_sibling!(feature_i, tree, leaf)
  if !isnothing(leaf.features_histograms[feature_i])
    return ()
  end

  parent = parent_node(tree, leaf)
  if parent != nothing
    sibling = (parent.left === leaf ? parent.right : parent.left)

    if parent != nothing && !isempty(parent.features_histograms) && !isempty(sibling.features_histograms)
      parent_histogram  = parent.features_histograms[feature_i]
      sibling_histogram = sibling.features_histograms[feature_i]

      if !isnothing(parent_histogram) && !isnothing(sibling_histogram)
        histogram = Histogram(max_bins)

        histogram.Σ∇losses     .= parent_histogram.Σ∇losses     .- sibling_histogram.Σ∇losses
        histogram.Σ∇∇losses    .= parent_histogram.Σ∇∇losses    .- sibling_histogram.Σ∇∇losses
        histogram.data_weights .= parent_histogram.data_weights .- sibling_histogram.data_weights

        leaf.features_histograms[feature_i] = histogram
      end
    end
  end

  ()
end

# Mutates the histogram parts: Σ∇losses, Σ∇∇losses, data_weights
function build_histogram_unrolled!(feature_binned, ∇losses, ∇∇losses, weights, leaf_is, Σ∇losses, Σ∇∇losses, data_weights)

  _build_histogram_unrolled!(feature_binned, ∇losses, ∇∇losses, weights, leaf_is, Σ∇losses, Σ∇∇losses, data_weights)

  # The last couple points...
  @inbounds for ii in ((1:2:(length(leaf_is)-1)).stop + 2):length(leaf_is)
    i = leaf_is[ii]
    bin_i = feature_binned[i]
    Σ∇losses[bin_i]     += ∇losses[i]
    Σ∇∇losses[bin_i]    += ∇∇losses[i]
    data_weights[bin_i] += weights[i]
  end
end

function _build_histogram_unrolled!(feature_binned, ∇losses, ∇∇losses, weights, leaf_is, Σ∇losses, Σ∇∇losses, data_weights)
  # thread_id = Threads.threadid()

  # Memory per datapoint = 4 (leaf_is) + 1  (featue_binned) + 4  (∇losses) + 4  (∇∇losses) +  4 (weights) =  17 bytes if dense
  # Memory per datapoint = 4 (leaf_is) + 64 (featue_binned) + 64 (∇losses) + 64 (∇∇losses) + 64 (weights) = 260 bytes if sparse (different cache lines)
  #
  # Conservatively assuming 1MB shared cache, that's 60,000 datapoints in cache if dense, or 4,000 if sparse
  #
  # Measured drift (MacOS, 4 cores) is 1 datapoint drift for every 6 datapoints, which means we want to
  # sync at most every 360,000 points (dense) or 24,000 points (sparse)

  @inbounds for ii in 1:2:(length(leaf_is)-1)
    i1 = leaf_is[ii]
    i2 = leaf_is[ii+1]
    # i3 = leaf_is[ii+2]
    # i4 = leaf_is[ii+3]

    bin_i1 = feature_binned[i1]
    bin_i2 = feature_binned[i2]
    # bin_i3 = feature_binned[i3]
    # bin_i4 = feature_binned[i4]

    Σ∇losses[bin_i1]     += ∇losses[i1]
    Σ∇losses[bin_i2]     += ∇losses[i2]
    # Σ∇losses[bin_i3]     += ∇losses[i3]
    # Σ∇losses[bin_i4]     += ∇losses[i4]
    Σ∇∇losses[bin_i1]    += ∇∇losses[i1]
    Σ∇∇losses[bin_i2]    += ∇∇losses[i2]
    # Σ∇∇losses[bin_i3]    += ∇∇losses[i3]
    # Σ∇∇losses[bin_i4]    += ∇∇losses[i4]
    data_weights[bin_i1] += weights[i1]
    data_weights[bin_i2] += weights[i2]
    # data_weights[bin_i3] += weights[i3]
    # data_weights[bin_i4] += weights[i4]

    # thread_progresses[thread_id] += 2
  end

  ()
end

function sync(thread_sync_points)
  thread_id = Threads.threadid()
  thread_sync_points[thread_id] += 1
  my_point = thread_sync_points[thread_id]
  for thread_i in 1:Threads.nthreads()
    thread_i == thread_id && continue
    while thread_sync_points[thread_i] < my_point
      Libc.systemsleep(0.00001)
    end
  end
  ()
end

# Calculates and returns the histogram for feature_i over leaf_is
function calculate_feature_histogram(X_binned :: Data, ∇losses, ∇∇losses, weights, feature_i, leaf_is, thread_sync_points)

  histogram = Histogram(max_bins)

  feature_binned = get_feature(X_binned, feature_i)

  Σ∇losses     = histogram.Σ∇losses
  Σ∇∇losses    = histogram.Σ∇∇losses
  data_weights = histogram.data_weights

  # # Synchronize start to see if we can get some L3 cache locality benefits for looking at the same peices of ∇losses, ∇∇losses, and weights
  # my_features_completed = thread_features_completed[Threads.threadid()]
  # while any(features_completed -> features_completed < my_features_completed, thread_features_completed)
  # end

  sync_interval = 100_000

  for start in 1:sync_interval:length(leaf_is)
    build_histogram_unrolled!(feature_binned, ∇losses, ∇∇losses, weights, view(leaf_is, start:min(start + sync_interval - 1, length(leaf_is))), Σ∇losses, Σ∇∇losses, data_weights)
    sync(thread_sync_points)
  end

  # thread_race[Threads.threadid()] = thread_progresses[Threads.threadid()] - minimum(thread_progresses)

  # thread_features_completed[Threads.threadid()] += 1

  histogram
end

# Returns SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)
function find_best_split(features_histograms, feature_is, min_data_weight_in_leaf, l2_regularization, max_delta_score)
  best_expected_Δloss, best_feature_i, best_split_i = (Loss(0.0), 0, UInt8(0))

  # This is fast enough that threading won't help. It's only O(max_bins*feature_count).
  # And it's already ridiculously fast.

  for feature_i in feature_is
    histogram = features_histograms[feature_i]

    if isnothing(histogram)
      continue
    end

    expected_Δloss, split_i = best_split_for_feature(histogram, min_data_weight_in_leaf, l2_regularization, max_delta_score)

    if expected_Δloss < best_expected_Δloss
      best_expected_Δloss = expected_Δloss
      best_feature_i      = feature_i
      best_split_i        = split_i
    end
  end # for feature_i in feature_is

  SplitCandidate(best_expected_Δloss, best_feature_i, best_split_i)
end

# Returns (expected_Δloss, best_split_i)
function best_split_for_feature(histogram, min_data_weight_in_leaf, l2_regularization, max_delta_score)
  best_expected_Δloss, best_split_i = (Loss(0.0), UInt8(0))

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

    if expected_Δloss < best_expected_Δloss
      best_expected_Δloss = expected_Δloss
      best_split_i        = bin_i
    end
  end # for bin_i in 1:(max_bins-1)

  (best_expected_Δloss, best_split_i)
end

end # module MemoryConstrainedTreeBoosting
