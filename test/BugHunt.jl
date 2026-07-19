push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Property-based / differential tests hunting for bugs.
# Run with: JULIA_NUM_THREADS=4 julia --project=. test/BugHunt.jl

using Test
import Random
using MemoryConstrainedTreeBoosting

const MCTB = MemoryConstrainedTreeBoosting

const Loss       = MCTB.Loss
const Score      = MCTB.Score
const DataWeight = MCTB.DataWeight
const max_bins   = MCTB.max_bins

rng = Random.MersenneTwister(20260712)

histogram_capacity = 4*max_bins + Int(64/sizeof(Loss))
new_histogram() = fill!(resize!(Vector{Loss}(undef, histogram_capacity), 4*max_bins), Loss(0))
new_scratch_accss() = [[Vector{Float64}(undef, 4*max_bins + Int(64/sizeof(Float64))) for _ in 1:MCTB.features_max_chunk_size] for _ in 1:Threads.nthreads()]

# Reference: bin_i = index of first split > value, or bin_count if none.
ref_bin(splits, value) = searchsortedlast(splits, value) + 1

function walk_tree(tree, x_binned_row)
  while isa(tree, MCTB.Node)
    tree = x_binned_row[tree.feature_i] <= tree.split_i ? tree.left : tree.right
  end
  tree
end

tree_depth(tree) = isa(tree, MCTB.Leaf) ? 1 : 1 + max(tree_depth(tree.left), tree_depth(tree.right))

println("Testing with $(Threads.nthreads()) thread(s)")

@testset "MemoryConstrainedTreeBoosting bug hunt" begin

@testset "apply_bins matches reference binning" begin
  for bin_count in (2, 3, 4, 5, 17, 100, 254, 255)
    split_count = bin_count - 1
    # Feature 1: strictly increasing splits. Feature 2: splits with duplicates (skewed data).
    splits1 = sort(rand(rng, Float32, split_count))
    splits2 = sort(rand(rng, Float32[0.25f0, 0.5f0, 0.75f0], split_count))
    bin_splits = [splits1, splits2]

    # Values: random, exact split values, neighbors of splits, extremes.
    values = Float32[rand(rng, Float32, 50);
                     splits1; splits2;
                     nextfloat.(splits1); prevfloat.(splits1);
                     -1f6; 1f6; 0f0; 1f0]
    X = hcat(values, values)

    X_binned = apply_bins(X, bin_splits)

    @test all(1 .<= X_binned .<= bin_count)
    for i in 1:length(values)
      @test Int(X_binned[i, 1]) == ref_bin(splits1, values[i])
      @test Int(X_binned[i, 2]) == ref_bin(splits2, values[i])
    end
  end
end

@testset "prepare_bin_splits" begin
  @test_throws Exception prepare_bin_splits(rand(rng, Float32, 100, 2); bin_count = 1)
  @test_throws Exception prepare_bin_splits(rand(rng, Float32, 100, 2); bin_count = 256)

  # Uniform data: splits should be sorted, correct count, and produce all bins.
  X = rand(rng, Float32, 100_000, 2)
  for bin_count in (2, 10, 255)
    bin_splits = prepare_bin_splits(X; bin_count = bin_count)
    @test length(bin_splits) == 2
    for splits in bin_splits
      @test length(splits) == bin_count - 1
      @test issorted(splits)
    end
    X_binned = apply_bins(X, bin_splits)
    @test sort(unique(X_binned[:,1])) == UInt8.(1:bin_count) # roughly equal-frequency bins on uniform data must hit every bin
  end

  # Constant feature: must not crash; everything lands in one bin.
  X_const = hcat(fill(3.25f0, 1000), rand(rng, Float32, 1000))
  bin_splits = prepare_bin_splits(X_const; bin_count = 10)
  X_const_binned = apply_bins(X_const, bin_splits)
  @test length(unique(X_const_binned[:,1])) == 1

  # Tiny datasets (fewer points than bins) must not crash.
  for n in (1, 2, 3, 7)
    Xn = rand(rng, Float32, n, 2)
    bs = prepare_bin_splits(Xn; bin_count = 255)
    @test all(issorted, bs)
    @test all(1 .<= apply_bins(Xn, bs) .<= 255)
  end

  # Float64 features work and preserve type.
  bs64 = prepare_bin_splits(rand(rng, Float64, 1000, 2); bin_count = 5)
  @test eltype(bs64[1]) == Float64
end

@testset "partition! and parallel_partition! match filter" begin
  for n in (0, 1, 2, 3, 7, 8, 63, 100, 1000)
    for f in (iseven, x -> true, x -> false, x -> x <= UInt32(n ÷ 3))
      in_arr = UInt32.(Random.shuffle(rng, 1:max(n,0)))[1:n]
      trues  = Vector{UInt32}(undef, n)
      falses = Vector{UInt32}(undef, n)
      t, fs = MCTB.partition!(f, trues, falses, copy(in_arr))
      @test t  == filter(f, in_arr)
      @test fs == filter(!f, in_arr)

      out    = Vector{UInt32}(undef, n)
      trues2  = Vector{UInt32}(undef, n)
      falses2 = Vector{UInt32}(undef, n)
      t2, f2 = MCTB.parallel_partition!(f, out, trues2, falses2, copy(in_arr))
      @test t2 == filter(f, in_arr)
      @test f2 == filter(!f, in_arr)

      # in-place: out === in
      in_copy = copy(in_arr)
      t3, f3 = MCTB.parallel_partition!(f, in_copy, trues2, falses2, in_copy)
      @test t3 == filter(f, in_arr)
      @test f3 == filter(!f, in_arr)
    end
  end
end

@testset "consolidate_∇losses_∇∇losses_weights! gathers correct rows" begin
  for n in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100, 1001)
    llw = rand(rng, Loss, 4*n)
    for leaf_count in unique(filter(c -> c <= n, [0, 1, 2, 3, 4, 5, n ÷ 2, n]))
      leaf_is = UInt32.(sort(Random.shuffle(rng, 1:n)[1:leaf_count]))
      out = fill(Loss(-999), 4*max(leaf_count, 1))
      MCTB.consolidate_∇losses_∇∇losses_weights!(llw, leaf_is, out)
      for (ii, i) in enumerate(leaf_is)
        @test out[4ii-3 : 4ii] == llw[4i-3 : 4i]
      end
    end
  end
end

# Float64 reference histogram builder
function naive_histograms(X_binned, llw_consolidated, leaf_is, feature_is)
  hists = Dict(f => zeros(Float64, 4*max_bins) for f in feature_is)
  for (ii, i) in enumerate(leaf_is)
    for f in feature_is
      base = 4*Int(X_binned[i, f]) - 3
      hists[f][base]   += llw_consolidated[4ii-3]
      hists[f][base+1] += llw_consolidated[4ii-2]
      hists[f][base+2] += llw_consolidated[4ii-1]
    end
  end
  hists
end

function check_histograms_against_naive(X_binned, leaf_is, feature_is; rtol = 1e-4, atol = 1f-4)
  llw = rand(rng, Loss, 4*length(leaf_is)) .- 0.3f0
  features_histograms = Vector{Union{MCTB.Histogram,Nothing}}(nothing, size(X_binned,2))
  for f in feature_is
    features_histograms[f] = new_histogram()
  end
  MCTB.compute_histograms!(X_binned, llw, feature_is, features_histograms, new_scratch_accss(), leaf_is)
  ref = naive_histograms(X_binned, llw, leaf_is, feature_is)
  # Slots ≡ 0 mod 4 are the unused SIMD "dummy" lane; the library never reads them.
  real_slots = filter(k -> k % 4 != 0, 1:4*max_bins)
  for f in feature_is
    computed = features_histograms[f]
    expected = ref[f]
    ok = all(isapprox(computed[k], expected[k]; rtol = rtol, atol = atol) for k in real_slots)
    @test ok
    if !ok
      bad = findfirst(k -> !isapprox(computed[k], expected[k]; rtol = rtol, atol = atol), real_slots)
      k = real_slots[bad]
      println("  histogram mismatch: n=$(length(leaf_is)) feature=$f slot=$k computed=$(computed[k]) expected=$(expected[k])")
    end
  end
end

@testset "compute_histograms! matches naive reference" begin
  n_total = 200
  n_features = 13 # enough to hit the 3-feature, 2-feature, and 1-feature unrolled paths
  X_binned = rand(rng, UInt8(1):UInt8(255), n_total, n_features)

  # Small leaf sizes hit all the unroll-remainder paths.
  for leaf_count in 0:9
    leaf_is = UInt32.(sort(Random.shuffle(rng, 1:n_total)[1:leaf_count]))
    check_histograms_against_naive(X_binned, leaf_is, collect(1:n_features))
  end

  # UnitRange (root leaf) path.
  check_histograms_against_naive(X_binned, UInt32(1):UInt32(n_total), collect(1:n_features))

  # Odd feature counts, odd subsets.
  for fcount in 1:7
    leaf_is = UInt32.(sort(Random.shuffle(rng, 1:n_total)[1:97]))
    feature_is = sort(Random.shuffle(rng, 1:n_features)[1:fcount])
    check_histograms_against_naive(X_binned, leaf_is, feature_is)
  end

  # Large enough to cross the rows_per_acc_chunk=20000 Float64-accumulator boundary.
  n_big = 50_011 # odd on purpose
  X_big = rand(rng, UInt8(1):UInt8(255), n_big, 4)
  check_histograms_against_naive(X_big, UInt32(1):UInt32(n_big), collect(1:4); atol = 2f-3)
  leaf_is_big = UInt32.(sort(Random.shuffle(rng, 1:n_big)[1:30_001]))
  check_histograms_against_naive(X_big, leaf_is_big, collect(1:4); atol = 2f-3)
end

@testset "make_a_chunk_of_histograms with tiny acc chunks" begin
  # Exercise the (recently modified) rows_per_acc_chunk keyword: frequent Float64 dumps
  # must give the same answer as one big pass.
  n = 1000
  n_features = 5
  X_binned = rand(rng, UInt8(1):UInt8(255), n, n_features)
  llw = rand(rng, Loss, 4*n) .- 0.3f0
  leaf_is = UInt32.(sort(Random.shuffle(rng, 1:n)[1:n÷2]))
  llw_consolidated = zeros(Loss, 4*length(leaf_is))
  MCTB.consolidate_∇losses_∇∇losses_weights!(llw, leaf_is, llw_consolidated)

  ref = naive_histograms(X_binned, llw_consolidated, leaf_is, 1:n_features)

  for (is_chunk_size, rows_per_acc_chunk) in ((3, 7), (320, 7), (17, 100), (1000, 20000))
    features_histograms = Vector{Union{MCTB.Histogram,Nothing}}(nothing, n_features)
    for f in 1:n_features
      features_histograms[f] = new_histogram()
    end
    hists = MCTB.Hists(Loss[], Loss[], Loss[])
    accs = [Vector{Float64}(undef, 4*max_bins + 8) for _ in 1:MCTB.features_max_chunk_size]
    MCTB.make_a_chunk_of_histograms(X_binned, llw_consolidated, leaf_is, 1, n_features, is_chunk_size, hists, collect(1:n_features), features_histograms, accs; rows_per_acc_chunk = rows_per_acc_chunk)
    real_slots = filter(k -> k % 4 != 0, 1:4*max_bins) # ≡ 0 mod 4 is the unused SIMD dummy lane
    for f in 1:n_features
      @test all(isapprox(features_histograms[f][k], ref[f][k]; rtol = 1e-4, atol = 1f-4) for k in real_slots)
    end
  end
end

@testset "logloss and score math" begin
  @test MCTB.logloss(0f0, 0f0) == 0f0
  @test MCTB.logloss(1f0, 1f0) == 0f0
  for y in (0f0, 1f0), ŷ in 0f0:0.05f0:1f0
    @test MCTB.logloss(y, ŷ) >= 0f0
  end
  @test MCTB.σ(0f0) ≈ 0.5f0
  @test MCTB.σ(1f6) ≈ 1f0
  @test MCTB.σ(-1f6) ≈ 0f0

  for _ in 1:1000
    Σ∇  = (rand(rng, Loss) - 0.5f0) * 100f0
    Σ∇∇ = rand(rng, Loss) * 100f0
    λ   = rand(rng, Loss) * 2f0
    cap = rand(rng, (0.5f0, 1f10))
    Δ = MCTB.optimal_Δscore(Σ∇, Σ∇∇, λ, cap)
    @test abs(Δ) <= cap
    # expected Δloss of taking the optimal step is never positive
    @test MCTB.leaf_expected_Δloss(Σ∇, Σ∇∇, λ, cap) <= 1f-6
  end
end

@testset "expected_Δloss_for_feature agrees with best_split_for_feature" begin
  for _ in 1:20
    hist = new_histogram()
    hist[1:4*max_bins] = rand(rng, Loss, 4*max_bins) .- 0.3f0
    hist[3:4:end] = rand(rng, Loss, max_bins) .* 10f0 # weights positive
    features_histograms = Union{MCTB.Histogram,Nothing}[hist]
    min_w = 5f0
    λ = 1f0
    cap = 1f10
    Δloss = MCTB.expected_Δloss_for_feature(hist, min_w, λ, cap)
    candidate = MCTB.best_split_for_feature(1, features_histograms, min_w, λ, cap)
    @test candidate.expected_Δloss ≈ Δloss
    if candidate.split_i > 0
      # data weights of the two sides must sum to the leaf total and respect the minimum
      total_w = sum(hist[3:4:end])
      @test candidate.left_data_weight + candidate.right_data_weight ≈ total_w  rtol=1e-3
      @test candidate.left_data_weight  >= min_w
      @test candidate.right_data_weight >= min_w
    end
  end
end

@testset "mpi helpers (non-MPI paths)" begin
  @test MCTB.mpi_max(nothing, 42) == 42
  @test MCTB.mpi_mean(nothing, 10f0, 4f0) == 2.5f0
  @test MCTB.mpi_compute_on_one_and_share(() -> 7, nothing) == 7
end

@testset "mean logloss / probability with tiny inputs vs threads" begin
  # counts smaller than nthreads must still work
  y = Float32[1]
  scores = Float32[0]
  @test MCTB.compute_mean_logloss(y, scores) ≈ MCTB.logloss(1f0, 0.5f0)
  @test MCTB.compute_mean_logloss(y, scores; weights = Float32[2]) ≈ MCTB.logloss(1f0, 0.5f0)
  @test MCTB.compute_mean_probability(Float32[1, 0, 1, 1], Float32[1, 1, 1, 1]) ≈ 0.75f0
  @test MCTB.compute_mean_probability(Float32[1, 0], Float32[3, 1]) ≈ 0.75f0
end

# ---------- End-to-end training properties ----------

function make_dataset(rng, n)
  X = rand(rng, Float32, n, 5)
  # Feature 1 strong, feature 4 weak, feature 5 = pure copy of label (for exclude tests)
  y = Float32.((X[:,1] .> 0.5f0) .⊻ (X[:,4] .> 0.9f0))
  X[:,5] = y
  (X, y)
end

@testset "training learns, respects constraints" begin
  X, y = make_dataset(rng, 4000)

  config = (
    bin_count = 32,
    iteration_count = 60,
    min_data_weight_in_leaf = 40.0,
    max_leaves = 8,
    max_depth = 3,
    learning_rate = 0.3,
    bagging_temperature = 0.0,
    exclude_features = [5],
  )

  bin_splits, trees = train(X, y; config...)

  X_binned = apply_bins(X, bin_splits)
  ŷ = predict_on_binned(X_binned, trees)
  accuracy = sum((ŷ .> 0.5f0) .== (y .> 0.5f0)) / length(y)
  @test accuracy > 0.95

  for tree in trees
    leaves = MCTB.tree_leaves(tree)
    @test length(leaves) <= config.max_leaves
    @test tree_depth(tree) <= config.max_depth
    for node in MCTB.tree_split_nodes(tree)
      @test node.feature_i != 5 # excluded feature must never be used
    end
    # every leaf of a split tree must hold at least min_data_weight_in_leaf points (weights = 1, bagging off)
    if length(leaves) > 1
      leaf_counts = Dict{Any,Int}()
      for i in 1:size(X_binned, 1)
        leaf = walk_tree(tree, @view X_binned[i, :])
        leaf_counts[leaf] = get(leaf_counts, leaf, 0) + 1
      end
      @test length(leaf_counts) == length(leaves) # no dead leaves
      @test all(c -> c >= config.min_data_weight_in_leaf, values(leaf_counts))
    end
  end

  # sanity check of the exclude test: without exclusion feature 5 gets used
  _, trees_no_exclude = train(X, y; config..., exclude_features = [])
  @test any(node.feature_i == 5 for tree in trees_no_exclude for node in MCTB.tree_split_nodes(tree))

  # training loss decreases as trees are added (bagging off, so greedy should be monotone-ish)
  losses = map(k -> MCTB.compute_mean_logloss(y, predict_on_binned(X_binned, trees[1:k], output_raw_scores = true)), 1:length(trees))
  @test losses[end] < losses[1] * 0.5
  @test issorted(losses, rev = true) # each greedy tree should not hurt training loss

  # determinism: same inputs, same model
  bin_splits2, trees2 = train(X, y; config...)
  @test predict_on_binned(X_binned, trees2) == ŷ
end

@testset "binned and unbinned predictors agree (save/load round trip)" begin
  X, y = make_dataset(rng, 2000)
  bin_splits, trees = train(X, y; bin_count = 16, iteration_count = 30, max_depth = 4, min_data_weight_in_leaf = 10.0, learning_rate = 0.3)

  path = tempname()
  save(path, bin_splits, trees)
  bin_splits2, trees2 = load(path)
  @test bin_splits2 == bin_splits
  @test repr(MCTB.strip_tree_training_info.(trees)) == repr(trees2)

  X_test = rand(rng, Float32, 500, 5)
  # include values exactly on bin boundaries to stress < vs <= consistency
  for j in 1:5, k in 1:min(15, length(bin_splits[j]))
    X_test[k, j] = bin_splits[j][k]
  end

  binned_ŷ   = predict(X_test, bin_splits2, trees2)
  unbinned_ŷ = load_unbinned_predictor(path)(X_test)
  @test maximum(abs.(binned_ŷ - unbinned_ŷ)) < 1e-5
end

@testset "weights: duplicating a row ≈ doubling its weight" begin
  n = 1000
  X, y = make_dataset(rng, n)
  weights = ones(Float32, n)
  weights[1:100] .= 2f0

  config = (bin_count = 16, iteration_count = 20, min_data_weight_in_leaf = 20.0, learning_rate = 0.3, bagging_temperature = 0.0)

  bin_splits = prepare_bin_splits(X, bin_count = 16)
  _, trees_w = train(X, y; bin_splits = bin_splits, weights = weights, config...)

  X_dup = vcat(X, X[1:100, :])
  y_dup = vcat(y, y[1:100])
  _, trees_dup = train(X_dup, y_dup; bin_splits = bin_splits, config...)

  ŷ_w   = predict(X, bin_splits, trees_w)
  ŷ_dup = predict(X, bin_splits, trees_dup)
  @test maximum(abs.(ŷ_w - ŷ_dup)) < 0.02
end

@testset "validation early stopping" begin
  n = 400
  X = rand(rng, Float32, n, 3)
  y = Float32.(rand(rng, Bool, n)) # pure noise: validation loss cannot keep improving
  X_val = rand(rng, Float32, n, 3)
  y_val = Float32.(rand(rng, Bool, n))

  miw = 5
  bin_splits, trees = train(X, y;
    bin_count = 8,
    iteration_count = 500,
    min_data_weight_in_leaf = 5.0,
    learning_rate = 0.5,
    bagging_temperature = 0.0,
    validation_X = X_val,
    validation_y = y_val,
    max_iterations_without_improvement = miw,
  )

  @test length(trees) < 501 # must have stopped early

  # The trees kept should be the best prefix. KNOWN BUG (see report): the callback
  # starts with best_loss = Inf and never scores the model *before* the first new tree,
  # so the bare prior model (prefix length 1) can never win even when every added tree
  # hurts validation loss. We therefore only check prefixes of length >= 2 here.
  X_val_binned = apply_bins(X_val, bin_splits)
  kept_loss = MCTB.compute_mean_logloss(y_val, predict_on_binned(X_val_binned, trees, output_raw_scores = true))
  losses = map(k -> MCTB.compute_mean_logloss(y_val, predict_on_binned(X_val_binned, trees[1:k], output_raw_scores = true)), 1:length(trees))
  @test kept_loss ≈ minimum(losses[2:end])  atol=1f-5
  @test_broken kept_loss <= minimum(losses) + 1f-6 # fails: the harmful first tree is always kept
end

@testset "prior_trees continuation and starting_scores" begin
  X, y = make_dataset(rng, 1000)
  bin_splits = prepare_bin_splits(X, bin_count = 16)
  config = (bin_count = 16, iteration_count = 10, min_data_weight_in_leaf = 20.0, learning_rate = 0.3, bagging_temperature = 0.0)

  _, trees10 = train(X, y; bin_splits = bin_splits, config...)
  _, trees20 = train(X, y; bin_splits = bin_splits, prior_trees = copy(trees10), config...)
  @test length(trees20) > length(trees10)

  X_binned = apply_bins(X, bin_splits)
  loss10 = MCTB.compute_mean_logloss(y, predict_on_binned(X_binned, trees10, output_raw_scores = true))
  loss20 = MCTB.compute_mean_logloss(y, predict_on_binned(X_binned, trees20, output_raw_scores = true))
  @test loss20 < loss10

  # predict with starting_scores == predicting with the trees that produced them
  scores10 = predict_on_binned(X_binned, trees10, output_raw_scores = true)
  rest = trees20[length(trees10)+1 : end]
  @test predict_on_binned(X_binned, rest, starting_scores = scores10) ≈ predict_on_binned(X_binned, trees20)  atol=1e-6
end

@testset "degenerate inputs" begin
  # All labels identical: must not crash, predictions must be extreme
  X = rand(rng, Float32, 100, 2)
  y1 = ones(Float32, 100)
  bin_splits, trees = train(X, y1; bin_count = 4, iteration_count = 3, bagging_temperature = 0.0)
  @test all(predict(X, bin_splits, trees) .> 0.99f0)

  # Fewer data points than threads
  X_tiny = rand(rng, Float32, 3, 2)
  y_tiny = Float32[0, 1, 0]
  bin_splits, trees = train(X_tiny, y_tiny; bin_count = 2, iteration_count = 2, bagging_temperature = 0.0)
  @test length(predict(X_tiny, bin_splits, trees)) == 3

  # Single feature
  X1 = reshape(rand(rng, Float32, 500), (500, 1))
  y1 = Float32.(X1[:,1] .> 0.5f0)
  bin_splits, trees = train(X1, y1; bin_count = 8, iteration_count = 20, min_data_weight_in_leaf = 10.0, learning_rate = 0.5, bagging_temperature = 0.0)
  ŷ = predict(X1, bin_splits, trees)
  @test sum((ŷ .> 0.5f0) .== (y1 .> 0.5f0)) / 500 > 0.9 # 8-bin quantization can't place the 0.5 boundary exactly

  # feature_fraction and second_opinion smoke tests
  X, y = make_dataset(rng, 1000)
  for extra in ((feature_fraction = 0.5,), (second_opinion_weight = 1.0, normalize_second_opinion = true), (bagging_temperature = 1.0,))
    bs, ts = train(X, y; bin_count = 8, iteration_count = 10, min_data_weight_in_leaf = 10.0, extra...)
    @test length(ts) == 11
    @test all(isfinite, predict(X, bs, ts))
  end

  # Float64 pipeline end-to-end
  X64 = rand(rng, Float64, 500, 3)
  y64 = Float32.(X64[:,1] .> 0.5)
  bs64, ts64 = train(X64, y64; bin_count = 8, iteration_count = 10, min_data_weight_in_leaf = 10.0, learning_rate = 0.5, bagging_temperature = 0.0)
  path = tempname()
  save(path, bs64, ts64)
  bs64b, ts64b = load(path)
  @test eltype(bs64b[1]) == Float64
  @test predict(X64, bs64b, ts64b) == predict(X64, bs64, ts64)
end

@testset "tree utilities" begin
  leaf_l  = MCTB.Leaf(1f0)
  leaf_r  = MCTB.Leaf(2f0)
  leaf_rr = MCTB.Leaf(3f0)
  right   = MCTB.Node(2, 0x05, leaf_r, leaf_rr, [])
  root    = MCTB.Node(1, 0x03, leaf_l, right, [])

  @test MCTB.parent_node(root, leaf_l)  === root
  @test MCTB.parent_node(root, leaf_rr) === right
  @test MCTB.sibling_node(root, leaf_r) === leaf_rr
  @test MCTB.sibling_node(root, right)  === leaf_l
  @test MCTB.sibling_node(root, root)   === nothing
  @test MCTB.leaf_depth(root, leaf_l)  == 2
  @test MCTB.leaf_depth(root, leaf_rr) == 3
  @test MCTB.leaf_depth(root, MCTB.Leaf(9f0)) === nothing
  @test length(MCTB.tree_leaves(root)) == 3
  @test length(MCTB.tree_split_nodes(root)) == 2

  # apply_tree via FastNodes matches a plain walk
  X_binned = rand(rng, UInt8(1):UInt8(8), 100, 2)
  scores = MCTB.apply_tree(X_binned, root)
  for i in 1:100
    @test scores[i] == walk_tree(root, @view X_binned[i, :]).Δscore
  end

  importance = MCTB.feature_importance_by_appearance_count(MCTB.Tree[root])
  @test Dict(importance) == Dict(1 => 1, 2 => 1)
end

end # top-level testset

println("Done.")
