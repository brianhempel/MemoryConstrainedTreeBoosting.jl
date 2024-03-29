# MemoryConstrainedTreeBoosting.jl

Gradient-boosted decision trees when you want to use all the memory on your machine. Also supports distributed learning if your data doesn't fit on one machine. Also, it's quite fast.

For input, only `Float32` data is supported right now, unless you perform your own initial binning down to `UInt8`. Note if you do so: because of Julia's indexing, bins are 1-255, not 0-255. (You can use fewer than 255 bins, but start at 1, not 0.)

Only binary classification with logloss is supported right now. If you need to, you can probably fork the repo transform that to single variable regression pretty easily. (Multivariate loss is harder.)

Powers [nadocast.com](http://nadocast.com).

## Quickstart

The package isn't published yet—you'll have to [point to Github](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages).

```
(@v1.6) pkg> add https://github.com/brianhempel/MemoryConstrainedTreeBoosting.jl
```

If you aren't memory constrained, you can just call `train`:

```julia
using MemoryConstrainedTreeBoosting

# 100 datapoints, 10 features each. Super small.
X = rand(Float32, 100, 10)
y = Float32.(rand(Bool, 100))

# Training:
bin_splits, trees = train(X, y); # Only X and y are required.
# OR
bin_splits, trees =
  train(
    X, y;
    # Defaults below:
    weights                  = nothing,
    bin_count                = 255,
    iteration_count          = 100,
    min_data_weight_in_leaf  = 9.9, # Because of numeric instability, use a number slightly lower that what you mean (10.0)
    l2_regularization        = 1.0,
    max_leaves               = 32,
    max_depth                = 6,
    max_delta_score          = 1.0e10, # Before shrinkage.
    learning_rate            = 0.03,
    feature_fraction         = 1.0, # Per tree.
    second_opinion_weight    = 0.0, # 0.0 = no second opinion. 1.0 = look at expected gains for sibling when choosing feature splits, choose feature that maximizes gains for both siblings. Inspired by Catboost choosing same split and feature across an entire level, so-called "oblivious" decision trees. But we are not going so far as to choose the same split point.
    normalize_second_opinion = false, # true = make the best expected gain on the sibling match the leaf's best expected gain before applying the second opinion weight (in case of highly imbalanced nodes, this makes the leaf with more data count less)
    exclude_features         = [], # Indices. Applied before feature_fraction
    bagging_temperature      = 1.0, # Same as Catboost's Bayesian bagging. 0.0 doesn't change the weights. 1.0 samples from the exponential distribution to scale each datapoint's weight.
    iteration_callback                 = nothing, # Optional. Callback is given trees. If you want to override the default early stopping validation callback which is auto-generated if validation data is provided.
  	validation_X                       = nothing,
 	  validation_y                       = nothing,
    validation_weights                 = nothing,
    max_iterations_without_improvement = typemax(Int64)
  );

# Saving:
save_path = tempname()
save(save_path, bin_splits, trees)

# Predicting:
unbinned_predict = load_unbinned_predictor(save_path)
ŷ = unbinned_predict(X)
```

Of course, the goal here is to use all your machine's available memory to hold the training data. If your data is `Float32` it will be binned down to `UInt8`; that is, your machine could hold 4x as much data in memory once that data is binned. BUT you don't want to thrash your machine during data loading.

Here's what to do.

Load a subsample of your data, then compute the bin boundaries using `prepare_bin_splits`.

````julia
X_sample   = ... # Column major, Float32s. (Rows are data points, columns are features)
bin_splits = prepare_bin_splits(X_sample, bin_count = 255)
````

Figure out how much data you have by reading your files on disk. I'm going to hard code this for this example.

```
feature_count         = 872
data_count            = 298349
validation_data_count = 29835
```

Allocate arrays to hold the binned training and validation data.

```julia
X_binned          = Array{UInt8}(undef, (data_count, feature_count))
validation_binned = Array{UInt8}(undef, (validation_data_count, feature_count))
```

Read your `Float32` training data in chunks that are smaller than your machine's memory. For each chunk, bin the data into `UInt8` with `apply_bins` and then copy that into the appropriate place in the binned data.

For example, my data is in several files. Each file is a natural chunk.

```julia
row_i = 1
for file_name in train_file_names
  chunk = load_file(file_name) # Column major, Float32s. (Columns are features.)
  chunk_ndata = size(chunk, 1)
  X_binned[row_i:(row_i + chunk_ndata - 1), :] = apply_bins(chunk, bin_splits)
  row_i += chunk_ndata
end
```

Do the same for your validation data. Now you're ready to set up early stopping and train:

```julia
# Load your labels and weights as Float32s. (Weights are optional.)
y                  = ...
weights            = ...
validation_y       = ...
validation_weights = ...

make_callback() = make_callback_to_track_validation_loss(
  validation_binned,
  validation_y;
  validation_weights = validation_weights,
  max_iterations_without_improvement = 20
)

trees = train_on_binned(
  X_binned, y,
  weights             = weights,
  iteration_callback  = make_callback(),
  iteration_count     = 100000,
  max_delta_score     = 1.8,
  learning_rate       = 0.03,
  feature_fraction    = 1.0,
  bagging_temperature = 0.25
);

# Save:
save_path = tempname()
save(save_path, bin_splits, trees)

# Predict:
test_X = ... # Float32s
unbinned_predict = load_unbinned_predictor(save_path)
ŷ = unbinned_predict(test_X)

# Or:
bin_splits, trees = load(save_path)
test_X_binned = apply_bins(test_X, bin_splits)
ŷ = predict_on_binned(test_X_binned, trees)
```

If early stopping is triggered, the trees returned from `train_on_binned` will *not* include the last `max_iterations_without_improvement` trees; that is, `trees` will be the model that performed best on the validation data.

The callback generated by `make_callback_to_track_validation_loss` will return the current validation loss. You can build on that to write your own callback. For example, if you are doing hyperparameter search, you can only save the best model so far:

```julia
global_best_loss = Inf32

hyperparameter_configs = [
  (:learning_rate => 0.1,   :min_data_weight_in_leaf => 1.0),
  (:learning_rate => 0.05,  :min_data_weight_in_leaf => 10.0),
  (:learning_rate => 0.025, :min_data_weight_in_leaf => 100.0),
]
for config in hyperparameter_configs
  # The original callback is stateful (it keeps track of the validation
  # loss and number of iterations without improvment during training).
  # Regenerate it in this loop so it's fresh for each training round.
  original_callback = make_callback_to_track_validation_loss(
    validation_binned,
    validation_y;
    validation_weights = validation_weights,
    max_iterations_without_improvement = 20
  )

  function my_iteration_callback(trees)
    global global_best_loss

    validation_loss = original_callback(trees)

    if validation_loss < global_best_loss
      save("model_with_loss_$(validation_loss)", bin_splits, trees)
      global_best_loss = validation_loss
    end
  end

  train_on_binned(
    X_binned, y; weights = weights,
    iteration_callback = my_iteration_callback,
    config...
  )
end
```

## Distributed Learning

If your data will not fit in memory on a single machine, you can distribute the data across multiple machines.

Determining bin boundaries still needs to be done on a single node. You might do it in a separate script and save them off:

```julia
using MemoryConstrainedTreeBoosting

# Prepare a sample of your data
X_sample   = ... # Column major, Float32s. (Rows are data points, columns are features)
bin_splits = prepare_bin_splits(X_sample, bin_count = 255)
save("bin_splits", bin_splits, [])

# Load later with...
bin_splits, _ = load("bin_splits")
```

MemoryConstrainedTreeBoosting uses MPI for distributed learning: the same program runs on all machines in the cluster, differing only in the value of a single index variable. This means that distribution is actually fairly simple.

Add [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to your project. In your training script, the MPI "rank" is the process index of the current process in the cluster, from 0 to n-1, for n processes. Rank 0 is the "root" process, traditionally used for the single-machine portions of the computation. The handle for your process group is a "communication group" called `comm`. If you give `comm` to `train_on_binned`, it will use MPI. Only `train_on_binned` and `make_callback_to_track_validation_loss` support MPI.

```julia
using MPI

MPI.Init()
comm       = MPI.COMM_WORLD # The communication group for all processes
root       = 0
rank       = MPI.Comm_rank(comm) # Zero-indexed
rank_count = MPI.Comm_size(comm) # Number of processes in cluster

# Load previously computed bin splits
bin_splits, _ = load("bin_splits")
# *OR* just compute the bin splits on the root process
bin_splits =
	if rank == root
    # Prepare a sample of your data
    X_sample   = ... # Column major, Float32s. (Rows are data points, columns are features)
    bin_splits = prepare_bin_splits(X_sample, bin_count = 255)
  else
  	nothing
  end
# Share bin_splits with all processes
# (not necessary if you used load("bin_splits"), because all processes did that)
bin_splits = MPI.bcast(bin_splits, root, comm)

# Here's a convenience function for determining a process's chunk of a large array.
#
# Usage: chunk_range(rank+1, rank_count, length(my_data))
#
# chunk_range(1, 3, 10) => 1:3
# chunk_range(2, 3, 10) => 4:6
# chunk_range(3, 3, 10) => 7:10
function chunk_range(chunk_i, n_chunks, array_len)
  start = div((chunk_i-1) * array_len, n_chunks) + 1
  stop  = div( chunk_i    * array_len, n_chunks)
  start:stop
end

# Use the rank of the process to load just the process's portion of the data
# (You might use chunk_range above.)
# So the X, validation_X, and y, will differ between processes.
X            = ...
y            = ...
validation_X = ...
validation_y = ...

# Each processes bins its part of the data
# (See example in the last section if you need to load/bin the
# data in streaming manner to avoid maxing out the memory)
X_binned          = apply_bins(X, bin_splits)
validation_binned = apply_bins(validation_X, bin_splits)

# Now just hand the comm to the validation callback and train_on_binned

make_callback() =
  make_callback_to_track_validation_loss(
      validation_X_binned,
      validation_y;
      mpi_comm = comm,
      max_iterations_without_improvement = 20
   	)

trees = train_on_binned(
  X_binned, y,
  weights                 = weights,
  iteration_callback      = make_callback(),
  iteration_count         = 20,
  min_data_weight_in_leaf = 2.0,
  learning_rate           = 0.3,
  mpi_comm                = comm,
);

MPI.Barrier(comm)
if rank == root
  save("trained", bin_splits, trees)
end
```

