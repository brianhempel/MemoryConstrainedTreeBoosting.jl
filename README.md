# MemoryConstrainedTreeBoosting.jl

Fast, gradient-boosted decision trees when you want to use all the memory on your machine. Only binary classification with logloss on `Float32` data is supported right now.

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
    weights                 = nothing,
    bin_count               = 255,
    iteration_count         = 100,
    min_data_weight_in_leaf = 10.0,
    l2_regularization       = 1.0,
    max_leaves              = 32,
    max_depth               = 6,
    max_delta_score         = 1.0e10, # Before shrinkage.
    learning_rate           = 0.03,
    feature_fraction        = 1.0, # Per tree.
    bagging_temperature     = 1.0, # Same as Catboost's Bayesian bagging. 0.0 doesn't change the weights. 1.0 samples from the exponential distribution to scale each datapoint's weight.
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
bin_splits = prepare_bin_splits(X_sample, 255)
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





