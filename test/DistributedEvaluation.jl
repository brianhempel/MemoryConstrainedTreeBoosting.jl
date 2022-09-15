
# echo "import MPI\nMPI.install_mpiexecjl()" | julia --project=..
#
# $ JULIA_NUM_THREADS=4 mpiexecjl -n 3 julia --project=.. DistributedEvaluation.jl
#
# This file is the MPI version of Evaluation.jl, moreso for testing MPI

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MPI
using MemoryConstrainedTreeBoosting

using MPI
MPI.Init()
comm       = MPI.COMM_WORLD
root       = 0
rank       = MPI.Comm_rank(comm)
rank_count = MPI.Comm_size(comm)

print("Hi from rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))!\n")
MPI.Barrier(comm)

function chunk_range(chunk_i, n_chunks, array_len)
  start = div((chunk_i-1) * array_len, n_chunks) + 1
  stop  = div( chunk_i    * array_len, n_chunks)
  start:stop
end


# See https://catboost.ai/#benchmark for comparison.
#
#          CatBoost           LightGBM           XGBoost           H20
#          Tuned    Default   Tuned    Default   Tuned   Default   Tuned   Default
# Adult    0.26974  0.27298   0.27602  0.28716   0.27542 0.28009   0.27510 0.27607
#                   +1.21%    +2.33%   +6.46%    +2.11%  +3.84%    +1.99%  +2.35%

function make_numeric(column)
  # If numeric, pass through. Map ? to 0
  map(column) do val
    val == "?" ? 0f0 : parse(Float32, val)
  end
end

function column_to_numeric_columns_function(column, labels)
  try
    make_numeric(column) # If it fails, we'll catch.
    (column -> make_numeric(column))
  catch
    # If categorical, determine number of categories.
    categories     = unique(column)
    category_count = length(categories)
    if category_count == 2
      # 1 one-hot column
      (column -> Vector{Float32}(column .== categories[1]))
    else
    # elseif category_count == 3
      # One-hot columns
      make_one_hot_columns(column) = begin
        cols =
          map(categories) do category
            Vector{Float32}(column .== category)
          end

        hcat(cols...)
      end
      make_one_hot_columns
    end
  end
end

# Returns (data, labels)
function load_income_file(file_name; column_functions = nothing)
  raw_data = map(line -> split(line, ", "), split(read(joinpath(@__DIR__, "adult_income_dataset", file_name), String), "\n"))

  raw_columns =
    map(1:15) do col_i
      map(row -> row[col_i], raw_data)
    end

  # Last column is label.
  labels = Vector{Float32}(raw_columns[15] .== ">50K")

  if isnothing(column_functions)
    column_functions =
      map(1:14) do col_i
        column_to_numeric_columns_function(raw_columns[col_i], labels)
      end
  end

  col_chunks =
    map(1:14) do col_i
      column_functions[col_i](raw_columns[col_i])
    end

  # for chunk in col_chunks
  #   println(size(chunk))
  # end

  (hcat(col_chunks...), labels, column_functions)
end

X_full, y_full, column_functions        = load_income_file("adult.data")
validation_X_full, validation_y_full, _ = load_income_file("adult.test"; column_functions = column_functions)

Random.seed!(123456)
bin_splits = rank == root ? prepare_bin_splits(X_full) : nothing
bin_splits = MPI.bcast(bin_splits, root, comm)
Random.seed!(123456 + rank)

my_chunk_range            = chunk_range(rank+1, rank_count, length(y_full))
my_validation_chunk_range = chunk_range(rank+1, rank_count, length(validation_y_full))
X            = X_full[my_chunk_range, :]
y            = y_full[my_chunk_range]
validation_X = validation_X_full[my_validation_chunk_range, :]
validation_y = validation_y_full[my_validation_chunk_range]

X_binned            = apply_bins(X, bin_splits)
validation_X_binned = apply_bins(validation_X, bin_splits)

iteration_callback() =
  make_callback_to_track_validation_loss(
      validation_X_binned,
      validation_y;
      mpi_comm = comm,
      max_iterations_without_improvement = 150
    )


duration = @elapsed train_on_binned(
  X_binned, y;
  mpi_comm = comm,
  iteration_count = 10_000,
  learning_rate = 0.01,
  l2_regularization = 10.0,
  feature_fraction = 0.4,
  second_opinion_weight = 0.75,
  normalize_second_opinion = false,
  min_data_weight_in_leaf = 2.9,
  max_delta_score = 5.0,
  max_leaves = 120,
  max_depth = 8,
  bagging_temperature = 0.0,
  iteration_callback = iteration_callback()
)

MPI.Barrier(comm)
if rank == root
  println()
  println("$duration seconds. Expected best validation loss: 0.2807881 (or thereabouts)")
  println()
end
